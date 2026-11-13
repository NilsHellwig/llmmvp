import sys
import os
import json
import itertools
import numpy as np
import pandas as pd
from collections import Counter
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.config import StructuredOutputsConfig

# Path to absa-toolkit (https://github.com/NilsHellwig/absa-toolkit)
TOOLKIT_PATH = os.environ.get("ABSA_TOOLKIT_PATH", os.path.expanduser("~/absa-toolkit"))
sys.path.append(TOOLKIT_PATH)

if TOOLKIT_PATH and sys:
    from helper import *
    from gpu_monitor import GPUMonitor


def format_logprobs(logprobs_list):
    """Formatiert Logprobs effizient: {Token: gerundeter_Logprob} pro Position."""
    formatted = []
    for lp_dict in logprobs_list:
        pos_data = {}
        for token_id, logprob_obj in lp_dict.items():
            token_text = logprob_obj.decoded_token if hasattr(
                logprob_obj, 'decoded_token') else str(token_id)
            val = float(logprob_obj.logprob)
            if np.isinf(val):
                pos_data[token_text] = "-Infinity" if val < 0 else "Infinity"
            else:
                pos_data[token_text] = round(val, 3)
        formatted.append(pos_data)
    return formatted


def run_experiment(llm, dataset_name, task, n_shot, seeds, active_approaches, subset="test"):
    print(
        f"\n--- Starte Experiment: {dataset_name} | {task} | {n_shot}-shot ---")

    # Vorbereitung
    safe_llm_name = llm.llm_engine.model_config.model.replace("/", "_")
    output_dir = f"./results/prompting/{safe_llm_name}/"
    os.makedirs(output_dir, exist_ok=True)

    considered_sentiment_elements = considered_aspects_tasks[task]
    test_dataset = get_dataset(
        dataset_name, subset, task, base_path=TOOLKIT_PATH + "/data")
    unique_aspect_categories = get_unique_aspect_categories(
        dataset_name, task, base_path=TOOLKIT_PATH + "/data")

    sentiment_elements_list = considered_aspects_tasks[task]
    # Alle Permutationen generieren, die Standard-Permutation ist index 0
    all_perms = list(itertools.permutations(sentiment_elements_list))
    if ENABLE_PREFIX_CACHING == False:
        if "llm-mvp" in active_approaches:
            all_perms = [all_perms[0]]

    # Cache valid phrases
    valid_phrases_cache = {}
    for idx, example in enumerate(test_dataset):
        vp = find_valid_phrases_list(example["text"], MAX_CHARS_PHRASE)
        valid_phrases_cache[idx] = [p for p in vp if not p.startswith(
            (",", ".", " ", ")", "(", "'")) and not p.endswith((",", ".", " ", ")", "(", "'"))]

    if any(app in active_approaches for app in ["llm-mvp", "no-guided"]):
        for app in [a for a in ["llm-mvp", "no-guided"] if a in active_approaches]:
            for p_idx, perm in enumerate(all_perms):
                # Check if results exist for all seeds
                if all(os.path.exists(os.path.join(output_dir, f"{dataset_name}_{subset}_{task}_{n_shot}_{seed}_{safe_llm_name}_{app}_{p_idx}{enable_prefix_caching_prefix}.txt")) for seed in seeds):
                    print(f"  Skipping {app} | Permutation {p_idx} (results exist)")
                    continue

                prompts, params, info = [], [], []
                for seed in seeds:
                    few_shot = get_fs_examples_new_with_seed(dataset_name, task, n_shot, seed)
                    for idx, example in enumerate(test_dataset):
                        p = get_prompt(dataset_name, task, example["text"], few_shot, unique_aspect_categories, considered_aspects=list(perm))
                        
                        s_params = {"temperature": 0, "max_tokens": 512, "logprobs": 5}
                        if app == "llm-mvp":
                            regex = get_regex_pattern_tuple(unique_aspect_categories, POLARITIES, list(perm), valid_phrases_cache[idx])
                            s_params["structured_outputs"] = StructuredOutputsParams(regex=regex)
                        if app == "no-guided":  # no-guided
                            p += "[('"
                            s_params["stop"] = ")]"
                        
                        prompts.append(p)
                        params.append(SamplingParams(**s_params))
                        info.append((seed, idx))

                print(f"  Approach {app} | Permutation {p_idx}/{len(all_perms)-1}: {len(prompts)} prompts...")
                monitor = GPUMonitor(); monitor.start()
                outputs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
                mean_watt, duration = monitor.stop()

                for seed in seeds:
                    base_fn = f"{dataset_name}_{subset}_{task}_{n_shot}_{seed}_{safe_llm_name}_{app}_{p_idx}"
                    # TXT
                    with open(os.path.join(output_dir, f"{base_fn}{enable_prefix_caching_prefix}.txt"), "w") as f:
                        for target_idx in range(len(test_dataset)):
                            out = next(o for i, o in enumerate(outputs) if info[i] == (seed, target_idx))
                            text = out.outputs[0].text.replace("\n", " ")
                            if app == "no-guided": 
                                text = "[('" + text + ")]"
                            f.write(text + "\n")
                    # JSON
                    print(outputs[0].outputs[0].text)
                    # store prompt exemplar
                    
                    with open(os.path.join(output_dir, f"{base_fn}{enable_prefix_caching_prefix}.json"), "w") as f:
                        seed_outs = [next(o for i, o in enumerate(outputs) if info[i] == (seed, target_idx)) for target_idx in range(len(test_dataset))]
                        json.dump({
                            "logprobs": [format_logprobs(out.outputs[0].logprobs) for out in seed_outs],
                            "gpu_stats": {"mean_watt": mean_watt, "duration_sec": duration, "energy_joule": mean_watt * duration}
                        }, f)

    # --- PHASE 2.5: SC ---
    if "sc" in active_approaches:
        # Check if SC results already exist (indicator: seed 0, execution 0)
        N_SC = len(all_perms)
        if ENABLE_PREFIX_CACHING == False:
            N_SC = 1
        prompts = []
        params = []
        info = []
        for seed in seeds:
            few_shot = get_fs_examples_new_with_seed(
                dataset_name, task, n_shot, seed)
            for idx, example in enumerate(test_dataset):
                p = get_prompt(
                    dataset_name, task, example["text"], few_shot, unique_aspect_categories)
                regex = get_regex_pattern_tuple(
                    unique_aspect_categories, POLARITIES, considered_sentiment_elements, valid_phrases_cache[idx])
                prompts.append(p)
                params.append(SamplingParams(temperature=TEMP_SC, max_tokens=512, n=N_SC,
                              logprobs=1, structured_outputs=StructuredOutputsParams(regex=regex)))
                info.append((seed, idx))

        print(f"Generating SC for {len(prompts)} prompts...")
        monitor = GPUMonitor()
        monitor.start()
        outputs = llm.generate(prompts, sampling_params=params, use_tqdm=True)
        mean_watt, duration = monitor.stop()

        for seed in seeds:
            for n in range(N_SC):
                base_fn = f"{dataset_name}_{subset}_{task}_{n_shot}_{seed}_{safe_llm_name}_sc_{n}"
                with open(os.path.join(output_dir, f"{base_fn}{enable_prefix_caching_prefix}.txt"), "w") as f:
                    for target_idx in range(len(test_dataset)):
                        out = next(o for i, o in enumerate(outputs)
                                   if info[i] == (seed, target_idx))
                        f.write(out.outputs[n].text.replace("\n", " ") + "\n")
                with open(os.path.join(output_dir, f"{base_fn}{enable_prefix_caching_prefix}.json"), "w") as f:
                    seed_sc_outs_by_order = [next(o for i, o in enumerate(outputs) if info[i] == (
                        seed, target_idx)) for target_idx in range(len(test_dataset))]
                    data = {
                        "logprobs": [format_logprobs(out.outputs[n].logprobs) if out.outputs[n].logprobs else [] for out in seed_sc_outs_by_order],
                        "gpu_stats": {
                            "mean_watt": mean_watt,
                            "duration_sec": duration,
                            "energy_joule": mean_watt * duration
                        }
                    }
                    json.dump(data, f)


TEMP_SC = 0.8
LLM_NAME = "google/gemma-4-31B"
ENABLE_PREFIX_CACHING = True
enable_prefix_caching_prefix = "" if ENABLE_PREFIX_CACHING else "_no-caching"
MAX_CHARS_PHRASE = 64
POLARITIES = ["positive", "negative", "neutral"]

if __name__ == "__main__":
    llm = LLM(
        model=LLM_NAME,
        max_model_len=8192*2,
        dtype="bfloat16",
        quantization="bitsandbytes",
        trust_remote_code=True,
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
        max_num_seqs=512,
    )

    tasks = ["tasd", "asqp"]
    datasets = ["rest16", "rest15", "flightabsa", "hotels", "coursera"]
    shots = [100, 10, 50, 0]
    seeds = [0, 1, 2, 3, 4]
    approaches = ["llm-mvp", "sc"]

    for task in tasks:
        for ds in datasets:
            for n in shots:
                try:
                    run_experiment(llm, ds, task, n, seeds, approaches)
                except Exception as e:
                    print(f"Fehler bei {ds} {task} {n}-shot: {e}")
