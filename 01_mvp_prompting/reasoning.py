import sys
import os
import time
import numpy as np
import json
import threading
import subprocess
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logging.txt"),
        logging.StreamHandler()
    ],
    force=True
)
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

if True:
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")))
    fixed = True

# Path to absa-toolkit (https://github.com/NilsHellwig/absa-toolkit)
TOOLKIT_PATH = os.environ.get("ABSA_TOOLKIT_PATH", os.path.expanduser("~/absa-toolkit"))
sys.path.append(TOOLKIT_PATH)
if TOOLKIT_PATH:
    from helper import *


def initialize_llm(model_name):
    """Initialize the LLM model."""
    setup_gpu_environment()

    if "gemma" in model_name:
        llm = LLM(
            model=model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization="bitsandbytes",
            max_model_len=16384,
            max_num_seqs=512,
            seed=0
        )
    else:
        llm = LLM(
            model=model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            max_model_len=16384,
            max_num_seqs=512,
            seed=0
        )

    return llm


# store valid phrases globally to avoid recomputation
dict_valid_phrases = {}


def create_structured_outputs_params(data, unique_aspect_categories, polarities, considered_aspects):
    """Create structured output parameters for each example."""
    structured_outputs_params = []

    for example in data:
        if example["text"] in dict_valid_phrases:
            valid_phrases = dict_valid_phrases[example["text"]]
        else:
            valid_phrases = find_valid_phrases_list(example["text"], 64)
            dict_valid_phrases[example["text"]] = valid_phrases
        regex_pattern = get_regex_pattern_tuple(
            unique_aspect_categories, polarities, considered_aspects, valid_phrases)

        structured_outputs_params.append(
            StructuredOutputsParams(regex=regex_pattern)
        )

    return structured_outputs_params


def generate_test_predictions(llm, all_prompts, all_structured_params, sampling_strategy, seed=0):
    logging.info("Generating test predictions...")
    sampling_params_batch = []
    for structured_params in all_structured_params:
        sampling_params_batch.append(
            SamplingParams(
                temperature=0.0,
                max_tokens=1024,
                structured_outputs=structured_params,
                logprobs=10 if sampling_strategy not in ["single", "sc", "reasoning"] else None,
                seed=seed,
            )
        )

    all_outputs = llm.generate(
        prompts=all_prompts,
        sampling_params=sampling_params_batch,
    )
    logging.info("Finished generating test predictions.")

    return all_outputs


def shuffle_examples(examples, seed):
    """Shuffle few-shot examples based on the given strategy."""
    np.random.seed(seed)
    # Create a copy to avoid mutating the original list
    examples_copy = examples.copy()
    np.random.shuffle(examples_copy)
    # Additional strategies can be implemented here
    return examples_copy


def main():
    evaluation_mode = "performance" # performance or efficient
    sampling_strategies = ["reasoning"]  # Can be extended "sc", "mvp", "reasoning", "single"
    if "reasoning" in sampling_strategies:
        llm_name = "openai/gpt-oss-120b"
    else:
        llm_name = "google/gemma-3-27b-it"

    # Configuration from run_script.py
    datasets = ["rest16", "rest15", "coursera", "hotels", "flightabsa"]
    tasks = ["tasd", "asqp"]
    n_shots_list = [100, 10, 50, 0]
    n_seeds_single = 1
    n_seeds_sc = 5
    n_seeds_reasoning = 1
    n_seeds_run = 5 if evaluation_mode == "performance" else 1  # Picks of few-shot examples to use for each configuration

    # Initialize LLM and Embedding model once
    clear_memory()
    setup_gpu_environment()
    llm = initialize_llm(llm_name)

    # Process all configurations
    for n_shots in n_shots_list:
        for dataset_name in datasets:
            for task in tasks:
                for sampling_strategy in sampling_strategies:
                    logging.info(f"Evaluating {dataset_name}_{task}_{sampling_strategy}_{n_shots}")
                    # continue if already computed
                    file_path = f"results/prompting{"" if evaluation_mode == "performance" else "_efficient"}/{llm_name.replace('/', '_')}/{dataset_name}_{task}_{sampling_strategy}_{n_shots}_0_0.json"
                    if os.path.exists(file_path):
                        logging.info(
                            f"Skipping {dataset_name}_{task}_{sampling_strategy}_{n_shots} as results already exist.")
                        continue

                    prompts = []
                    structured_output_params = []
                    cond = []

                    polarities_list = ["positive", "negative", "neutral"]
                    unique_aspect_categories_list = get_unique_aspect_categories(
                        dataset_name, task, base_path=TOOLKIT_PATH + "/data")
                    test_data = get_dataset(
                        dataset_name, split="test", task=task, base_path=TOOLKIT_PATH + "/data")

                    considered_sentiment_elements_list = get_considered_aspects(
                        task)
                    considered_sentiment_elements_combinations = get_all_combinations(
                        considered_sentiment_elements_list)

                    for seed_run in range(n_seeds_run):

                        few_shot_examples = get_fs_examples_new_with_seed(
                            dataset_name,
                            task,
                            n_shots,
                            seed=seed_run,
                            use_dev=False
                        )

                        n_seeds_strategy = {"single": n_seeds_single, "sc": n_seeds_sc, "reasoning": n_seeds_reasoning, "mvp": len(
                            # Number of random seeds per strategy
                            considered_sentiment_elements_combinations)}

                        for idx_seed_random, seed_random in enumerate(range(n_seeds_strategy[sampling_strategy])):

                            if sampling_strategy == "mvp":
                                sentiment_elements = considered_sentiment_elements_combinations[
                                    idx_seed_random]
                            elif sampling_strategy == "single" or sampling_strategy == "sc" or sampling_strategy == "reasoning":
                                sentiment_elements = considered_sentiment_elements_list

                            few_shot_examples_shuffled = shuffle_examples(
                                few_shot_examples,
                                seed=seed_random
                            )
                            for example in test_data:
                                prompt = get_prompt(
                                    dataset_name,
                                    task,
                                    text_pred=example["text"],
                                    examples=few_shot_examples_shuffled,
                                    unique_aspect_categories=unique_aspect_categories_list,
                                    polarities=polarities_list,
                                    approach="tuple",
                                    considered_aspects=sentiment_elements,
                                )

                                if sampling_strategy == "reasoning":
                                    prompt = """<|start|>system<|message|>
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-11-15

Reasoning: medium

# Valid channels: analysis, final.
<|end|>

<|start|>user<|message|>""" + prompt + """\n<|end|>

<|start|>assistant<|channel|>analysis<|message|>\n"""

                                prompts.append(prompt)

                                structured_output_params.append(
                                    create_structured_outputs_params(
                                        [example],
                                        unique_aspect_categories=unique_aspect_categories_list,
                                        polarities=polarities_list,
                                        considered_aspects=sentiment_elements,
                                    )[0]
                                )

                                cond.append({
                                    "seed_run": seed_run,
                                    "seed_random": seed_random,
                                })
                    # execute generation
                    time_start = time.time()
                    # start watt monitoring
                    monitoring = True
                    watts = []

                    def monitor_watts():
                        while monitoring:
                            result = subprocess.run(
                                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                            if result.returncode == 0:
                                try:
                                    power = float(result.stdout.strip())
                                    watts.append(power)
                                except ValueError:
                                    pass
                            time.sleep(0.1)
                    thread = threading.Thread(target=monitor_watts)
                    thread.start()
                    logging.info("Start generation")
                    if prompts:
                        logging.info(f"First prompt: {prompts[0]}")
                    outputs = generate_test_predictions(
                        llm,
                        prompts,
                        structured_output_params,
                        sampling_strategy,
                        seed=0
                    )
                    if outputs:
                        logging.info(f"First prediction: {outputs[0].outputs[0].text}")
                    # end watt monitoring
                    monitoring = False
                    thread.join()
                    avg_watt = sum(watts) / len(watts) if watts else 0
                    total_time = time.time() - time_start

                    # store as json
                    for i in range(len(outputs)):
                        output_text = outputs[i].outputs[0].text
                        if sampling_strategy == "reasoning":
                            # Extract final answer after 'assistantfinal' marker
                            if "assistantfinal" in output_text:
                                output_text = output_text.split(
                                    "assistantfinal")[1].strip()
                            else:
                                output_text = "[]"
                        
                        outputs_item = {
                            "output": output_text,
                            "metadata": {
                                "seed_run": cond[i]["seed_run"],
                                "seed_random": cond[i]["seed_random"],
                            }
                        }
                        if sampling_strategy not in ["single", "sc", "reasoning"]:
                            outputs_item["logprobs"] = convert_to_list_of_probs_for_each_token(outputs[i].outputs[0].logprobs)
                        
                        outputs[i] = outputs_item

                    # Group outputs by seed_run and seed_random
                    grouped_outputs = {}
                    for output in outputs:
                        seed_run = output["metadata"]["seed_run"]
                        seed_random = output["metadata"]["seed_random"]
                        key = (seed_run, seed_random)
                        if key not in grouped_outputs:
                            grouped_outputs[key] = []
                        grouped_outputs[key].append(output)

                    # Save each group to a separate file
                    for (seed_run, seed_random), group_outputs in grouped_outputs.items():
                        file_path = f"results/prompting{"" if evaluation_mode == "performance" else "_efficient"}/{llm_name.replace('/', '_')}/{dataset_name}_{task}_{sampling_strategy}_{n_shots}_{seed_random}_{seed_run}.json"
                        os.makedirs(os.path.dirname(
                            file_path), exist_ok=True)
                        
                        # convert group_outputs in this format: {"preds": [...list of string predictions...], "logprobs": [...list of list of floats...], "metadata": [...]}
                        preds = [output["output"] for output in group_outputs]
                        metadata = [output["metadata"] for output in group_outputs]
                        
                        final_outputs = {
                            "preds": preds,
                        }
                        if sampling_strategy not in ["single", "sc", "reasoning"]:
                            final_outputs["logprobs"] = [output["logprobs"] for output in group_outputs]

                        # Save results to a JSON file
                        with open(file_path, "w") as f:
                            sentiment_elements = considered_sentiment_elements_combinations[
                                seed_random] if sampling_strategy == "mvp" else considered_sentiment_elements_list
                            json.dump(
                                {"preds": final_outputs, "considered_sentiment_elements": sentiment_elements, "avg_time_per_example": total_time / len(outputs), "avg_watt": avg_watt, "metadata": metadata}, f, indent=4)


if __name__ == "__main__":
    main()
