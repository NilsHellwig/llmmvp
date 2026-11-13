<div align="center">

# Prompting Is All You Need: Multi-view Prompting Large Language Models for Aspect-Based Sentiment Analysis

Accepted at **Findings of the Association for Computational Linguistics: EACL 2027**<br>
Athens, Greece · March 9–14, 2027

[![Venue](https://img.shields.io/badge/EACL%202027-Findings-1f4e79)](https://2027.eacl.org/)
[![Paper](https://img.shields.io/badge/Paper-TBA-lightgrey)](#citation)
[![Toolkit](https://img.shields.io/badge/Data%20%26%20Prompts-absa--toolkit-2e7d32)](https://github.com/NilsHellwig/absa-toolkit)

**Nils Constantin Hellwig**<sup>1</sup> · **Niklas Donhauser**<sup>1</sup> · **Jakob Fehle**<sup>1</sup> · **Udo Kruschwitz**<sup>2</sup> · **Christian Wolff**<sup>1</sup>

<sup>1</sup>Media Informatics Group, University of Regensburg, Germany<br>
<sup>2</sup>Information Science Group, University of Regensburg, Germany

</div>

---

## 📄 Abstract

Recent work explored the capabilities of Large Language Models (LLMs) in Aspect-Based Sentiment Analysis (ABSA) through few-shot prompting, requiring substantially fewer annotated examples while achieving notable improvements over zero-shot baselines. However, a performance gap remained compared to models fine-tuned on hundreds of examples, and the computational costs of LLM inference present practical barriers to deployment. We introduce **LLM-based Multi-view Prompting (LLM-MvP)**, which adapts the multi-view principle of considering multiple element orderings to LLM prompting. By combining schema-constrained decoding with prefix batching, LLM-MvP achieves performance competitive with or superior to fine-tuned approaches while substantially reducing computational overhead. Extensive experiments across five benchmark datasets demonstrate that LLM-MvP closes the gap between few-shot prompting and fine-tuned models, offering a practical and efficient solution for ABSA.

## 🧭 Method

LLM-MvP prompts an LLM once for every permutation of the sentiment elements in a tuple (6 for TASD, 24 for ASQP). For each output, the average token-level entropy is computed, and the *m* = 5 lowest-entropy outputs are aggregated via majority voting. Aspect and opinion terms, aspect categories, and polarities are constrained to valid values through schema-constrained decoding (XGrammar in vLLM), and all permutations share a cached prompt prefix, which keeps the cost of multi-view inference manageable.

We evaluate on **TASD** (aspect term, category, polarity) and **ASQP** (aspect term, category, opinion term, polarity) across five datasets: Rest15, Rest16, FlightABSA, OATS Coursera, and OATS Hotels, in 0-, 10-, 50-, and 100-shot settings with five independently sampled sets of demonstrations each.

## 📁 Repository Contents

```plaintext
.
├── 01_mvp_prompting/
│   ├── predict.py      # LLM-MvP, single-order, self-consistency, and w/o guided decoding runs
│   └── reasoning.py    # Reasoning baseline (gpt-oss-120B)
├── results/
│   └── scores/         # Precision, recall, micro-F1, and macro-F1 for all models and conditions
├── requirements.txt
└── README.md
```

## ⚙️ Setup

Datasets, few-shot examples, prompt construction, and evaluation helpers are provided by the publicly available [**absa-toolkit**](https://github.com/NilsHellwig/absa-toolkit), which the scripts import at runtime.

```bash
git clone https://github.com/NilsHellwig/llmmvp.git
cd llmmvp
pip install -r requirements.txt

# absa-toolkit (datasets + helpers); default location is ~/absa-toolkit
git clone https://github.com/NilsHellwig/absa-toolkit.git ~/absa-toolkit
# or point the scripts to another location:
export ABSA_TOOLKIT_PATH=/path/to/absa-toolkit
```

All experiments were run with vLLM 0.20.0 on a single NVIDIA RTX PRO 6000 Blackwell GPU (96 GB).

## ▶️ Usage

Run the scripts from the repository root; predictions are written to `results/prompting/<model>/`.

```bash
# LLM-MvP (all permutations) and self-consistency with Gemma 4 (31B)
python 01_mvp_prompting/predict.py

# Reasoning baseline with gpt-oss-120B
python 01_mvp_prompting/reasoning.py
```

The experimental configuration is set at the bottom of each script:

| Setting | `predict.py` | `reasoning.py` |
|---|---|---|
| Model | `LLM_NAME` | `llm_name` |
| Approaches | `approaches` (`llm-mvp`, `no-guided`, `sc`) | `sampling_strategies` |
| Datasets / tasks / shots / seeds | `datasets`, `tasks`, `shots`, `seeds` | `datasets`, `tasks`, `n_shots_list` |
| Prefix caching | `ENABLE_PREFIX_CACHING` | – |

Single-order prompting corresponds to the first (default) permutation of the LLM-MvP runs.

## 📊 Scores

`results/scores/` contains one file per metric and model, e.g. `f1_scores_google_gemma-4-31B.json`, structured as

```text
{dataset: {task: {method: {n_shots: [score per few-shot sample (5 runs)]}}}}
```

The paper reports the mean over the five runs. Method keys include `llm-mvp`, `single`, `sc`, `no-guided` (LLM-MvP without guided decoding), and `llm-mvp-eff`; suffixes `-m-<k>` denote majority voting over the *k* lowest-entropy permutations.

## 📝 Citation

The BibTeX entry will be added once the paper is available in the ACL Anthology.

```bibtex
TBA
```

## ✉️ Contact

Nils Constantin Hellwig · [nils-constantin.hellwig@ur.de](mailto:nils-constantin.hellwig@ur.de)
