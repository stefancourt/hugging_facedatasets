---
configs:
- config_name: ConvFinQA
  data_files:
  - split: train
    path: train_turn.json
  - split: validation
    path: dev_turn.json
task_categories:
- text-classification
- question-answering
- zero-shot-classification
language:
- en
tags:
- finance
---

# Adapting Large Language Models to Domains via Continual Pre-Training
This repo contains the **ConvFinQA dataset** used in our **ICLR 2024** paper [Adapting Large Language Models via Reading Comprehension](https://huggingface.co/papers/2309.09530).

We explore **continued pre-training on domain-specific corpora** for large language models. While this approach enriches LLMs with domain knowledge, it significantly hurts their prompting ability for question answering. Inspired by human learning via reading comprehension, we propose a simple method to **transform large-scale pre-training corpora into reading comprehension texts**, consistently improving prompting performance across tasks in biomedicine, finance, and law domains. **Our 7B model competes with much larger domain-specific models like BloombergGPT-50B**. 

### [2024/11/29] 🤗 Introduce the multimodal version of AdaptLLM at [AdaMLLM](https://huggingface.co/AdaptLLM/Adapt-MLLM-to-Domains), for adapting MLLMs to domains 🤗

**************************** **Updates** ****************************
* 2024/11/29: Released [AdaMLLM](https://huggingface.co/AdaptLLM/Adapt-MLLM-to-Domains) for adapting MLLMs to domains
* 2024/9/20: Our [research paper for Instruction-Pretrain](https://huggingface.co/papers/2406.14491) has been accepted by EMNLP 2024
* 2024/8/29: Updated [guidelines](https://huggingface.co/datasets/AdaptLLM/finance-tasks) on evaluating any 🤗Huggingface models on the domain-specific tasks
* 2024/6/22: Released the [benchmarking code](https://github.com/microsoft/LMOps/tree/main/adaptllm)
* 2024/6/21: Released the general version of AdaptLLM at [Instruction-Pretrain](https://huggingface.co/instruction-pretrain)
* 2024/4/2: Released the [raw data splits (train and test)](https://huggingface.co/datasets/AdaptLLM/ConvFinQA) of all the evaluation datasets
* 2024/1/16: Our [research paper for AdaptLLM](https://huggingface.co/papers/2309.09530) has been accepted by ICLR 2024
* 2023/12/19: Released our [13B base models](https://huggingface.co/AdaptLLM/law-LLM-13B) developed from LLaMA-1-13B
* 2023/12/8: Released our [chat models](https://huggingface.co/AdaptLLM/law-chat) developed from LLaMA-2-Chat-7B
* 2023/9/18: Released our [paper](https://huggingface.co/papers/2309.09530), [code](https://github.com/microsoft/LMOps), [data](https://huggingface.co/datasets/AdaptLLM/law-tasks), and [base models](https://huggingface.co/AdaptLLM/law-LLM) developed from LLaMA-1-7B


## Domain-Specific LLaMA-1
### LLaMA-1-7B
In our paper, we develop three domain-specific models from LLaMA-1-7B, which are also available in Huggingface: [Biomedicine-LLM](https://huggingface.co/AdaptLLM/medicine-LLM), [Finance-LLM](https://huggingface.co/AdaptLLM/finance-LLM) and [Law-LLM](https://huggingface.co/AdaptLLM/law-LLM), the performances of our AdaptLLM compared to other domain-specific LLMs are:

<p align='center'>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/650801ced5578ef7e20b33d4/6efPwitFgy-pLTzvccdcP.png" width="700">
</p>

### LLaMA-1-13B
Moreover, we scale up our base model to LLaMA-1-13B to see if **our method is similarly effective for larger-scale models**, and the results are consistently positive too: [Biomedicine-LLM-13B](https://huggingface.co/AdaptLLM/medicine-LLM-13B), [Finance-LLM-13B](https://huggingface.co/AdaptLLM/finance-LLM-13B) and [Law-LLM-13B](https://huggingface.co/AdaptLLM/law-LLM-13B).

## Domain-Specific LLaMA-2-Chat
Our method is also effective for aligned models! LLaMA-2-Chat requires a [specific data format](https://huggingface.co/blog/llama2#how-to-prompt-llama-2), and our **reading comprehension can perfectly fit the data format** by transforming the reading comprehension into a multi-turn conversation. We have also open-sourced chat models in different domains: [Biomedicine-Chat](https://huggingface.co/AdaptLLM/medicine-chat), [Finance-Chat](https://huggingface.co/AdaptLLM/finance-chat) and [Law-Chat](https://huggingface.co/AdaptLLM/law-chat)

## Domain-Specific Tasks

### Pre-templatized/Formatted Testing Splits
To easily reproduce our prompting results, we have uploaded the filled-in zero/few-shot input instructions and output completions of the test each domain-specific task: [biomedicine-tasks](https://huggingface.co/datasets/AdaptLLM/medicine-tasks), [finance-tasks](https://huggingface.co/datasets/AdaptLLM/finance-tasks), and [law-tasks](https://huggingface.co/datasets/AdaptLLM/law-tasks).

**Note:** those filled-in instructions are specifically tailored for models before alignment and do NOT fit for the specific data format required for chat models.

### Raw Datasets
We have also uploaded the raw training and testing splits, for facilitating fine-tuning or other usages: 
- [ChemProt](https://huggingface.co/datasets/AdaptLLM/ChemProt)
- [RCT](https://huggingface.co/datasets/AdaptLLM/RCT)
- [ConvFinQA](https://huggingface.co/datasets/AdaptLLM/ConvFinQA)
- [FiQA_SA](https://huggingface.co/datasets/AdaptLLM/FiQA_SA)
- [Headline](https://huggingface.co/datasets/AdaptLLM/Headline)
- [NER](https://huggingface.co/datasets/AdaptLLM/NER)
- [FPB](https://huggingface.co/datasets/AdaptLLM/FPB)

The other datasets used in our paper have already been available in huggingface, and you can directly load them with the following code:
```python
from datasets import load_dataset

# MQP:
dataset = load_dataset('medical_questions_pairs')
# PubmedQA:
dataset = load_dataset('bigbio/pubmed_qa')
# USMLE:
dataset=load_dataset('GBaker/MedQA-USMLE-4-options')
# SCOTUS
dataset = load_dataset("lex_glue", 'scotus')
# CaseHOLD
dataset = load_dataset("lex_glue", 'case_hold')
# UNFAIR-ToS
dataset = load_dataset("lex_glue", 'unfair_tos')
```

## Citation
If you find our work helpful, please cite us:
```bibtex
@inproceedings{
cheng2024adapting,
title={Adapting Large Language Models via Reading Comprehension},
author={Daixuan Cheng and Shaohan Huang and Furu Wei},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=y886UXPEZ0}
}
```

and the original dataset:
```bibtex
@inproceedings{ConvFinQA,
  author       = {Zhiyu Chen and
                  Shiyang Li and
                  Charese Smiley and
                  Zhiqiang Ma and
                  Sameena Shah and
                  William Yang Wang},
  title        = {ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational
                  Finance Question Answering},
  booktitle    = {{EMNLP}},
  pages        = {6279--6292},
  publisher    = {Association for Computational Linguistics},
  year         = {2022}
}
```