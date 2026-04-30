---
license: cc-by-nc-4.0
dataset_info:
  features:
  - name: ID
    dtype: string
  - name: split
    dtype: string
  - name: knowledge concept
    dtype: string
  - name: question
    dtype: string
  - name: option
    dtype: string
  - name: answer
    dtype: string
  - name: image_path
    dtype: image
  - name: key
    dtype: string
  - name: question number
    dtype: int64
  - name: knowledge concept description
    dtype: string
  splits:
  - name: testmini
    num_bytes: 44509869
    num_examples: 1740
  download_size: 23075805
  dataset_size: 44509869
task_categories:
- question-answering
- text-generation
language:
- en
tags:
- LLM
- NLP
- CV
size_categories:
- 1K<n<10K
---



# Dataset Card for WE-MATH (ACL 2025)

[GitHub](https://github.com/We-Math/We-Math) | [Paper](https://arxiv.org/pdf/2407.01284) | [Website](https://we-math.github.io/)

Inspired by human-like mathematical reasoning, we introduce We-Math, the first benchmark specifically designed to explore the problem-solving principles beyond the end-to-end performance. We meticulously collect and categorize 6.5K visual math problems, spanning 67 hierarchical knowledge concepts and 5 layers of knowledge granularity.

## Citation
If you find the content of this project helpful, please cite our paper as follows:


```
@article{qiao2024we,
  title={We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?},
  author={Qiao, Runqi and Tan, Qiuna and Dong, Guanting and Wu, Minhui and Sun, Chong and Song, Xiaoshuai and GongQue, Zhuoma and Lei, Shanglin and Wei, Zhe and Zhang, Miaoxuan and others},
  journal={arXiv preprint arXiv:2407.01284},
  year={2024}
}
```