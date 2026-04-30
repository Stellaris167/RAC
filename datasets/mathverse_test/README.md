---
task_categories:
- multiple-choice
- question-answering
- visual-question-answering
language:
- en
size_categories:
- 1K<n<10K
configs:
- config_name: testmini
  data_files:
  - split: testmini
    path: testmini.parquet
- config_name: testmini_text_only
  data_files:
  - split: testmini_text_only
    path: testmini_text_only.parquet
dataset_info:
- config_name: testmini
  features:
  - name: sample_index
    dtype: string
  - name: problem_index
    dtype: string
  - name: problem_version
    dtype: string
  - name: question
    dtype: string
  - name: image
    dtype: image
  - name: answer
    dtype: string
  - name: question_type
    dtype: string
  - name: metadata
    struct:
    - name: split
      dtype: string
    - name: source
      dtype: string
    - name: subject
      dtype: string
    - name: subfield
      dtype: string
  - name: query_wo
    dtype: string
  - name: query_cot
    dtype: string
  - name: question_for_eval
    dtype: string
  splits:
  - name: testmini
    num_bytes: 166789963
    num_examples: 3940
- config_name: testmini_text_only
  features:
  - name: sample_index
    dtype: string
  - name: problem_index
    dtype: string
  - name: problem_version
    dtype: string
  - name: question
    dtype: string
  - name: image
    dtype: string
  - name: answer
    dtype: string
  - name: question_type
    dtype: string
  - name: metadata
    struct:
    - name: split
      dtype: string
    - name: source
      dtype: string
    - name: subject
      dtype: string
    - name: subfield
      dtype: string
  - name: query_wo
    dtype: string
  - name: query_cot
    dtype: string
  - name: question_for_eval
    dtype: string
  splits:
  - name: testmini_text_only
    num_bytes: 250959
    num_examples: 788
license: mit
---
# Dataset Card for MathVerse

- [Dataset Description](https://huggingface.co/datasets/AI4Math/MathVerse/blob/main/README.md#dataset-description)
- [Paper Information](https://huggingface.co/datasets/AI4Math/MathVerse/blob/main/README.md#paper-information)
- [Dataset Examples](https://huggingface.co/datasets/AI4Math/MathVerse/blob/main/README.md#dataset-examples)
- [Leaderboard](https://huggingface.co/datasets/AI4Math/MathVerse/blob/main/README.md#leaderboard)
- [Citation](https://huggingface.co/datasets/AI4Math/MathVerse/blob/main/README.md#citation)

## Dataset Description
The capabilities of **Multi-modal Large Language Models (MLLMs)** in **visual math problem-solving** remain insufficiently evaluated and understood. We investigate current benchmarks to incorporate excessive visual content within textual questions, which potentially assist MLLMs in deducing answers without truly interpreting the input diagrams.

<p align="center">
    <img src="https://raw.githubusercontent.com/ZrrSkywalker/MathVerse/main/figs/fig1.png" width="90%"> <br>
</p>

To this end, we introduce **MathVerse**, an all-around visual math benchmark designed for an equitable and in-depth evaluation of MLLMs. We meticulously collect 2,612 high-quality, multi-subject math problems with diagrams from publicly available sources. Each problem is then transformed by human annotators into **six distinct versions**, each offering varying degrees of information content in multi-modality, contributing to **15K** test samples in total. This approach allows MathVerse to comprehensively assess ***whether and how much MLLMs can truly understand the visual diagrams for mathematical reasoning.*** 

<p align="center">
    <img src="https://raw.githubusercontent.com/ZrrSkywalker/MathVerse/main/figs/fig2.png" width="90%"> <br>
    Six different versions of each problem in <b>MathVerse</b> transformed by expert annotators.
</p>

In addition, we propose a **Chain-of-Thought (CoT) Evaluation strategy** for a fine-grained assessment of the output answers. Rather than naively judging True or False, we employ GPT-4(V) to adaptively extract crucial reasoning steps, and then score each step with detailed error analysis, which can reveal the intermediate CoT reasoning quality by MLLMs.

<p align="center">
    <img src="https://raw.githubusercontent.com/ZrrSkywalker/MathVerse/main/figs/fig3.png" width="90%"> <br>
    The two phases of the CoT evaluation strategy.
</p>

## Paper Information
- Code: https://github.com/ZrrSkywalker/MathVerse
- Project: https://mathverse-cuhk.github.io/
- Visualization: https://mathverse-cuhk.github.io/#visualization
- Leaderboard: https://mathverse-cuhk.github.io/#leaderboard
- Paper: https://arxiv.org/abs/2403.14624

## Dataset Examples
🖱 Click to expand the examples for six problems versions within three subjects</summary>

<details>
<summary>🔍 Plane Geometry</summary>

<p align="center">
    <img src="https://raw.githubusercontent.com/ZrrSkywalker/MathVerse/main/figs/ver1.png" width="50%"> <br>
</p>
</details>

<details>
<summary>🔍 Solid Geometry</summary>

<p align="center">
    <img src="https://raw.githubusercontent.com/ZrrSkywalker/MathVerse/main/figs/ver2.png" width="50%"> <br>
</p>
</details>

<details>
<summary>🔍 Functions</summary>

<p align="center">
    <img src="https://raw.githubusercontent.com/ZrrSkywalker/MathVerse/main/figs/ver3.png" width="50%"> <br>
</p>
</details>

## Leaderboard
### Contributing to the Leaderboard

🚨 The [Leaderboard](https://mathverse-cuhk.github.io/#leaderboard) is continuously being updated. 

The evaluation instructions and tools will be released soon. For now, please send your results on the ***testmini*** set to this email: 1700012927@pku.edu.cn. Please refer to the following template to prepare your result json file.

- [output_testmini_template.json]()

## License
This project is released under the MIT license.

## Citation

If you find **MathVerse** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@inproceedings{zhang2024mathverse,
  title={MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?},
  author={Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, Hongsheng Li},
  booktitle={arXiv},
  year={2024}
}
```