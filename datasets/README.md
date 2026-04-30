---
license: apache-2.0
task_categories:
- reinforcement-learning
language:
- en
---
# [NeurIPS 2025] Enhancing the Outcome Reward-based RL Training of MLLMs with Self-Consistency Sampling

**A simple, general sampling method for RLVR with multi-choice dataset to solve unfaithful reasoning phenomenon!**

# SCS Resouces

 [**📖 Paper**](https://arxiv.org/abs/2511.10648) | [**🤗 Dataset**](https://huggingface.co/datasets/GenuineWWD/SCS_data) | [**💻 Code**](https://github.com/GenuineWWD/SCS)


## 🔔News
- **🔥[2025-11-9] Release the eval codes! 🚀**
- **🔥[2025-10-13] Release the dataset the codes! 🚀**
- **🔥[2025-9-17] Our SCS paper is accepted by NeurIPS 2025! 🚀**
  
## To-do
- [x] Release the eval codes

## 📖 Introduction
**Self‑Consistency Sampling (SCS)** improves outcome‑reward reinforcement learning for multimodal large language models (MLLMs). In multiple‑choice reasoning tasks, models often get the correct answer through faulty reasoning and receive unmerited rewards. SCS mitigates this by introducing visual perturbations and repeated resampling of reasoning trajectories, rewarding only consistent reasoning paths. Integrated into methods like RLOO, GRPO, and REINFORCE++, SCS boosts accuracy by up to **7.7%** on six multimodal benchmarks with minimal extra cost, and generalizes across models including **Qwen2.5‑VL** and **InternVL3**.
![Overview](assets/overview2.png)

## Training
Please refer to [code repo](https://github.com/GenuineWWD/SCS) for more details.

## Evaluation
Please refer to [code repo](https://github.com/GenuineWWD/SCS) for more details.

## Contact
- Jiahao Wang: wjhwdscience@stu.xjtu.edu.cn
- Weiye Xu: ustcxwy0271@mail.ustc.edu.cn

## Citation

**BibTeX:**
```bibtex
@article{wang2025enhancing,
  title={Enhancing the Outcome Reward-based RL Training of MLLMs with Self-Consistency Sampling},
  author={Wang, Jiahao and Xu, Weiye and Yang, Aijun and Zhou, Wengang and Lu, Lewei and Li, Houqiang and Wang, Xiaohua and Zhu, Jinguo},
  journal={arXiv preprint arXiv:2511.10648},
  year={2025}
}
```