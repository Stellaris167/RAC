# RLC — Ranking-aware Language Confidence for Multimodal RL

Paper: <https://arxiv.org/abs/2511.10648>
Dataset: <https://huggingface.co/datasets/GenuineWWD/SCS_data>

---

## 1. Method Overview

### 1.1 Goal

Improve **robustness and calibration** of MLLM during RL training by enforcing
confidence ordering:

- Correct responses should have higher confidence than incorrect ones.
- Under stronger corruption, confidence should decrease in a controlled way.

### 1.2 Confidence Definition

$$\kappa = P_\theta(\hat{a} \mid x, I)$$

Extracted from `<confidence>` tags in the model's chain-of-thought output.

### 1.3 Composite Reward

$$r = r^{acc} + \lambda_{rank}\,r^{rank} + \lambda_{corr}\,r^{corr} + \lambda_{fmt}\,r^{fmt} + \lambda_{len}\,r^{len}$$

| Component | Description |
|-----------|-------------|
| $r^{acc}$ | 1 if answer matches ground truth, 0 otherwise |
| $r^{rank}$ | Within-group pairwise confidence ranking |
| $r^{corr}$ | Clean-vs-corrupt pair confidence reward |
| $r^{fmt}$ | 1 if output contains `<think>`, `<answer>`, `<confidence>` |
| $r^{len}$ | Optional length bonus |

### 1.4 Data Modes

| Mode | Description |
|------|-------------|
| `clean` | Train on original images only |
| `noisy` | Train on noise-augmented images only |
| `pair` | Train on (clean, noisy) expanded pairs sharing `pair_id` |

---

## 2. Architecture

Built on the **verl** GRPO trainer with minimal extensions:

```
RLC/
├── verl/
│   ├── trainer/
│   │   ├── ppo/ray_trainer.py          # +5 lines: optional reward_shaping hook
│   │   └── reward_fn/
│   │       ├── confidence_reward.py    # Per-sample reward (compute_score)
│   │       ├── reward_shaping.py       # Batch-level ranking + pair shaping
│   │       └── confidence_dataset.py   # Custom Dataset for parquet data
│   └── tools/
│       ├── build_noise_dataset.py      # Noise augmentation tool
│       └── preprocess_benchmarks.py    # Raw HF parquet → standardised test sets
│
└── examples/confidence_rl/
    ├── run_qwen2_5vl7b.sh             # Qwen2.5-VL-7B training launcher
    └── run_internvl2_5_8b.sh          # InternVL2.5-8B training launcher
```

### What was modified in verl core

Only **one file**: `verl/trainer/ppo/ray_trainer.py`

1. In `__init__()`: load `reward.reward_shaping.{path,name}` if present
2. In `fit()`: call reward shaping function after `extract_reward(batch)`

Both are no-ops when `reward.reward_shaping` is absent.

---

## 3. Quick Start

### 3.1 Prerequisites

```bash
conda activate mllm_v2
export PYTHONPATH=/path/to/RLC:$PYTHONPATH
```

### 3.2 Train (Qwen2.5-VL, clean mode, GRPO)

```bash
cd RLC
ALGO=grpo MODE=clean bash examples/confidence_rl/run_qwen2_5vl7b.sh
```

### 3.3 Train (InternVL2.5-8B)

```bash
ALGO=grpo MODE=clean bash examples/confidence_rl/run_internvl2_5_8b.sh
```

### 3.4 Train (pair mode with clean-corrupt shaping)

```bash
python -m verl.tools.build_noise_dataset \
    --input_parquet /path/to/train.parquet \
    --output /path/to/train_pair.parquet \
    --mode pair --noise_type gaussian --severity 0.3

ALGO=grpo MODE=pair CORR_COEF=0.3 RANK_COEF=0.5 \
    bash examples/confidence_rl/run_qwen2_5vl7b.sh
```

---

## 4. Configuration Reference

### 4.1 Data

| Override | Default | Description |
|----------|---------|-------------|
| `data.custom_cls.path` | `confidence_dataset.py` | Custom Dataset class |
| `data.prompt_key` | `message_qwenvl` | Prompt column (`message_internvl` for InternVL) |
| `data.max_prompt_length` | 2048 | Max prompt tokens |
| `data.max_response_length` | 6000 | Max response tokens |
| `data.train_batch_size` | 256 | Global batch size |

### 4.2 Reward & Shaping

| Override | Default | Description |
|----------|---------|-------------|
| `reward.custom_reward_function.path` | `confidence_reward.py` | Per-sample reward |
| `+reward.reward_shaping.rank_reward_coef` | 0.5 | Ranking coefficient |
| `+reward.reward_shaping.corr_reward_coef` | 0.0 | Pair coefficient |
| `+reward.reward_shaping.len_reward_coef` | 0.0 | Length bonus coefficient |
| `+reward.reward_shaping.rank_reward_margin` | 0.05 | Ranking hinge margin δ |
| `+reward.reward_shaping.corr_reward_margin` | 0.05 | Pair hinge margin |
| `+reward.reward_shaping.corr_reward_alpha` | 0.1 | Corruption-level scaling |

### 4.3 Actor / Rollout

| Override | Default | Description |
|----------|---------|-------------|
| `actor_rollout_ref.rollout.n` | 8 | Rollouts per prompt |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 2 | vLLM TP |
| `actor_rollout_ref.actor.optim.lr` | 2e-6 | Learning rate |
| `actor_rollout_ref.actor.use_kl_loss` | true | GRPO KL loss |
| `actor_rollout_ref.actor.kl_loss_coef` | 0.01 | KL coefficient |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | KL type |

---

## 5. Evaluation Protocol

### 5.1 Wandb Metrics

Logged automatically via `reward_extra_infos_dict`:

| Metric | Source |
|--------|--------|
| `val-core/{dataset}/acc/mean@N` | Per-dataset accuracy |
| `val-aux/{dataset}/confidence/mean` | Mean confidence |
| `val-aux/{dataset}/high_conf_error_ratio/mean` | Wrong predictions with κ > 0.8 |
| `val-aux/{dataset}/overconfidence_rate/mean` | Wrong with κ > 0.5 |
| `val-aux/{dataset}/underconfidence_rate/mean` | Correct with κ < 0.5 |
| `val-aux/{dataset}/ece_bin_*/gap` | Per-bin calibration gap |
| `train/reward_shaping/binary_ece` | Training ECE |
| `train/reward_shaping/high_conf_error_ratio` | Training high-confidence errors |

### 5.2 Benchmarks (6 test sets, hardcoded)

| Benchmark | Path |
|-----------|------|
| M3CoT | `m3cot_test_processed/test.parquet` |
| MathVerse | `mathverse_test_processed/test.parquet` |
| MathVision | `mathvision_test_processed/test.parquet` |
| MMMU | `mmmu_test_processed/test.parquet` |
| ScienceQA | `scienceqa_test_processed/test.parquet` |
| WeMath | `we_math_test_processed/test.parquet` |

Validation runs every `test_freq` steps via verl's `_validate()` loop.

---

## 6. Citation

```bibtex
@article{wang2025enhancing,
  title={Enhancing the Outcome Reward-based RL Training of MLLMs
         with Self-Consistency Sampling},
  author={Wang, Jiahao and Xu, Weiye and Yang, Aijun and Zhou, Wengang
          and Lu, Lewei and Li, Houqiang and Wang, Xiaohua and Zhu, Jinguo},
  journal={arXiv preprint arXiv:2511.10648},
  year={2025}
}
```


idea:
对每条训练样本，采样 $$K$$ 次 rollout：  
$$\{y_i\}_{i=1}^{K},\quad \text{每次 rollout 得到 }(s_i,\kappa_i)$$

每条轨迹的总 reward
$$r_i = r_i^{\text{acc}} + \lambda_{\text{rank}} r_i^{\text{rank}}
+ \lambda_{\text{corr}} r_i^{\text{corr}}
+ \lambda_{\text{fmt}} r_i^{\text{fmt}}
+ \lambda_{\text{len}} r_i^{\text{len}}$$

含义：
- $$r^{\text{acc}}$$：正确性奖励（对/错或分数）  
- $$r^{\text{fmt}}$$：输出格式约束奖励  
- $$r^{\text{len}}$$：长度/简洁性约束  
- $$r^{\text{rank}}$$：策略 1：Correctness Ranking Reward  
- $$r^{\text{corr}}$$：策略 2：Corruption/CutMix Negative Pair Reward
  

---

5. 方法：将 Ranking Loss 融入 Reward（两种策略 + 公式）

符号
- rollout 索引：$$i,j\in\{1,\ldots,K\}$$
- correctness：$$s_i$$  
- confidence：$$\kappa_i$$
  

---

5.1 策略 1：Correctness Ranking Reward（rollout 组内构造 pairs）

排序符号函数
$$g(s_i,s_j)=
\begin{cases}
1,& s_i>s_j\\
0,& s_i=s_j\\
-1,& s_i<s_j
\end{cases}$$

Pairwise Correctness-Ranking Loss（CRL 风格）
$$\mathcal L_{\text{rank}}(i,j)=\max\Big(0,\; -g(s_i,s_j)(\kappa_i-\kappa_j)+|s_i-s_j|\Big)$$

转为 per-rollout 的 ranking reward（负 loss）
$$r_i^{\text{rank}} = -\frac{1}{K-1}\sum_{j\ne i}\mathcal L_{\text{rank}}(i,j)$$

直觉
- 若 $s_i=1, s_j=0$，希望 $$\kappa_i$$ 比 $$\kappa_j$$ 大（带 margin）。  
- 若 $s_i=s_j$，不强制排序，避免噪声梯度。
  
可调版本（推荐便于调参）  
将 $$|s_i-s_j|$$ 换成可调 margin $\delta$：  
$$\mathcal L_{\text{rank}}(i,j)=\max\big(0,\; -g(s_i,s_j)(\kappa_i-\kappa_j)+\delta\big)
\quad \text{当 } s_i\ne s_j$$


---

5.2 策略 2：Corruption / CutMix Negative Pair Reward（clean vs corrupt 天然对比）

对同一训练样本构造：  
- clean 输入：$$(x,I)\rightarrow (s^{0},\kappa^{0})$$
- corrupt 输入：$$(x,\tilde I^{(t)})\rightarrow (s^{t},\kappa^{t})$$
  
期望约束：  
$$\kappa^{0}>\kappa^{t}\quad\text{（强度越大，差距越大）}$$

5.2.1 Margin Pair Loss → Pair Reward（推荐）
设强度相关 margin：  
$$\Delta(t)=\Delta_{\min}+\alpha t$$

Pairwise loss：  
$$\mathcal L_{\text{corr}}(t)=\max\big(0,\; \kappa^{t}-\kappa^{0}+\Delta(t)\big)$$

定义 pair score：  
$$\phi(t)=\big(\kappa^{0}-\kappa^{t}\big)-\mathcal L_{\text{corr}}(t)$$

将其分配到两条轨迹上（clean 得正向，corrupt 得反向）：  
$$r^{\text{corr}}_{0}=+\phi(t),\qquad r^{\text{corr}}_{t}=-\phi(t)$$

效果
- 最大化 $$r^{\text{corr}}_{0}$$ 推大 $\kappa^{0}-\kappa^{t}$；  
- 最大化 $$r^{\text{corr}}_{t}$$ 会进一步压低 $$\kappa^{t}$$ 相对 $\kappa^{0}$。
  
5.2.2 复用 CRL 结构的质量排序（备选）
给 clean/corrupt 定义“质量标签”：  
$$q^{0}=1,\qquad q^{t}=1-\frac{t}{T}\quad (\text{或直接 }q^t=0)$$

复用 CRL 形式：  
$$\mathcal L_{\text{q-rank}}(0,t)=\max\Big(0,\; -g(q^{0},q^{t})(\kappa^{0}-\kappa^{t}) + |q^{0}-q^{t}|\Big)$$

并设：  
$$r_{0}^{\text{corr}}=-\mathcal L_{\text{q-rank}}(0,t),\qquad
r_{t}^{\text{corr}}=-\mathcal L_{\text{q-rank}}(t,0)$$
