"""Batch-level reward shaping: ranking bonus, pair bonus, ECE, calibration metrics.

Applied after per-sample reward, before advantage estimation.
Implements Strategy A (within-group confidence ranking) and Strategy B (clean-vs-corrupt pairs).
Includes comprehensive batch-level calibration/performance metrics for wandb & case study.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


def _binary_ece(conf: np.ndarray, acc: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for bin_index, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if bin_index == 0:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf > low) & (conf <= high)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def _ece_per_bin(conf: np.ndarray, acc: np.ndarray, n_bins: int = 10) -> list[dict]:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for bin_index, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if bin_index == 0:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf > low) & (conf <= high)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": low, "hi": high, "count": 0, "avg_conf": 0.0, "avg_acc": 0.0, "gap": 0.0})
        else:
            avg_acc, avg_conf = float(acc[mask].mean()), float(conf[mask].mean())
            bins.append(
                {"lo": low, "hi": high, "count": count, "avg_conf": avg_conf, "avg_acc": avg_acc, "gap": abs(avg_acc - avg_conf)}
            )
    return bins


def _high_confidence_error_ratio(conf: np.ndarray, acc: np.ndarray, threshold: float = 0.8) -> float:
    high_conf = conf >= threshold
    if high_conf.sum() == 0:
        return 0.0
    return float((acc[high_conf] < 0.5).mean())


def _build_groups_by_uid(uid_arr, n, n_per_prompt):
    if uid_arr is not None and len(uid_arr) == n:
        groups = {}
        for index, uid in enumerate(uid_arr):
            groups.setdefault(uid, []).append(index)
        return [indices for indices in groups.values() if len(indices) > 1]

    n_groups = n // n_per_prompt if n_per_prompt > 0 else 0
    return [list(range(group * n_per_prompt, (group + 1) * n_per_prompt)) for group in range(n_groups)]


def _ranking_bonus(accs, confs, groups, margin):
    bonus = np.zeros_like(accs, dtype=np.float64)
    for indices in groups:
        if len(indices) <= 1:
            continue
        group_acc, group_conf = accs[indices], confs[indices]
        for local_i, global_i in enumerate(indices):
            if group_conf[local_i] < 0:
                continue
            loss_sum, count = 0.0, 0
            for local_j in range(len(indices)):
                if local_i == local_j or group_conf[local_j] < 0:
                    continue
                delta_score = group_acc[local_i] - group_acc[local_j]
                if abs(delta_score) < 1e-8:
                    continue
                sign = 1.0 if delta_score > 0 else -1.0
                loss_sum += max(0.0, -sign * (group_conf[local_i] - group_conf[local_j]) + margin)
                count += 1
            if count:
                bonus[global_i] = -(loss_sum / count)
    return bonus


def _apply_pairwise_penalty(
    bonus: np.ndarray,
    diffs: list[float],
    accs: np.ndarray,
    confs: np.ndarray,
    corr_levels: np.ndarray,
    clean_indices: list[int] | np.ndarray,
    noisy_indices: list[int] | np.ndarray,
    margin: float,
    alpha: float,
) -> int:
    pair_count = 0
    pair_len = min(len(clean_indices), len(noisy_indices))
    for offset in range(pair_len):
        clean_idx = int(clean_indices[offset])
        noisy_idx = int(noisy_indices[offset])
        if accs[clean_idx] <= 0.5 or confs[clean_idx] < 0 or confs[noisy_idx] < 0:
            continue

        delta = margin + alpha * corr_levels[noisy_idx]
        pair_loss = max(0.0, confs[noisy_idx] - confs[clean_idx] + delta)
        phi = (confs[clean_idx] - confs[noisy_idx]) - pair_loss
        bonus[clean_idx] += phi
        bonus[noisy_idx] -= phi
        diffs.append(confs[clean_idx] - confs[noisy_idx])
        pair_count += 1
    return pair_count


def _pair_bonus(accs, confs, pair_ids, is_noisy, corr_levels, n, margin, alpha, uid_arr=None):
    bonus = np.zeros(n, dtype=np.float64)
    diffs = []
    pair_count = 0

    if uid_arr is not None and len(uid_arr) == n:
        uid_groups = {}
        for index, uid in enumerate(uid_arr):
            uid_groups.setdefault(uid, []).append(index)

        pair_to_prompt_groups: dict[int, dict[str, list[list[int]]]] = {}
        for indices in uid_groups.values():
            if not indices:
                continue
            pair_id = int(pair_ids[indices[0]])
            if pair_id < 0:
                continue
            bucket = "noisy" if is_noisy[indices[0]] >= 0.5 else "clean"
            pair_to_prompt_groups.setdefault(pair_id, {"clean": [], "noisy": []})[bucket].append(indices)

        for pair_id, prompt_groups in pair_to_prompt_groups.items():
            clean_groups = sorted(prompt_groups["clean"], key=lambda indices: indices[0])
            noisy_groups = sorted(prompt_groups["noisy"], key=lambda indices: indices[0])
            if not clean_groups or not noisy_groups:
                continue

            for clean_group, noisy_group in zip(clean_groups, noisy_groups, strict=False):
                pair_count += _apply_pairwise_penalty(
                    bonus=bonus,
                    diffs=diffs,
                    accs=accs,
                    confs=confs,
                    corr_levels=corr_levels,
                    clean_indices=clean_group,
                    noisy_indices=noisy_group,
                    margin=margin,
                    alpha=alpha,
                )

        clean_minus_noisy_conf = float(np.mean(diffs)) if diffs else 0.0
        return bonus, pair_count, clean_minus_noisy_conf

    for pair_id in np.unique(pair_ids):
        if pair_id < 0:
            continue
        indices = np.where(pair_ids == pair_id)[0]
        if len(indices) < 2:
            continue
        clean_indices = indices[is_noisy[indices] < 0.5]
        noisy_indices = indices[is_noisy[indices] >= 0.5]
        if len(clean_indices) == 0 or len(noisy_indices) == 0:
            continue

        pair_count += _apply_pairwise_penalty(
            bonus=bonus,
            diffs=diffs,
            accs=accs,
            confs=confs,
            corr_levels=corr_levels,
            clean_indices=clean_indices,
            noisy_indices=noisy_indices,
            margin=margin,
            alpha=alpha,
        )

    clean_minus_noisy_conf = float(np.mean(diffs)) if diffs else 0.0
    return bonus, pair_count, clean_minus_noisy_conf


def apply_reward_shaping(
    reward_tensor: torch.Tensor,
    reward_extra_infos_dict: dict[str, list],
    batch,
    config,
) -> tuple[torch.Tensor, dict[str, list]]:
    shaping_cfg = config.reward.get("reward_shaping", {})
    rank_coef = float(shaping_cfg.get("rank_reward_coef", 0.0))
    rank_margin = float(shaping_cfg.get("rank_reward_margin", 0.05))
    corr_coef = float(shaping_cfg.get("corr_reward_coef", 0.0))
    corr_margin = float(shaping_cfg.get("corr_reward_margin", 0.05))
    corr_alpha = float(shaping_cfg.get("corr_reward_alpha", 0.1))
    len_coef = float(shaping_cfg.get("len_reward_coef", 0.0))
    n_per_prompt = int(config.actor_rollout_ref.rollout.n)
    n = reward_tensor.shape[0]
    metric_accs = np.asarray(reward_extra_infos_dict.get("acc", [0.0] * n), dtype=np.float64)
    shaping_accs = np.asarray(reward_extra_infos_dict.get("reward_acc", metric_accs), dtype=np.float64)
    confs = np.asarray(reward_extra_infos_dict.get("confidence", [-1.0] * n), dtype=np.float64)

    uid_arr = None
    if "uid" in batch.non_tensor_batch:
        uid_arr = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
    groups = _build_groups_by_uid(uid_arr=uid_arr, n=n, n_per_prompt=n_per_prompt)

    rank_bonus = np.zeros(n)
    if rank_coef != 0.0 and len(groups) > 0:
        rank_bonus = _ranking_bonus(shaping_accs, confs, groups, rank_margin)

    pair_bonus = np.zeros(n)
    pair_count, clean_minus_noisy_conf = 0, 0.0
    if corr_coef != 0.0:
        def _get_array(key, default):
            if key in batch.non_tensor_batch:
                return np.asarray(batch.non_tensor_batch[key], dtype=np.float64)
            if key in reward_extra_infos_dict:
                return np.asarray(reward_extra_infos_dict[key], dtype=np.float64)
            return np.full(n, default, dtype=np.float64)

        pair_ids = _get_array("pair_id", -1).astype(np.int64)
        is_noisy = _get_array("is_noisy", 0.0)
        corr_levels = _get_array("corruption_level", 0.0)
        pair_bonus, pair_count, clean_minus_noisy_conf = _pair_bonus(
            shaping_accs, confs, pair_ids, is_noisy, corr_levels, n, corr_margin, corr_alpha, uid_arr=uid_arr
        )

    len_bonus = np.zeros(n)
    if len_coef != 0.0:
        prompt_len = batch.batch["prompts"].shape[1]
        valid_response = batch.batch["attention_mask"][:, prompt_len:].sum(dim=1).cpu().numpy()
        len_bonus = valid_response / 1000.0

    total_bonus = rank_coef * rank_bonus + corr_coef * pair_bonus + len_coef * len_bonus
    if abs(total_bonus.sum()) > 1e-12:
        prompt_len = batch.batch["prompts"].shape[1]
        valid_response_lengths = batch.batch["attention_mask"][:, prompt_len:].sum(dim=1).long()
        for index in range(n):
            pos = int(valid_response_lengths[index].item()) - 1
            if pos >= 0:
                reward_tensor[index, pos] += total_bonus[index]

    reward_extra_infos_dict["rank_bonus"] = rank_bonus.tolist()
    reward_extra_infos_dict["pair_bonus"] = pair_bonus.tolist()
    reward_extra_infos_dict["pair_count"] = [float(pair_count)] * n
    reward_extra_infos_dict["clean_minus_noisy_conf"] = [clean_minus_noisy_conf] * n

    valid_mask = confs >= 0
    if valid_mask.sum() > 0:
        valid_confs, valid_accs = confs[valid_mask], metric_accs[valid_mask]
        reward_extra_infos_dict["ece"] = [_binary_ece(valid_confs, valid_accs)] * n
        reward_extra_infos_dict["mean_confidence"] = [float(valid_confs.mean())] * n
        reward_extra_infos_dict["high_conf_error_ratio"] = [_high_confidence_error_ratio(valid_confs, valid_accs)] * n
        reward_extra_infos_dict["overconfidence_rate"] = [float(((valid_accs < 0.5) & (valid_confs > 0.5)).mean())] * n
        reward_extra_infos_dict["underconfidence_rate"] = [float(((valid_accs > 0.5) & (valid_confs < 0.5)).mean())] * n
        ece_bins = _ece_per_bin(valid_confs, valid_accs)
        for entry in ece_bins:
            tag = f"ece_bin_{entry['lo']:.1f}_{entry['hi']:.1f}"
            reward_extra_infos_dict[f"{tag}/count"] = [entry["count"]] * n
            reward_extra_infos_dict[f"{tag}/avg_conf"] = [entry["avg_conf"]] * n
            reward_extra_infos_dict[f"{tag}/avg_acc"] = [entry["avg_acc"]] * n
            reward_extra_infos_dict[f"{tag}/gap"] = [entry["gap"]] * n

        reward_extra_infos_dict["mean_accuracy"] = [float(valid_accs.mean())] * n
        reward_extra_infos_dict["accuracy_std"] = [float(valid_accs.std())] * n
        reward_extra_infos_dict["confidence_std"] = [float(valid_confs.std())] * n

        if len(valid_confs) >= 5 and scipy_stats is not None:
            try:
                tau, tau_p = scipy_stats.kendalltau(valid_confs, valid_accs)
                reward_extra_infos_dict["kendall_tau"] = [float(tau) if not np.isnan(tau) else 0.0] * n
                reward_extra_infos_dict["kendall_tau_pvalue"] = [float(tau_p) if not np.isnan(tau_p) else 1.0] * n
            except Exception:
                reward_extra_infos_dict["kendall_tau"] = [0.0] * n
                reward_extra_infos_dict["kendall_tau_pvalue"] = [1.0] * n

            try:
                pearson_r, _ = scipy_stats.pearsonr(valid_confs, valid_accs)
                reward_extra_infos_dict["pearson_r"] = [float(pearson_r) if not np.isnan(pearson_r) else 0.0] * n
            except Exception:
                reward_extra_infos_dict["pearson_r"] = [0.0] * n
        else:
            reward_extra_infos_dict["kendall_tau"] = [0.0] * n
            reward_extra_infos_dict["kendall_tau_pvalue"] = [1.0] * n
            reward_extra_infos_dict["pearson_r"] = [0.0] * n

        brier = float(((valid_confs - valid_accs) ** 2).mean())
        reward_extra_infos_dict["brier_score"] = [brier] * n
        reward_extra_infos_dict["conf_p10"] = [float(np.percentile(valid_confs, 10))] * n
        reward_extra_infos_dict["conf_p50"] = [float(np.percentile(valid_confs, 50))] * n
        reward_extra_infos_dict["conf_p90"] = [float(np.percentile(valid_confs, 90))] * n

        format_ok = np.asarray(reward_extra_infos_dict.get("format_ok", [1.0] * n), dtype=np.float64)
        reward_extra_infos_dict["format_rate"] = [float(format_ok.mean())] * n

        think_lengths = reward_extra_infos_dict.get("think_length", None)
        if think_lengths is not None and len(think_lengths) == n:
            think_lengths = np.asarray(think_lengths, dtype=np.float64)
            reward_extra_infos_dict["mean_think_length"] = [float(think_lengths.mean())] * n
            reward_extra_infos_dict["think_length_std"] = [float(think_lengths.std())] * n

        response_lengths = reward_extra_infos_dict.get("response_length", None)
        if response_lengths is not None and len(response_lengths) == n:
            response_lengths = np.asarray(response_lengths, dtype=np.float64)
            reward_extra_infos_dict["mean_response_length"] = [float(response_lengths.mean())] * n

        if len(groups) > 0:
            group_acc_vars, group_conf_vars = [], []
            rank_violations = 0
            rank_total = 0
            for indices in groups:
                group_acc, group_conf = metric_accs[indices], confs[indices]
                valid_group_conf = group_conf[group_conf >= 0]
                valid_group_acc = group_acc[group_conf >= 0]
                if len(valid_group_conf) > 1:
                    group_acc_vars.append(float(valid_group_acc.var()))
                    group_conf_vars.append(float(valid_group_conf.var()))
                    for i in range(len(valid_group_conf)):
                        for j in range(i + 1, len(valid_group_conf)):
                            if (valid_group_conf[i] - valid_group_conf[j]) * (valid_group_acc[i] - valid_group_acc[j]) < 0:
                                rank_violations += 1
                            rank_total += 1
            if group_acc_vars:
                reward_extra_infos_dict["mean_group_acc_var"] = [float(np.mean(group_acc_vars))] * n
                reward_extra_infos_dict["mean_group_conf_var"] = [float(np.mean(group_conf_vars))] * n
            if rank_total > 0:
                reward_extra_infos_dict["rank_violation_rate"] = [float(rank_violations / rank_total)] * n

    return reward_tensor, reward_extra_infos_dict