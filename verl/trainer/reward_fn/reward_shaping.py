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
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def _ece_per_bin(conf: np.ndarray, acc: np.ndarray, n_bins: int = 10) -> list[dict]:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": lo, "hi": hi, "count": 0, "avg_conf": 0.0, "avg_acc": 0.0, "gap": 0.0})
        else:
            ac, cc = float(acc[mask].mean()), float(conf[mask].mean())
            bins.append({"lo": lo, "hi": hi, "count": count, "avg_conf": cc, "avg_acc": ac, "gap": abs(ac - cc)})
    return bins


def _high_confidence_error_ratio(conf: np.ndarray, acc: np.ndarray, threshold: float = 0.8) -> float:
    high_conf = conf >= threshold
    if high_conf.sum() == 0:
        return 0.0
    return float((acc[high_conf] < 0.5).mean())


def _build_groups_by_uid(uid_arr, n, n_per_prompt):
    """Build rollout groups by uid when available; fallback to contiguous chunks.

    Why: trainer.balance_batch=True can reorder samples before reward shaping,
    which breaks contiguous grouping assumptions.
    """
    if uid_arr is not None and len(uid_arr) == n:
        groups = {}
        for idx, uid in enumerate(uid_arr):
            groups.setdefault(uid, []).append(idx)
        return [idxs for idxs in groups.values() if len(idxs) > 1]

    # fallback for compatibility
    n_groups = n // n_per_prompt if n_per_prompt > 0 else 0
    return [list(range(g * n_per_prompt, (g + 1) * n_per_prompt)) for g in range(n_groups)]


def _ranking_bonus(accs, confs, groups, margin):
    bonus = np.zeros_like(accs, dtype=np.float64)
    for idxs in groups:
        if len(idxs) <= 1:
            continue
        ga, gc = accs[idxs], confs[idxs]
        for local_i, global_i in enumerate(idxs):
            if gc[local_i] < 0:
                continue
            loss_sum, cnt = 0.0, 0
            for local_j in range(len(idxs)):
                if local_i == local_j or gc[local_j] < 0:
                    continue
                ds = ga[local_i] - ga[local_j]
                if abs(ds) < 1e-8:
                    continue
                sign = 1.0 if ds > 0 else -1.0
                loss_sum += max(0.0, -sign * (gc[local_i] - gc[local_j]) + margin)
                cnt += 1
            if cnt:
                bonus[global_i] = -(loss_sum / cnt)
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
        for idx, uid in enumerate(uid_arr):
            uid_groups.setdefault(uid, []).append(idx)

        pair_to_prompt_groups: dict[int, dict[str, list[list[int]]]] = {}
        for idxs in uid_groups.values():
            if not idxs:
                continue
            pair_id = int(pair_ids[idxs[0]])
            if pair_id < 0:
                continue
            bucket = "noisy" if is_noisy[idxs[0]] >= 0.5 else "clean"
            pair_to_prompt_groups.setdefault(pair_id, {"clean": [], "noisy": []})[bucket].append(idxs)

        for pair_id, prompt_groups in pair_to_prompt_groups.items():
            clean_groups = sorted(prompt_groups["clean"], key=lambda idxs: idxs[0])
            noisy_groups = sorted(prompt_groups["noisy"], key=lambda idxs: idxs[0])
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

        cmn = float(np.mean(diffs)) if diffs else 0.0
        return bonus, pair_count, cmn

    for pid in np.unique(pair_ids):
        if pid < 0:
            continue
        idx = np.where(pair_ids == pid)[0]
        if len(idx) < 2:
            continue
        clean_idx = idx[is_noisy[idx] < 0.5]
        noisy_idx = idx[is_noisy[idx] >= 0.5]
        if len(clean_idx) == 0 or len(noisy_idx) == 0:
            continue

        pair_count += _apply_pairwise_penalty(
            bonus=bonus,
            diffs=diffs,
            accs=accs,
            confs=confs,
            corr_levels=corr_levels,
            clean_indices=clean_idx,
            noisy_indices=noisy_idx,
            margin=margin,
            alpha=alpha,
        )

    cmn = float(np.mean(diffs)) if diffs else 0.0
    return bonus, pair_count, cmn


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

    rank_b = np.zeros(n)
    if rank_coef != 0.0 and len(groups) > 0:
        rank_b = _ranking_bonus(shaping_accs, confs, groups, rank_margin)

    pair_b = np.zeros(n)
    pair_count, cmn_conf = 0, 0.0
    if corr_coef != 0.0:
        def _get_array(key, default):
            if key in batch.non_tensor_batch:
                return np.asarray(batch.non_tensor_batch[key], dtype=np.float64)
            if key in reward_extra_infos_dict:
                return np.asarray(reward_extra_infos_dict[key], dtype=np.float64)
            return np.full(n, default, dtype=np.float64)
        pid = _get_array("pair_id", -1).astype(np.int64)
        noisy = _get_array("is_noisy", 0.0)
        corr_lv = _get_array("corruption_level", 0.0)
        pair_b, pair_count, cmn_conf = _pair_bonus(
            shaping_accs, confs, pid, noisy, corr_lv, n, corr_margin, corr_alpha, uid_arr=uid_arr
        )

    len_b = np.zeros(n)
    if len_coef != 0.0:
        prompt_len = batch.batch["prompts"].shape[1]
        valid_resp = batch.batch["attention_mask"][:, prompt_len:].sum(dim=1).cpu().numpy()
        len_b = valid_resp / 1000.0

    total_bonus = rank_coef * rank_b + corr_coef * pair_b + len_coef * len_b
    if abs(total_bonus.sum()) > 1e-12:
        prompt_len = batch.batch["prompts"].shape[1]
        valid_resp_lens = batch.batch["attention_mask"][:, prompt_len:].sum(dim=1).long()
        for i in range(n):
            pos = int(valid_resp_lens[i].item()) - 1
            if pos >= 0:
                reward_tensor[i, pos] += total_bonus[i]

    reward_extra_infos_dict["rank_bonus"] = rank_b.tolist()
    reward_extra_infos_dict["pair_bonus"] = pair_b.tolist()
    reward_extra_infos_dict["pair_count"] = [float(pair_count)] * n
    reward_extra_infos_dict["clean_minus_noisy_conf"] = [cmn_conf] * n

    valid_mask = confs >= 0
    if valid_mask.sum() > 0:
        v_confs, v_accs = confs[valid_mask], metric_accs[valid_mask]
        reward_extra_infos_dict["ece"] = [_binary_ece(v_confs, v_accs)] * n
        reward_extra_infos_dict["mean_confidence"] = [float(v_confs.mean())] * n
        reward_extra_infos_dict["high_conf_error_ratio"] = [_high_confidence_error_ratio(v_confs, v_accs)] * n
        reward_extra_infos_dict["overconfidence_rate"] = [
            float(((v_accs < 0.5) & (v_confs > 0.5)).mean())
        ] * n
        reward_extra_infos_dict["underconfidence_rate"] = [
            float(((v_accs > 0.5) & (v_confs < 0.5)).mean())
        ] * n
        ece_bins = _ece_per_bin(v_confs, v_accs)
        for b in ece_bins:
            tag = f"ece_bin_{b['lo']:.1f}_{b['hi']:.1f}"
            reward_extra_infos_dict[f"{tag}/count"] = [b["count"]] * n
            reward_extra_infos_dict[f"{tag}/avg_conf"] = [b["avg_conf"]] * n
            reward_extra_infos_dict[f"{tag}/avg_acc"] = [b["avg_acc"]] * n
            reward_extra_infos_dict[f"{tag}/gap"] = [b["gap"]] * n

        # --- Enhanced batch-level metrics for case study ---
        # Accuracy stats
        reward_extra_infos_dict["mean_accuracy"] = [float(v_accs.mean())] * n
        reward_extra_infos_dict["accuracy_std"] = [float(v_accs.std())] * n
        reward_extra_infos_dict["confidence_std"] = [float(v_confs.std())] * n

        # Confidence-accuracy correlation (Kendall's tau) - key calibration signal
        if len(v_confs) >= 5 and scipy_stats is not None:
            try:
                tau, tau_p = scipy_stats.kendalltau(v_confs, v_accs)
                reward_extra_infos_dict["kendall_tau"] = [float(tau) if not np.isnan(tau) else 0.0] * n
                reward_extra_infos_dict["kendall_tau_pvalue"] = [float(tau_p) if not np.isnan(tau_p) else 1.0] * n
            except Exception:
                reward_extra_infos_dict["kendall_tau"] = [0.0] * n
                reward_extra_infos_dict["kendall_tau_pvalue"] = [1.0] * n

            # Pearson correlation
            try:
                r, r_p = scipy_stats.pearsonr(v_confs, v_accs)
                reward_extra_infos_dict["pearson_r"] = [float(r) if not np.isnan(r) else 0.0] * n
            except Exception:
                reward_extra_infos_dict["pearson_r"] = [0.0] * n
        else:
            reward_extra_infos_dict["kendall_tau"] = [0.0] * n
            reward_extra_infos_dict["kendall_tau_pvalue"] = [1.0] * n
            reward_extra_infos_dict["pearson_r"] = [0.0] * n

        # Brier score: mean (conf - acc)^2
        brier = float(((v_confs - v_accs) ** 2).mean())
        reward_extra_infos_dict["brier_score"] = [brier] * n

        # Confidence distribution percentiles
        reward_extra_infos_dict["conf_p10"] = [float(np.percentile(v_confs, 10))] * n
        reward_extra_infos_dict["conf_p50"] = [float(np.percentile(v_confs, 50))] * n
        reward_extra_infos_dict["conf_p90"] = [float(np.percentile(v_confs, 90))] * n

        # Format compliance rate
        fmt_ok = np.asarray(reward_extra_infos_dict.get("format_ok", [1.0] * n), dtype=np.float64)
        reward_extra_infos_dict["format_rate"] = [float(fmt_ok.mean())] * n

        # Think length stats (if available)
        think_lens = reward_extra_infos_dict.get("think_length", None)
        if think_lens is not None and len(think_lens) == n:
            tl = np.asarray(think_lens, dtype=np.float64)
            reward_extra_infos_dict["mean_think_length"] = [float(tl.mean())] * n
            reward_extra_infos_dict["think_length_std"] = [float(tl.std())] * n

        # Response length stats (if available)
        resp_lens = reward_extra_infos_dict.get("response_length", None)
        if resp_lens is not None and len(resp_lens) == n:
            rl = np.asarray(resp_lens, dtype=np.float64)
            reward_extra_infos_dict["mean_response_length"] = [float(rl.mean())] * n

        # Per-group (prompt) accuracy/confidence variance - diversity signal
        if len(groups) > 0:
            group_acc_vars, group_conf_vars = [], []
            rank_violations = 0
            rank_total = 0
            for idxs in groups:
                ga, gc = metric_accs[idxs], confs[idxs]
                gc_valid = gc[gc >= 0]
                ga_valid = ga[gc >= 0]
                if len(gc_valid) > 1:
                    group_acc_vars.append(float(ga_valid.var()))
                    group_conf_vars.append(float(gc_valid.var()))
                    # Count rank violations: higher conf but lower acc
                    for i in range(len(gc_valid)):
                        for j in range(i + 1, len(gc_valid)):
                            if (gc_valid[i] - gc_valid[j]) * (ga_valid[i] - ga_valid[j]) < 0:
                                rank_violations += 1
                            rank_total += 1
            if group_acc_vars:
                reward_extra_infos_dict["mean_group_acc_var"] = [float(np.mean(group_acc_vars))] * n
                reward_extra_infos_dict["mean_group_conf_var"] = [float(np.mean(group_conf_vars))] * n
            if rank_total > 0:
                reward_extra_infos_dict["rank_violation_rate"] = [float(rank_violations / rank_total)] * n

    return reward_tensor, reward_extra_infos_dict
