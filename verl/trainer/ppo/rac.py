"""Token-routed task and calibration policy losses for RAC."""

from bisect import bisect_right
from dataclasses import replace

import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, get_policy_loss_fn
from verl.trainer.reward_fn.response_fields import confidence_value_span


RAC_BATCH_KEYS = (
    "reasoning_mask",
    "confidence_mask",
    "rank_advantages",
    "pair_advantages",
)


def build_rac_masks(responses, response_mask, tokenizer):
    """Reserve only numerical confidence tokens for calibration gradients."""
    confidence_mask = torch.zeros_like(response_mask)
    for row, (tokens, valid) in enumerate(zip(responses, response_mask, strict=True)):
        positions = valid.nonzero(as_tuple=True)[0]
        ids = tokens[positions].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        parsed = confidence_value_span(text)
        if parsed is None:
            continue
        _, (start_char, end_char) = parsed

        start = bisect_right(
            range(1, len(ids) + 1),
            start_char,
            key=lambda end: len(
                tokenizer.decode(ids[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            ),
        )
        for index in range(start, len(ids)):
            left = len(tokenizer.decode(ids[:index], skip_special_tokens=False, clean_up_tokenization_spaces=False))
            right = len(
                tokenizer.decode(ids[: index + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            )
            if left >= end_char:
                break
            if left >= start_char and right <= end_char:
                confidence_mask[row, positions[index]] = 1

    reasoning_mask = response_mask.masked_fill(confidence_mask.bool(), 0)
    return reasoning_mask, confidence_mask


def prepare_rac_batch(batch, tokenizer):
    reasoning_mask, confidence_mask = build_rac_masks(
        batch.batch["responses"], batch.batch["response_mask"], tokenizer
    )
    batch.batch["reasoning_mask"] = reasoning_mask
    batch.batch["confidence_mask"] = confidence_mask
    for branch in ("rank", "pair"):
        rewards = batch.batch[f"{branch}_rewards"]
        batch.batch[f"{branch}_advantages"] = rewards.unsqueeze(-1) * confidence_mask


def compute_rac_policy_loss(
    *,
    old_log_prob,
    log_prob,
    advantages,
    reasoning_mask,
    confidence_mask,
    rank_advantages,
    pair_advantages,
    loss_weights,
    config,
    batch_num_tokens=None,
    dp_size=1,
    rollout_is_weights=None,
):
    """Normalize task and calibration branches over their own token sets."""
    if batch_num_tokens is None:
        batch_num_tokens = {
            "task": reasoning_mask.sum().item(),
            "confidence": confidence_mask.sum().item(),
        }

    task_loss_fn = get_policy_loss_fn(config.policy_loss.get("loss_mode", "vanilla"))
    branches = (
        ("task", advantages, reasoning_mask, 1.0, task_loss_fn),
        ("rank", rank_advantages, confidence_mask, loss_weights.get("rank", 0.0), compute_policy_loss_vanilla),
        ("pair", pair_advantages, confidence_mask, loss_weights.get("pair", 0.0), compute_policy_loss_vanilla),
    )
    loss = log_prob.sum() * 0.0
    metrics = {}
    for name, branch_advantages, mask, weight, loss_fn in branches:
        if weight == 0.0 or not mask.any():
            metrics[f"{name}_pg_loss"] = 0.0
            continue

        branch_mask = mask.bool()
        branch_log_prob = torch.where(branch_mask, log_prob, 0.0)
        branch_old_log_prob = torch.where(branch_mask, old_log_prob, 0.0)
        branch_advantages = torch.where(branch_mask, branch_advantages, 0.0)
        branch_is_weights = None
        if rollout_is_weights is not None:
            branch_is_weights = torch.where(branch_mask, rollout_is_weights, 0.0)

        token_group = "task" if name == "task" else "confidence"
        branch_config = replace(
            config,
            global_batch_info={
                "dp_size": dp_size,
                "batch_num_tokens": batch_num_tokens[token_group],
            },
        )
        branch_loss, branch_metrics = loss_fn(
            old_log_prob=branch_old_log_prob,
            log_prob=branch_log_prob,
            advantages=branch_advantages,
            response_mask=mask,
            loss_agg_mode="token-mean",
            config=branch_config,
            rollout_is_weights=branch_is_weights,
        )
        loss = loss + weight * branch_loss
        metrics[f"{name}_pg_loss"] = branch_loss.detach().item()
        metrics.update({f"{name}_{key}": value for key, value in branch_metrics.items()})
    return loss, metrics
