from __future__ import annotations

import json
import logging
import os
from functools import partial
from pathlib import Path

import hydra
import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import auto_set_device
from verl.workers.config import ActorConfig, TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.losses import sft_loss

logger = logging.getLogger(__name__)


class SFTTrainingWorker(TrainingWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def enable_anomaly_detection(self):
        torch.autograd.set_detect_anomaly(True)


def _init_ray() -> None:
    if ray.is_initialized():
        return
    runtime_env = get_ppo_ray_runtime_env()
    if OmegaConf.is_config(runtime_env):
        runtime_env = OmegaConf.to_container(runtime_env, resolve=True)
    else:
        runtime_env = dict(runtime_env)
    runtime_env["env_vars"] = dict(runtime_env.get("env_vars", {}))
    runtime_env["env_vars"]["PYTHONDONTWRITEBYTECODE"] = "1"
    address = os.environ.get("RAY_ADDRESS")
    if address:
        ray.init(address=address, runtime_env=runtime_env)
    else:
        ray.init(runtime_env=runtime_env)


def _build_worker_group(config: DictConfig, model_config, total_steps: int) -> RayWorkerGroup:
    engine_config = omega_conf_to_dataclass(config.engine)
    optimizer_config = omega_conf_to_dataclass(config.optim)
    checkpoint_config = omega_conf_to_dataclass(config.checkpoint)

    engine_config.max_token_len_per_gpu = config.data.max_token_len_per_gpu
    engine_config.micro_batch_size_per_gpu = config.data.micro_batch_size_per_gpu
    engine_config.use_dynamic_bsz = config.data.use_dynamic_bsz
    engine_config.use_remove_padding = model_config.use_remove_padding
    optimizer_config.total_training_steps = total_steps

    worker_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
        profiler_config=None,
    )
    actor_config = ActorConfig(
        strategy=engine_config.strategy,
        rollout_n=1,
        ppo_mini_batch_size=config.data.train_batch_size,
        ppo_micro_batch_size_per_gpu=config.data.micro_batch_size_per_gpu,
        use_dynamic_bsz=config.data.use_dynamic_bsz,
        ppo_max_token_len_per_gpu=config.data.max_token_len_per_gpu,
        loss_agg_mode="token-mean",
    )

    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        use_gpu=True,
        max_colocate_count=1,
        name_prefix="confidence_sft",
    )
    ray_cls = RayClassWithInitArgs(ray.remote(SFTTrainingWorker), config=worker_config)
    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls,
        device_name=config.trainer.device,
        worker_env={"PYTHONDONTWRITEBYTECODE": "1"},
    )
    worker_group.reset()
    if config.trainer.get("detect_anomaly", False):
        worker_group.enable_anomaly_detection()
    worker_group.set_loss_fn(partial(sft_loss, config=actor_config))
    return worker_group


def _build_dataloader(config: DictConfig, model_config):
    dataset = MultiTurnSFTDataset(
        parquet_files=config.data.train_files,
        tokenizer=model_config.tokenizer,
        processor=model_config.processor,
        config=config.data,
        max_samples=config.data.train_max_samples,
    )
    collator = SFTTensorCollator(pad_mode=DatasetPadMode(config.data.pad_mode))
    return StatefulDataLoader(
        dataset,
        batch_size=config.data.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        num_workers=config.data.num_workers,
    )


def _num_training_steps(config: DictConfig, dataset_len: int) -> int:
    if config.trainer.total_training_steps is not None:
        return int(config.trainer.total_training_steps)
    steps_per_epoch = max(1, dataset_len // config.data.train_batch_size)
    return int(config.trainer.total_epochs * steps_per_epoch)


def _unwrap_output(output):
    return output.get() if hasattr(output, "get") else output


def _metrics_dict(output: TensorDict) -> dict:
    metrics = tu.get(output, "metrics")
    cleaned = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        cleaned[key] = value
    return cleaned


def _append_metrics(path: str | None, record: dict) -> None:
    if not path:
        return
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _save_checkpoint(worker_group: RayWorkerGroup, config: DictConfig, global_step: int) -> None:
    local_path = Path(config.trainer.default_local_dir) / f"global_step_{global_step}" / "actor"
    worker_group.save_checkpoint(
        local_path=str(local_path),
        hdfs_path=config.trainer.default_hdfs_dir,
        global_step=global_step,
        max_ckpt_to_keep=config.trainer.max_ckpt_to_keep,
    )
    latest = Path(config.trainer.default_local_dir) / "latest_checkpointed_iteration.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(global_step), encoding="utf-8")


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    if config.trainer.nnodes < 1 or config.trainer.n_gpus_per_node < 1:
        raise ValueError("trainer.nnodes and trainer.n_gpus_per_node must be positive.")
    if config.data.train_batch_size % (config.trainer.nnodes * config.trainer.n_gpus_per_node) != 0:
        raise ValueError("data.train_batch_size must be divisible by the total GPU count.")

    model_config = omega_conf_to_dataclass(config.model)
    dataloader = _build_dataloader(config, model_config)
    total_steps = _num_training_steps(config, len(dataloader.dataset))
    with open_dict(config):
        config.trainer.total_training_steps = total_steps

    _init_ray()
    worker_group = _build_worker_group(config, model_config=model_config, total_steps=total_steps)

    metrics_path = os.environ.get("VERL_FILE_LOGGER_PATH")
    global_step = 0
    for batch in dataloader:
        if global_step >= total_steps:
            break
        global_step += 1
        batch_td = TensorDict(batch, batch_size=[config.data.train_batch_size])
        tu.assign_non_tensor(
            batch_td,
            temperature=1.0,
            pad_mode=DatasetPadMode(config.data.pad_mode),
            update_lr_scheduler=True,
        )
        output = _unwrap_output(worker_group.train_batch(batch_td))
        metrics = _metrics_dict(output)
        record = {"step": global_step, **metrics}
        _append_metrics(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if config.trainer.save_freq > 0 and global_step % config.trainer.save_freq == 0:
            _save_checkpoint(worker_group, config, global_step)

    if global_step == 0:
        raise RuntimeError("No SFT training step was executed. Check dataset size and train_batch_size.")
    if config.trainer.save_freq <= 0 or global_step % config.trainer.save_freq != 0:
        _save_checkpoint(worker_group, config, global_step)
    logger.info("Finished SFT at global_step=%d", global_step)


if __name__ == "__main__":
    main()
