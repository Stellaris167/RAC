# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import getpass
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Generator, Optional

import cloudpickle as pickle
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from packaging import version as vs
from torch.distributed.device_mesh import DeviceMesh

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from vllm.config import LoRAConfig

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import get_device_id, is_npu_available, is_support_ipc, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.ray_utils import get_event_loop, ray_noset_visible_devices
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge, normalize_vllm_attention_backend_env
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_device_uuid,
    get_vllm_max_lora_rank,
    monkey_patch_compute_logits,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}

if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()
def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


class vLLMAsyncRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank)}
            if self.model_config.lora_rank > 0
            else {}
        )

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, _ = get_free_port(ip, with_alive_sock=True)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = get_event_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())
        return address

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _build_inference_engine(self) -> WorkerWrapperBase:
        try:
            return WorkerWrapperBase(vllm_config=self.vllm_config)
        except TypeError:
            return WorkerWrapperBase()

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        normalize_vllm_attention_backend_env()
        set_expandable_segments(False)

        if is_npu_available:
            for key, value in VLLM_ASCEND_REQUIRED_ENV_VARS.items():
                if key not in os.environ:
                    os.environ[key] = value

        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()

        vllm_config = all_kwargs[0]["vllm_config"]
        parallel_config = vllm_config.parallel_config

        rank = all_kwargs[0].get("rank", None)
        if rank is None:
            rank = int(os.environ.get("RANK", "0"))
        all_kwargs[0]["rank"] = int(rank)

        device_name = "NPU" if is_npu_available else "GPU"
        local_rank = all_kwargs[0].get("local_rank", None)
        if local_rank is None:
            dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
            tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
            if dp_size > 1 and tp_size == 1:
                local_rank = 0
            else:
                local_rank = 0 if not ray_noset_visible_devices() else int(
                    ray.get_runtime_context().get_accelerator_ids()[device_name][0]
                )
        all_kwargs[0]["local_rank"] = int(local_rank)

        self.vllm_config = vllm_config
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        if self.config.quantization is not None:
            supported_quantization = ["fp8", "torchao"]
            if self.config.quantization not in supported_quantization:
                raise ValueError(
                    f"Currently only support {supported_quantization} quantization, got: {self.config.quantization}"
                )
            if self.config.quantization == "fp8":
                apply_vllm_fp8_patches()

        self.inference_engine = self._build_inference_engine()
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        model = self.inference_engine.worker.model_runner.model
        monkey_patch_compute_logits(model, len(self.tokenizer))
        patch_vllm_moe_model_weight_loader(model)

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        if method == "load_model":
            return self._load_model(*args, **kwargs)
        return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=weights,
            )
            self.inference_engine.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
            return

        model_runner = self.inference_engine.worker.model_runner
        model = model_runner.model
        patch_vllm_moe_model_weight_loader(model)
        if is_fp8_model(model_runner.vllm_config):
            logger.info(f"FP8 model detected (async): {model_runner.vllm_config.quant_config}")
            loaded_params = load_quanted_weights(weights, model_runner)
            logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
        else:
            logger.info("Loading standard weights (non-FP8, async)")
            model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        raise NotImplementedError(
            "vLLMAsyncRollout does not support synchronous generate_sequences(). "
            "Please use the async server interface via vLLMReplica and AsyncLLMServerManager."
        )

    def get_zeromq_address(self):
        return self.address


class ServerAdapter(BaseRollout):
    """
    vLLM server adapter used in native async mode, serve as a client to request vLLM server
    to resume/release/update weights and kv_cache.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        super().__init__(config, model_config, device_mesh)
        self.server_handle: ray.actor.ActorHandle = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        if replica_rank == -1:
            self.replica_rank = rank // rollout_world_size
        else:
            self.replica_rank = replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        self.device_uuid = get_device_uuid(get_device_id())
        self.zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{self.device_uuid}.sock"

        self.use_shm = not is_support_ipc()
        if self.use_shm:
            logger.warning(
                "IPC is not supported on your devices. Falling back to shared memory for weight transfer, "
                "which may cause performance degradation. If you are using Ascend NPUs, please ensure that "
                "your software and CANN toolkit versions meet the requirements for IPC support. (Ascend HDK version "
                ">= 25.3.rc1 and CANN toolkit version >= 8.3.RC1)"
            )

    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        """Execute method on inference engine via ray.

        Args:
            method: The method name to execute on the server.
            non_block: If True, execute the method asynchronously and return immediately.
            timeout: Timeout for the collective_rpc call.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.

        Returns:
            The result of the method execution, or None if non_block=True.
        """
        if self.rollout_rank != 0:
            return None

        # Lazy init http server adapter because http server is launched after hybrid engine.
        if self.server_handle is None:
            self.server_handle = ray.get_actor(f"vllm_server_{self.replica_rank}_{self.node_rank}")

        future = self.server_handle.collective_rpc.remote(method, timeout=timeout, args=args, kwargs=kwargs)
        return future if non_block else await future

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            await self._execute_method("wake_up", kwargs={"tags": tags})

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            await self._execute_method("sleep", kwargs={"level": self.sleep_level})

    @torch.no_grad()
    async def update_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int = None, **kwargs
    ):
        """Update model weights via CUDA IPC (fallback to shared memory if IPC not supported) to inference workers."""
        start_time = time.time()

        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )

        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=self.use_shm,
        )
        await sender.async_send_weights(weights)

        if future is not None:
            await future

        # reset prefix cache after updating weights
        if self.rollout_rank == 0:
            await self.server_handle.clear_kv_cache.remote()
            if global_steps is not None:
                await self.server_handle.set_global_steps.remote(global_steps)

        if self.replica_rank == 0 and self.rollout_rank == 0:
            logger.info(f"update_weights done, time cost: {time.time() - start_time:.2f}s")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Note: ServerAdapter uses async server mode and does not support synchronous
        generation. Since SPMD mode was retired (PR #4411), the generation workflow
        should use the async server interface instead.

        Raises:
            NotImplementedError: Always raised as sync generation is not supported.
        """
        raise NotImplementedError(
            "ServerAdapter does not support synchronous generate_sequences(). "
            "The vLLM SPMD mode was retired in PR #4411. For batch generation, "
            "please use the async server interface via vLLMReplica and AsyncLLMServerManager, "
            "or use HFRollout for synchronous generation. "
            "See https://github.com/volcengine/verl/issues/4682 for more details."
        )
