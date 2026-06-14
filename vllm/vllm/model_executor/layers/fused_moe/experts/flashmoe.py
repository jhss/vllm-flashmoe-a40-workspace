# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental FlashMoE distributed MoE backend.

This adapter intentionally keeps the first integration narrow. FlashMoE owns
the router, distributed dispatch, expert compute, and combine inside its own
persistent kernel, so it cannot be expressed as a normal vLLM
prepare/finalize + experts pairing.
"""

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform

logger = init_logger(__name__)


@dataclass(frozen=True)
class _FlashMoEHandleKey:
    tokens_per_rank: int
    token_dim: int
    ffn_size: int
    num_experts: int
    top_k: int
    ep_world: int
    ep_rank: int
    device_id: int
    dtype: torch.dtype


class FlashMoEExperts(mk.FusedMoEExpertsMonolithic):
    """Thin adapter over the optional ``flashmoe`` Python package.

    The backend is explicit opt-in only via ``--moe-backend flashmoe``. It is
    currently intended for single-node EP experiments on unquantized BF16
    gated-SiLU MoE layers.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        self._flashmoe: Any | None = None
        self._flash_handle: Any | None = None
        self._router_handle: Any | None = None
        self._handle_key: _FlashMoEHandleKey | None = None
        self._expert_counts: torch.Tensor | None = None
        self._gate_weight_t: torch.Tensor | None = None
        self._gate_weight_ptr: int | None = None
        self._w13_gate: torch.Tensor | None = None
        self._w13_up: torch.Tensor | None = None
        self._w13_ptr: int | None = None
        self._w13_bias_gate: torch.Tensor | None = None
        self._w13_bias_up: torch.Tensor | None = None
        self._w13_bias_ptr: int | None = None
        self._zero_bias_gate: torch.Tensor | None = None
        self._zero_bias_up: torch.Tensor | None = None
        self._zero_bias_down: torch.Tensor | None = None

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExpertsMonolithic.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason

        parallel = moe_config.moe_parallel_config
        if not parallel.use_ep or parallel.ep_size <= 1:
            return False, "FlashMoE currently requires expert parallelism"
        if parallel.enable_eplb:
            return False, "FlashMoE does not support EPLB yet"
        if parallel.pcp_size != 1 or parallel.sp_size != 1:
            return False, "FlashMoE currently supports only plain EP ranks"
        if moe_config.num_experts % parallel.ep_size != 0:
            return False, "FlashMoE requires num_experts divisible by EP size"
        if moe_config.in_dtype != torch.bfloat16:
            return False, "FlashMoE vLLM adapter currently supports BF16 only"
        if moe_config.experts_per_token not in (1, 2):
            return False, "FlashMoE vLLM adapter currently supports top-1/top-2"
        if moe_config.routing_method not in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ):
            return (
                False,
                "FlashMoE currently matches renormalized softmax top-k routing only",
            )
        if moe_config.swiglu_limit is not None:
            return False, "FlashMoE does not support swiglu_limit yet"
        return True, None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.has_device_capability(70)
            and importlib.util.find_spec("flashmoe") is not None
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return moe_parallel_config.use_ep and moe_parallel_config.ep_size > 1

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # FlashMoE's router computes softmax + top-k internally, then stores
        # selected weights divided by the selected-weight sum.
        return routing_method in (
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        )

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return router_logits_dtype in (None, torch.float32, torch.bfloat16)

    def _stream_ptr(self, tensor: torch.Tensor) -> int:
        return int(torch.cuda.current_stream(tensor.device).cuda_stream)

    def _gpu_arch(self, device: torch.device) -> int:
        major, minor = torch.cuda.get_device_capability(device)
        return major * 100 + minor * 10

    def _make_expert_map(self) -> list[int]:
        parallel = self.moe_config.moe_parallel_config
        experts_per_rank = self.moe_config.num_experts // parallel.ep_size
        return [
            expert // experts_per_rank for expert in range(self.moe_config.num_experts)
        ]

    def _ensure_flashmoe_handles(self, hidden_states: torch.Tensor) -> None:
        assert hidden_states.device.type == "cuda"
        parallel = self.moe_config.moe_parallel_config
        device_id = hidden_states.device.index
        if device_id is None:
            device_id = torch.cuda.current_device()
        key = _FlashMoEHandleKey(
            tokens_per_rank=hidden_states.shape[0],
            token_dim=self.moe_config.hidden_dim,
            ffn_size=self.moe_config.intermediate_size_per_partition,
            num_experts=self.moe_config.num_experts,
            top_k=self.moe_config.experts_per_token,
            ep_world=parallel.ep_size,
            ep_rank=parallel.ep_rank,
            device_id=device_id,
            dtype=hidden_states.dtype,
        )
        if self._handle_key == key:
            return

        self._finalize_handles(hidden_states)

        import flashmoe

        flashmoe.cb.initialize()
        nvshmem_world = flashmoe.cb.get_world_size()
        if nvshmem_world != parallel.ep_size:
            raise RuntimeError(
                "FlashMoE NVSHMEM world size must match vLLM EP size "
                f"(got {nvshmem_world}, expected {parallel.ep_size})."
            )
        if dist.is_initialized() and dist.get_world_size() != parallel.ep_size:
            raise RuntimeError(
                "The experimental FlashMoE backend currently expects the torch "
                "distributed world to be exactly the EP world."
            )

        init_args = flashmoe.InitArgs(
            data_type=flashmoe.DataType.BF16,
            mlp_type=flashmoe.MLPType.GATED,
            act_type=flashmoe.ActivationType.SILU,
            tokens_per_rank=key.tokens_per_rank,
            token_dim=key.token_dim,
            ffn_size=key.ffn_size,
            num_experts=key.num_experts,
            top_k=key.top_k,
            gpu_arch=self._gpu_arch(hidden_states.device),
            stream_ptr=self._stream_ptr(hidden_states),
            device_id=key.device_id,
            ep_world=key.ep_world,
            num_local_experts=self.moe_config.num_local_experts,
            ep_rank=key.ep_rank,
            my_pe=flashmoe.cb.get_rank(),
            expert_map=self._make_expert_map(),
            rank_map=list(range(key.ep_world)),
            # Avoid silent token drops in the first integration. This is more
            # conservative than FlashMoE's average-load default.
            expert_peer_capacity=key.tokens_per_rank * key.top_k,
        )
        self._flashmoe = flashmoe
        self._flash_handle = flashmoe.initialize(init_args)
        self._router_handle = flashmoe.router.initialize(init_args)
        self._expert_counts = torch.empty(
            (key.num_experts,),
            device=hidden_states.device,
            dtype=torch.int32,
        )
        self._handle_key = key
        logger.info_once(
            "Initialized experimental FlashMoE backend for "
            "S=%d H=%d I=%d E=%d top_k=%d",
            key.tokens_per_rank,
            key.token_dim,
            key.ffn_size,
            key.num_experts,
            key.top_k,
        )

    def _finalize_handles(self, tensor: torch.Tensor | None = None) -> None:
        if self._flashmoe is None or self._flash_handle is None:
            return
        try:
            if tensor is None:
                if not torch.cuda.is_available():
                    return
                stream_ptr = int(torch.cuda.current_stream().cuda_stream)
            else:
                stream_ptr = self._stream_ptr(tensor)
            self._flashmoe.finalize(self._flash_handle, stream_ptr)
            if self._router_handle is not None:
                self._flashmoe.router.finalize(self._router_handle, stream_ptr)
        except Exception as e:
            logger.warning("Failed to finalize FlashMoE handles: %s", e)
        finally:
            self._flash_handle = None
            self._router_handle = None
            self._handle_key = None
            self._expert_counts = None

    def __del__(self) -> None:
        self._finalize_handles()

    def _split_w13(self, w13: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w13_ptr != w13.data_ptr():
            intermediate = w13.shape[1] // 2
            self._w13_gate = w13[:, :intermediate, :].contiguous()
            self._w13_up = w13[:, intermediate:, :].contiguous()
            self._w13_ptr = w13.data_ptr()
        assert self._w13_gate is not None
        assert self._w13_up is not None
        return self._w13_gate, self._w13_up

    def _split_w13_bias(
        self,
        w13_bias: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_experts = self.moe_config.num_local_experts
        intermediate = self.moe_config.intermediate_size_per_partition
        if w13_bias is not None:
            if self._w13_bias_ptr != w13_bias.data_ptr():
                self._w13_bias_gate = w13_bias[:, :intermediate].contiguous()
                self._w13_bias_up = w13_bias[:, intermediate:].contiguous()
                self._w13_bias_ptr = w13_bias.data_ptr()
            assert self._w13_bias_gate is not None
            assert self._w13_bias_up is not None
            return self._w13_bias_gate, self._w13_bias_up

        if (
            self._zero_bias_gate is None
            or self._zero_bias_gate.device != device
            or self._zero_bias_gate.dtype != dtype
            or self._zero_bias_gate.shape != (local_experts, intermediate)
        ):
            shape = (local_experts, intermediate)
            self._zero_bias_gate = torch.zeros(shape, device=device, dtype=dtype)
            self._zero_bias_up = torch.zeros(shape, device=device, dtype=dtype)
        assert self._zero_bias_up is not None
        return self._zero_bias_gate, self._zero_bias_up

    def _get_down_bias(
        self,
        w2_bias: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if w2_bias is not None:
            return w2_bias
        shape = (self.moe_config.num_local_experts, self.moe_config.hidden_dim)
        if (
            self._zero_bias_down is None
            or self._zero_bias_down.device != device
            or self._zero_bias_down.dtype != dtype
            or self._zero_bias_down.shape != shape
        ):
            self._zero_bias_down = torch.zeros(shape, device=device, dtype=dtype)
        return self._zero_bias_down

    def _get_gate_weight_t(self, gate_weight: torch.Tensor) -> torch.Tensor:
        if self._gate_weight_ptr != gate_weight.data_ptr():
            self._gate_weight_t = gate_weight.t().contiguous()
            self._gate_weight_ptr = gate_weight.data_ptr()
        assert self._gate_weight_t is not None
        return self._gate_weight_t

    def apply_with_gate_weight(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gate_weight: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError("FlashMoE vLLM adapter currently supports BF16 only.")
        if w1.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
            raise RuntimeError("FlashMoE expects BF16 expert weights.")
        if gate_weight.dtype != hidden_states.dtype:
            raise RuntimeError("FlashMoE expects a BF16 router gate weight.")
        if apply_router_weight_on_input:
            raise RuntimeError(
                "FlashMoE vLLM adapter does not support apply_router_weight_on_input."
            )
        if not w1.is_contiguous() or not w2.is_contiguous():
            raise RuntimeError("FlashMoE expects contiguous expert weights.")
        if gate_weight.shape != (
            self.moe_config.num_experts,
            self.moe_config.hidden_dim,
        ):
            raise RuntimeError(
                "FlashMoE expects an internal gate weight shaped "
                f"({self.moe_config.num_experts}, {self.moe_config.hidden_dim}); "
                f"got {tuple(gate_weight.shape)}."
            )

        self._ensure_flashmoe_handles(hidden_states)
        assert self._flashmoe is not None
        assert self._flash_handle is not None
        assert self._router_handle is not None
        assert self._expert_counts is not None

        self._expert_counts.zero_()
        w13_gate, w13_up = self._split_w13(w1)
        bias_gate, bias_up = self._split_w13_bias(
            self.w1_bias, hidden_states.device, hidden_states.dtype
        )
        bias_down = self._get_down_bias(
            self.w2_bias, hidden_states.device, hidden_states.dtype
        )
        gate_weight_t = self._get_gate_weight_t(gate_weight)
        output = torch.empty_like(hidden_states)
        stream_ptr = self._stream_ptr(hidden_states)

        router_args = self._flashmoe.router.RouterForwardArgs(
            tokens=hidden_states.data_ptr(),
            weights=gate_weight_t.data_ptr(),
            expert_counts=self._expert_counts.data_ptr(),
            stream_ptr=stream_ptr,
        )
        self._flashmoe.router.forward(
            self._router_handle,
            self._flash_handle,
            router_args,
        )

        forward_args = self._flashmoe.ForwardArgs(
            mt=self._flashmoe.MLPType.GATED,
            tokens=hidden_states.data_ptr(),
            expert_counts=self._expert_counts.data_ptr(),
            local_expert_up=w13_gate.data_ptr(),
            local_expert_up_v=w13_up.data_ptr(),
            local_bias_up=bias_gate.data_ptr(),
            local_bias_up_v=bias_up.data_ptr(),
            local_expert_down=w2.data_ptr(),
            local_bias_down=bias_down.data_ptr(),
            moe_out=output.data_ptr(),
            stream_ptr=stream_ptr,
        )
        self._flashmoe.forward(self._flash_handle, forward_args)
        return output

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        raise RuntimeError(
            "FlashMoE requires MoERunner's experimental gate-weight path. "
            "Use an internal-gate MoE layer with --moe-backend flashmoe."
        )
