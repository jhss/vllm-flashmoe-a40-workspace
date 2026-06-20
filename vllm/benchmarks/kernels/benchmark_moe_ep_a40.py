#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small MoE EP profiler for PCIe-only 2-GPU systems.

This benchmark intentionally uses the real FusedMoE layer path instead of
calling only the raw Triton expert kernel. For backends that expose raw
EP-group dispatch/combine, it is meant to separate:

* local routing/expert compute cost,
* allgather dispatch cost, and
* reduce-scatter combine cost

For backends such as DeepEP high-throughput, communication is driven through
the FusedMoE prepare/finalize path and raw EP-group dispatch/combine timings
are reported as null.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.multiprocessing import spawn

# On the local A40 PXB machine, NCCL P2P can hang. Let callers override this,
# but default to the stable path for the profiler.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

from vllm.config import (  # noqa: E402
    CompilationConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (  # noqa: E402
    cleanup_dist_env_and_memory,
    get_ep_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import get_forward_context, set_forward_context  # noqa: E402
from vllm.model_executor.layers.fused_moe import FusedMoE  # noqa: E402
from vllm.utils.math_utils import next_power_of_2  # noqa: E402
from vllm.utils.network_utils import get_distributed_init_method  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402
from vllm.v1.worker.workspace import (  # noqa: E402
    init_workspace_manager,
    is_workspace_manager_initialized,
)


@dataclass
class BenchResult:
    rank: int
    world_size: int
    backend: str
    tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    local_experts: int
    top_k: int
    dtype: str
    warmup: int
    iters: int
    full_forward_us: float
    rank0_forward_us: float
    rank1_forward_us: float | None
    critical_path_us: float
    topk_us: float | None
    dispatch_us: float | None
    combine_us: float | None
    expert_tokens_min: int
    expert_tokens_max: int
    expert_tokens_mean: float
    expert_tokens_cv: float
    expert_tokens_zero: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=1, choices=[1, 2])
    parser.add_argument("--backend", default="allgather_reducescatter")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument(
        "--section-profile-iters",
        type=int,
        default=0,
        help=(
            "Run extra synchronized forward passes with high-level MoE section "
            "timers. This perturbs overlap and is intended for diagnosis only."
        ),
    )
    parser.add_argument("--section-profile-warmup", type=int, default=2)
    parser.add_argument("--section-profile-output", type=Path)
    parser.add_argument(
        "--phase-name",
        default="single",
        help="Label written to profile outputs, e.g. prefill or decode.",
    )
    parser.add_argument(
        "--torch-profile-iters",
        type=int,
        default=0,
        help=(
            "Run extra forward passes under torch.profiler and write a compact "
            "CUDA kernel summary. This is a lightweight fallback when Nsight "
            "Compute/Systems are not installed."
        ),
    )
    parser.add_argument("--torch-profile-warmup", type=int, default=2)
    parser.add_argument("--torch-profile-output", type=Path)
    parser.add_argument(
        "--torch-profile-top-kernels",
        type=int,
        default=30,
        help="Number of CUDA events to keep in the torch profiler summary.",
    )
    parser.add_argument(
        "--torch-profile-chrome-trace",
        action="store_true",
        help="Also export a Chrome trace next to the JSON summary.",
    )
    return parser.parse_args()


def make_vllm_config(args: argparse.Namespace) -> VllmConfig:
    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=args.world_size,
        enable_expert_parallel=args.world_size > 1,
        all2all_backend=args.backend,
    )
    compilation_config = CompilationConfig()
    compilation_config.pass_config.fuse_allreduce_rms = False
    max_tokens = max(1, next_power_of_2(args.tokens))
    return VllmConfig(
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=max_tokens,
            max_num_seqs=min(max_tokens, max(1, args.tokens)),
        ),
    )


def init_rank(vllm_config: VllmConfig, rank: int, local_rank: int) -> None:
    torch.accelerator.set_device_index(local_rank)
    torch.set_default_device(torch.device("cuda", local_rank))

    pc = vllm_config.parallel_config
    pc.data_parallel_rank = rank
    pc.rank = 0

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=pc.world_size,
            rank=0,
            distributed_init_method=get_distributed_init_method(
                pc.master_addr, pc.master_port
            ),
            local_rank=local_rank,
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=pc.tensor_parallel_size,
            pipeline_model_parallel_size=pc.pipeline_parallel_size,
        )


def expert_ids_for_rank(layer: torch.nn.Module, num_experts: int) -> torch.Tensor:
    expert_map = layer.routed_experts.expert_map
    if expert_map is None:
        return torch.arange(num_experts, device="cuda")
    return torch.nonzero(expert_map >= 0, as_tuple=False).flatten().to("cuda")


def make_layer_and_inputs(
    args: argparse.Namespace,
    rank: int,
    vllm_config: VllmConfig,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    set_random_seed(args.seed)
    dtype = torch.bfloat16
    device = torch.device("cuda", rank)

    full_w13 = torch.randn(
        args.num_experts,
        2 * args.intermediate_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    ) / 10
    full_w2 = torch.randn(
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        device=device,
        dtype=dtype,
    ) / 10

    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(
            num_experts=args.num_experts,
            top_k=args.top_k,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            params_dtype=dtype,
            renormalize=False,
            quant_config=None,
            tp_size=1,
            dp_size=args.world_size,
            pcp_size=1,
            prefix=f"a40_moe_ep_profile_{rank}",
            activation="silu",
        )

    local_ids = expert_ids_for_rank(layer, args.num_experts)
    with torch.no_grad():
        layer.routed_experts.w13_weight.copy_(full_w13.index_select(0, local_ids))
        layer.routed_experts.w2_weight.copy_(full_w2.index_select(0, local_ids))
    layer.routed_experts.quant_method.process_weights_after_loading(
        layer.routed_experts
    )

    hidden_states = (
        torch.randn(args.tokens, args.hidden_size, device=device, dtype=dtype) / 10
    )
    router_logits = torch.randn(
        args.tokens, args.num_experts, device=device, dtype=dtype
    )
    return layer, hidden_states, router_logits


@contextmanager
def make_forward_context(
    args: argparse.Namespace,
    vllm_config: VllmConfig,
    device: torch.device,
):
    num_tokens_across_dp = torch.full(
        (args.world_size,), args.tokens, dtype=torch.int, device=device
    )
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=args.tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        ctx = get_forward_context()
        if ctx.dp_metadata is None:
            yield
        else:
            with ctx.dp_metadata.sp_local_sizes(1):
                yield


def time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def expert_token_stats(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[int, int, float, float, int]:
    counts = torch.bincount(
        topk_ids.flatten().to(torch.int64),
        minlength=num_experts,
    ).to(torch.float32)
    mean = counts.mean()
    cv = counts.std(unbiased=False) / mean if mean > 0 else counts.new_tensor(0.0)
    return (
        int(counts.min().item()),
        int(counts.max().item()),
        float(mean.item()),
        float(cv.item()),
        int((counts == 0).sum().item()),
    )


class SectionProfiler:

    def __init__(self) -> None:
        self.samples_ms: dict[str, list[float]] = {}

    @contextmanager
    def section(self, name: str):
        torch.accelerator.synchronize()
        torch.cuda.nvtx.range_push(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.accelerator.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            torch.cuda.nvtx.range_pop()
            self.samples_ms.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> dict[str, dict[str, float | int]]:
        summarized = {}
        for name, values in sorted(self.samples_ms.items()):
            summarized[name] = {
                "count": len(values),
                "avg_ms": sum(values) / len(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        return summarized


_SECTION_PROFILER: SectionProfiler | None = None


def _wrap_profiled_method(cls: type, method_name: str, section_name: str) -> None:
    original_attr = f"_moe_ep_profile_original_{method_name}"
    if hasattr(cls, original_attr):
        return

    original = getattr(cls, method_name)

    def wrapped(self, *args, **kwargs):
        profiler = _SECTION_PROFILER
        if profiler is None:
            return original(self, *args, **kwargs)
        with profiler.section(section_name):
            return original(self, *args, **kwargs)

    setattr(cls, original_attr, original)
    setattr(cls, method_name, wrapped)


def install_section_profilers() -> None:
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep

    _wrap_profiled_method(
        mk.FusedMoEKernelModularImpl, "_prepare", "moe_prepare_total"
    )
    _wrap_profiled_method(
        mk.FusedMoEKernelModularImpl, "_fused_experts", "moe_experts_total"
    )
    _wrap_profiled_method(
        mk.FusedMoEKernelModularImpl, "_finalize", "moe_finalize_total"
    )
    _wrap_profiled_method(
        naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular,
        "prepare",
        "agrs_prepare_dispatch",
    )
    _wrap_profiled_method(
        naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular,
        "finalize",
        "agrs_finalize_combine",
    )

    try:
        from vllm.model_executor.layers.fused_moe.prepare_finalize import deepep_ht
    except ImportError:
        return

    _wrap_deepep_ht_receiver_details(
        deepep_ht.DeepEPHTPrepareAndFinalize,
        deepep_ht,
        mk,
    )
    _wrap_deepep_ht_dispatch(deepep_ht.DeepEPHTPrepareAndFinalize)
    _wrap_deepep_ht_finalize(deepep_ht.DeepEPHTPrepareAndFinalize)


def _wrap_deepep_ht_receiver_details(
    cls: type,
    deepep_ht_module,
    mk_module,
) -> None:
    original_attr = "_moe_ep_profile_original__receiver"
    if hasattr(cls, original_attr):
        return

    original = getattr(cls, "_receiver")

    def wrapped(
        self,
        event,
        has_scales: bool,
        token_data: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        expert_topk_ids: torch.Tensor | None,
        num_experts: int,
        expert_num_tokens_per_expert_list: list[int],
        expert_topk_weights: torch.Tensor | None,
        a1_scale: torch.Tensor | None,
        quant_config,
        defer_input_quant: bool,
    ):
        profiler = _SECTION_PROFILER
        if profiler is None:
            return original(
                self,
                event,
                has_scales,
                token_data,
                expert_topk_ids,
                num_experts,
                expert_num_tokens_per_expert_list,
                expert_topk_weights,
                a1_scale,
                quant_config,
                defer_input_quant,
            )

        with profiler.section("deepep_receiver_wait"):
            if event.event is not None:
                event.current_stream_wait()

        with profiler.section("deepep_receiver_unpack"):
            if has_scales:
                expert_x, expert_x_scale = token_data
            else:
                expert_x, expert_x_scale = token_data, None

        with profiler.section("deepep_receiver_topk_remap"):
            assert expert_topk_ids is not None
            use_direct_assignment = self._preserve_raw_local_ids
            use_local_expert_ids = (
                deepep_ht_module.envs.VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS
                or use_direct_assignment
            )
            if not use_direct_assignment and use_local_expert_ids:
                local_num_experts = len(expert_num_tokens_per_expert_list)
                expert_topk_ids = deepep_ht_module.remap_deepep_ht_topk_ids(
                    expert_topk_ids,
                    local_num_experts,
                    0,
                )
            elif not use_direct_assignment:
                invalid_expert_id = (
                    num_experts - 1 if self.rank_expert_offset == 0 else 0
                )
                expert_topk_ids = deepep_ht_module.remap_deepep_ht_topk_ids(
                    expert_topk_ids,
                    invalid_expert_id,
                    self.rank_expert_offset,
                )

        with profiler.section("deepep_receiver_metadata"):
            assignment_layout = (
                mk_module.ExpertAssignmentLayout.DeepEPHTLocalRaw
                if use_direct_assignment
                else (
                    mk_module.ExpertAssignmentLayout.DeepEPHTLocalSentinel
                    if use_local_expert_ids
                    else mk_module.ExpertAssignmentLayout.Generic
                )
            )
            expert_tokens_meta = mk_module.ExpertTokensMetadata.make_from_list(
                expert_num_tokens_per_expert_list,
                device=expert_x.device,
                assignment_layout=assignment_layout,
            )

        if not quant_config.is_block_quantized and not defer_input_quant:
            with profiler.section("deepep_receiver_post_quant"):
                expert_x_scale = None
                if expert_x.numel() != 0:
                    expert_x, expert_x_scale = (
                        deepep_ht_module.moe_kernel_quantize_input(
                            expert_x,
                            a1_scale,
                            quant_dtype=quant_config.quant_dtype,
                            per_act_token_quant=False,
                            block_shape=quant_config.block_shape,
                            is_scale_swizzled=quant_config.is_scale_swizzled,
                        )
                    )

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            expert_topk_ids,
            expert_topk_weights,
        )

    setattr(cls, original_attr, original)
    setattr(cls, "_receiver", wrapped)


def _wrap_deepep_ht_dispatch(cls: type) -> None:
    original_attr = "_moe_ep_profile_original__do_dispatch"
    if hasattr(cls, original_attr):
        return

    original = getattr(cls, "_do_dispatch")

    def wrapped(self, *args, **kwargs):
        profiler = _SECTION_PROFILER
        if profiler is None:
            return original(self, *args, **kwargs)
        with profiler.section("deepep_dispatch_submit"):
            receiver = original(self, *args, **kwargs)

        def profiled_receiver():
            active_profiler = _SECTION_PROFILER
            if active_profiler is None:
                return receiver()
            with active_profiler.section("deepep_dispatch_receiver"):
                return receiver()

        return profiled_receiver

    setattr(cls, original_attr, original)
    setattr(cls, "_do_dispatch", wrapped)


def _wrap_deepep_ht_finalize(cls: type) -> None:
    original_attr = "_moe_ep_profile_original__finalize"
    if hasattr(cls, original_attr):
        return

    original = getattr(cls, "_finalize")

    def wrapped(self, *args, **kwargs):
        profiler = _SECTION_PROFILER
        if profiler is None:
            return original(self, *args, **kwargs)
        with profiler.section("deepep_combine_submit"):
            receiver = original(self, *args, **kwargs)
        if receiver is None:
            return None

        def profiled_receiver():
            active_profiler = _SECTION_PROFILER
            if active_profiler is None:
                return receiver()
            with active_profiler.section("deepep_combine_receiver_copy"):
                return receiver()

        return profiled_receiver

    setattr(cls, original_attr, original)
    setattr(cls, "_finalize", wrapped)


def section_profile_path(base: Path, rank: int) -> Path:
    return base.with_name(f"{base.stem}.rank{rank}{base.suffix}")


def run_section_profile(
    args: argparse.Namespace,
    rank: int,
    forward_once: Callable[[], object],
) -> None:
    global _SECTION_PROFILER

    if args.section_profile_iters <= 0:
        return
    if args.section_profile_output is None:
        raise ValueError("--section-profile-output is required for section profiling")

    install_section_profilers()
    for _ in range(args.section_profile_warmup):
        forward_once()
    torch.accelerator.synchronize()

    profiler = SectionProfiler()
    _SECTION_PROFILER = profiler
    try:
        for _ in range(args.section_profile_iters):
            forward_once()
        torch.accelerator.synchronize()
    finally:
        _SECTION_PROFILER = None

    output_path = section_profile_path(args.section_profile_output, rank)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": rank,
        "world_size": args.world_size,
        "backend": args.backend,
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "warmup": args.section_profile_warmup,
        "iters": args.section_profile_iters,
        "sections": profiler.summary(),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _profiler_event_time_us(event, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def run_torch_kernel_profile(
    args: argparse.Namespace,
    rank: int,
    forward_once: Callable[[], object],
) -> None:
    if args.torch_profile_iters <= 0:
        return
    if args.torch_profile_output is None:
        raise ValueError("--torch-profile-output is required for torch profiling")

    from torch.profiler import ProfilerActivity, profile, record_function

    for _ in range(args.torch_profile_warmup):
        forward_once()
    torch.accelerator.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with record_function(f"moe_ep_forward_{args.phase_name}"):
            for _ in range(args.torch_profile_iters):
                forward_once()
        torch.accelerator.synchronize()

    events = []
    total_cuda_us = 0.0
    for event in prof.key_averages():
        cuda_total_us = _profiler_event_time_us(
            event, "device_time_total", "cuda_time_total"
        )
        cuda_self_us = _profiler_event_time_us(
            event, "self_device_time_total", "self_cuda_time_total"
        )
        if cuda_total_us <= 0.0 and cuda_self_us <= 0.0:
            continue
        total_cuda_us += cuda_self_us
        events.append(
            {
                "name": event.key,
                "count": int(event.count),
                "cuda_total_us": cuda_total_us,
                "cuda_self_us": cuda_self_us,
                "cpu_total_us": float(getattr(event, "cpu_time_total", 0.0)),
                "cpu_self_us": float(getattr(event, "self_cpu_time_total", 0.0)),
            }
        )

    events.sort(key=lambda item: item["cuda_total_us"], reverse=True)
    output_path = section_profile_path(args.torch_profile_output, rank)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": rank,
        "world_size": args.world_size,
        "backend": args.backend,
        "phase_name": args.phase_name,
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "warmup": args.torch_profile_warmup,
        "iters": args.torch_profile_iters,
        "total_cuda_self_us": total_cuda_us,
        "top_cuda_events": events[: args.torch_profile_top_kernels],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if args.torch_profile_chrome_trace:
        trace_path = output_path.with_suffix(output_path.suffix + ".trace.json")
        prof.export_chrome_trace(str(trace_path))


def run_worker(
    local_rank: int,
    args: argparse.Namespace,
    vllm_config: VllmConfig,
) -> None:
    rank = local_rank
    init_rank(vllm_config, rank=rank, local_rank=local_rank)
    device = torch.device("cuda", local_rank)

    try:
        if not is_workspace_manager_initialized():
            init_workspace_manager(device)

        layer, hidden_states, router_logits = make_layer_and_inputs(
            args, rank, vllm_config
        )
        local_experts = layer.routed_experts.local_num_experts

        def forward_once():
            with make_forward_context(args, vllm_config, device):
                return layer(hidden_states, router_logits)

        full_forward_us = time_cuda(forward_once, args.warmup, args.iters)
        local_latency = torch.tensor(
            [full_forward_us], dtype=torch.float64, device=device
        )
        rank_latencies_tensor = [
            torch.empty_like(local_latency) for _ in range(args.world_size)
        ]
        torch.distributed.all_gather(rank_latencies_tensor, local_latency)
        rank_forward_us = [latency.item() for latency in rank_latencies_tensor]
        run_section_profile(args, rank, forward_once)
        run_torch_kernel_profile(args, rank, forward_once)

        topk_us: float | None = None
        dispatch_us: float | None = None
        combine_us: float | None = None

        def topk_once():
            return layer.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=layer._quant_method.topk_indices_dtype,
            )

        topk_us = time_cuda(topk_once, args.warmup, args.iters)
        topk_weights, topk_ids = topk_once()
        (
            expert_tokens_min,
            expert_tokens_max,
            expert_tokens_mean,
            expert_tokens_cv,
            expert_tokens_zero,
        ) = expert_token_stats(topk_ids, args.num_experts)
        torch.accelerator.synchronize()

        if args.world_size > 1:

            def dispatch_once():
                with make_forward_context(args, vllm_config, device):
                    return get_ep_group().dispatch(
                        hidden_states,
                        topk_weights,
                        topk_ids,
                        is_sequence_parallel=False,
                    )

            try:
                dispatch_us = time_cuda(dispatch_once, args.warmup, args.iters)
                with make_forward_context(args, vllm_config, device):
                    dispatched_hidden, _, _ = get_ep_group().dispatch(
                        hidden_states,
                        topk_weights,
                        topk_ids,
                        is_sequence_parallel=False,
                    )
                combine_input = torch.randn_like(dispatched_hidden)
                torch.accelerator.synchronize()

                def combine_once():
                    with make_forward_context(args, vllm_config, device):
                        return get_ep_group().combine(
                            combine_input,
                            is_sequence_parallel=False,
                        )

                combine_us = time_cuda(combine_once, args.warmup, args.iters)
            except NotImplementedError:
                dispatch_us = None
                combine_us = None

        result = BenchResult(
            rank=rank,
            world_size=args.world_size,
            backend=args.backend,
            tokens=args.tokens,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_experts=args.num_experts,
            local_experts=local_experts,
            top_k=args.top_k,
            dtype=str(hidden_states.dtype),
            warmup=args.warmup,
            iters=args.iters,
            full_forward_us=full_forward_us,
            rank0_forward_us=rank_forward_us[0],
            rank1_forward_us=rank_forward_us[1] if args.world_size > 1 else None,
            critical_path_us=max(rank_forward_us),
            topk_us=topk_us,
            dispatch_us=dispatch_us,
            combine_us=combine_us,
            expert_tokens_min=expert_tokens_min,
            expert_tokens_max=expert_tokens_max,
            expert_tokens_mean=expert_tokens_mean,
            expert_tokens_cv=expert_tokens_cv,
            expert_tokens_zero=expert_tokens_zero,
        )

        torch.distributed.barrier()
        if rank == 0:
            row = asdict(result)
            if args.csv:
                print(",".join(row.keys()))
                print(",".join("" if v is None else str(v) for v in row.values()))
            else:
                print(json.dumps(row, indent=2, sort_keys=True))
    finally:
        torch.accelerator.synchronize()
        cleanup_dist_env_and_memory()


def main() -> None:
    args = parse_args()
    if args.num_experts % args.world_size != 0:
        raise ValueError("--num-experts must be divisible by --world-size")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"Need {args.world_size} CUDA devices, found {torch.cuda.device_count()}"
        )

    vllm_config = make_vllm_config(args)
    if args.world_size == 1:
        run_worker(0, args, vllm_config)
    else:
        spawn(run_worker, args=(args, vllm_config), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
