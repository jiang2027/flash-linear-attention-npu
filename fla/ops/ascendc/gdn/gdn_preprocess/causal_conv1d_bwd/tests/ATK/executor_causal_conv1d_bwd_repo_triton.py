# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
"""ATK executor that calls the repository Triton bwd implementation directly."""
from pathlib import Path
import shutil
import sys
import tempfile

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.triton_base_api import TritonBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

from executor_causal_conv1d_bwd import (
    DTYPE_MAP,
    CausalConv1dBwdAclnnApi,
    _layout_meta,
    _logical_to_input,
    _normalize_layout,
    causal_conv1d_bwd_cpu,
    causal_conv1d_preactivation,
)


REPO_ROOT = Path(__file__).resolve().parents[8]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TRITON_CANN9_COMPAT_INSTALLED = False


def _target_input_dtype(task_result):
    case_config = getattr(task_result, "case_config", None)
    inputs = getattr(case_config, "inputs", None) or []
    for input_config in inputs:
        if getattr(input_config, "name", None) == "x":
            dtype = DTYPE_MAP.get(
                str(getattr(input_config, "dtype", "")).lower()
            )
            if dtype is not None:
                return dtype
    return torch.float32


def _activation_name_for_triton(activation):
    activation = int(activation)
    if activation == 0:
        return None
    if activation == 1:
        return "silu"
    if activation == 2:
        return "swish"
    raise ValueError(f"unsupported activation: {activation}")


def _validate_repo_supported(x, dy, weight, input_layout):
    layout = _normalize_layout(input_layout)
    if layout != "BNSD":
        raise ValueError(
            "repository Triton bwd expects x=[B,T,D] and dy=[B,H,T,K]; "
            f"only BNSD cases are supported, got {layout}"
        )
    if x.dim() != 3 or dy.dim() != 4:
        raise ValueError(
            f"repository Triton bwd expects x rank 3 and dy rank 4, "
            f"got x={tuple(x.shape)}, dy={tuple(dy.shape)}"
        )
    b, _, d = x.shape
    bd, heads, _, head_dim = dy.shape
    if b != bd or d != heads * head_dim:
        raise ValueError(
            f"dy shape must be [B,H,T,D/H], got x={tuple(x.shape)}, "
            f"dy={tuple(dy.shape)}"
        )
    if d != int(weight.shape[-1]):
        raise ValueError(
            f"weight last dim must equal D, got D={d}, weight={tuple(weight.shape)}"
        )
    if d % 512 != 0:
        raise ValueError(f"repository Triton bwd requires D % 512 == 0, got D={d}")
    if head_dim > 512 or 512 % head_dim != 0:
        raise ValueError(
            "repository Triton bwd uses BD=512 and requires head_dim to divide 512, "
            f"got head_dim={head_dim}"
        )


def _prepare_supported_input(input_data: InputDataset, dtype):
    input_layout = _normalize_layout(input_data.kwargs.get("inputLayout", "BNSD"))
    x = input_data.kwargs["x"].to(dtype).contiguous()
    weight = input_data.kwargs["weight"].to(dtype).contiguous()
    dy = input_data.kwargs["dy"].to(dtype).contiguous()
    _validate_repo_supported(x, dy, weight, input_layout)

    batch, seqlen, dim, n_heads, head_dim = _layout_meta(dy, input_layout)
    width = int(weight.shape[0])
    activation = int(input_data.kwargs["activation"])

    # ATK/pyaclnn type checking requires actual aclTensor pointers for optional
    # tensor slots. Zero state/dht is equivalent to no-state for dx/dw/db, while
    # the Triton backend below still calls the repository no-state branch.
    initial_state = torch.zeros(
        (batch, width, dim), dtype=dtype, device=x.device
    ).contiguous()
    dht = torch.zeros_like(initial_state)

    if activation == 0:
        y = torch.zeros_like(dy)
    else:
        y_logical = causal_conv1d_preactivation(
            x.detach().cpu(),
            weight.detach().cpu(),
            initial_state.detach().cpu(),
        ).to(dtype)
        y = _logical_to_input(
            y_logical, input_layout, n_heads, head_dim
        ).contiguous()

    input_data.kwargs["x"] = x
    input_data.kwargs["y"] = y.to(dtype).contiguous()
    input_data.kwargs["weight"] = weight
    input_data.kwargs["dy"] = dy
    input_data.kwargs["initial_state"] = initial_state
    input_data.kwargs["dht"] = dht
    input_data.kwargs["queryStartLoc"] = [0, int(seqlen)]
    input_data.kwargs["activation"] = activation
    input_data.kwargs["inputLayout"] = input_layout


def _install_triton_cann9_compat():
    """Patch only the temp-built Triton runtime extension source for CANN 9."""
    global _TRITON_CANN9_COMPAT_INSTALLED
    if _TRITON_CANN9_COMPAT_INSTALLED:
        return

    from triton.backends import backends

    driver_class = backends["ascend"].driver
    driver_globals = driver_class.__init__.__globals__
    original_build = driver_globals["_build_npu_ext"]
    old_enum = "RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE"
    new_enum = "RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE"
    compat_root = Path("/dev/shm/workspace")
    compat_root.mkdir(parents=True, exist_ok=True)

    def build_npu_ext_compat(
        obj_name,
        header_path,
        src_path,
        *,
        kernel_launcher="torch",
        precompile=False,
    ):
        source_path = Path(src_path)
        source = source_path.read_text()
        if obj_name == "npu_utils" and old_enum in source:
            compat_dir = Path(
                tempfile.mkdtemp(
                    prefix="triton_cann9_compat_",
                    dir=compat_root,
                )
            )
            source_path = compat_dir / source_path.name
            source_path.write_text(source.replace(old_enum, new_enum))
        return original_build(
            obj_name,
            header_path,
            str(source_path),
            kernel_launcher=kernel_launcher,
            precompile=precompile,
        )

    driver_globals["_build_npu_ext"] = build_npu_ext_compat
    _TRITON_CANN9_COMPAT_INSTALLED = True


def _ensure_bishengir_compiler():
    if shutil.which("bishengir-compile") is not None:
        return

    compiler_pattern = "cann-*/tools/bishengir/bin/bishengir-compile"
    candidates = sorted(
        (Path(sys.prefix) / "Ascend").glob(compiler_pattern),
        reverse=True,
    )
    if not candidates:
        return

    bin_dir = str(candidates[0].parent)
    import os

    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"


@register("executor_causal_conv1d_bwd_repo_supported")
class CausalConv1dBwdRepoSupportedCpuApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        _prepare_supported_input(input_data, _target_input_dtype(self.task_result))

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        dx, dw, db, _ = causal_conv1d_bwd_cpu(
            input_data.kwargs["x"],
            input_data.kwargs["y"],
            input_data.kwargs["weight"],
            input_data.kwargs["dy"],
            input_data.kwargs["initial_state"],
            input_data.kwargs["dht"],
            input_data.kwargs["queryStartLoc"],
            input_data.kwargs["activation"],
            input_data.kwargs["inputLayout"],
            _target_input_dtype(self.task_result),
        )
        dh0 = torch.zeros_like(input_data.kwargs["initial_state"])
        return dx, dw, db, dh0


@register("aclnn_causal_conv1d_bwd_repo_supported")
class CausalConv1dBwdRepoSupportedAclnnApi(CausalConv1dBwdAclnnApi):
    def __init__(self, task_result: TaskResult, backend):
        super().__init__(task_result, backend)
        self.input = None

    def init_by_input_data(self, input_data: InputDataset):
        _prepare_supported_input(input_data, _target_input_dtype(self.task_result))
        self.input = input_data
        return super().init_by_input_data(input_data)

    def after_call(self, output_packages):
        dx, dw, db, _ = super().after_call(output_packages)
        dh0 = torch.zeros_like(self.input.kwargs["initial_state"].cpu())
        return dx, dw, db, dh0

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND


@register("triton_causal_conv1d_bwd_repo")
class CausalConv1dBwdRepoTritonApi(TritonBaseApi):
    def __init__(self, task_result: TaskResult):
        # The executor imports the repository function directly; ATK does not
        # need triton_name/triton_ut_path discovery.
        BaseApi.__init__(self, task_result)

    def init_by_input_data(self, input_data: InputDataset):
        dtype = _target_input_dtype(self.task_result)
        _prepare_supported_input(input_data, dtype)
        input_data.kwargs["x"] = input_data.kwargs["x"].npu()
        input_data.kwargs["y"] = input_data.kwargs["y"].npu()
        input_data.kwargs["weight"] = input_data.kwargs["weight"].npu()
        input_data.kwargs["dy"] = input_data.kwargs["dy"].npu()
        input_data.kwargs["initial_state"] = None
        input_data.kwargs["dht"] = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        _ensure_bishengir_compiler()
        _install_triton_cann9_compat()
        from fla.ops.triton.triton_core.causal_conv1d import causal_conv1d_bwd_impl

        dy = input_data.kwargs["dy"]
        heads = int(dy.shape[1])
        dim = int(input_data.kwargs["x"].shape[-1])
        bias = torch.zeros(
            (dim,), dtype=input_data.kwargs["weight"].dtype, device=dy.device
        )

        dx, dw, db, _, _ = causal_conv1d_bwd_impl(
            x=input_data.kwargs["x"],
            dy=dy,
            H=heads,
            dht=input_data.kwargs["dht"],
            weight=input_data.kwargs["weight"],
            bias=bias,
            residual=None,
            initial_state=input_data.kwargs["initial_state"],
            activation=_activation_name_for_triton(input_data.kwargs["activation"]),
            cu_seqlens=None,
        )
        batch = int(input_data.kwargs["x"].shape[0])
        width = int(input_data.kwargs["weight"].shape[0])
        dh0 = torch.zeros(
            (batch, width, dim),
            dtype=input_data.kwargs["x"].dtype,
            device=dx.device,
        )
        return dx, dw, db, dh0
