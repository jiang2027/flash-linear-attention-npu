# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
"""ATK executor for PyAclnn vs CPU golden and original NPU Triton.

Run with jq311:

    atk node --backend pyaclnn --devices 0 \
        node --backend triton --devices 0 \
        node --backend cpu \
        task -c <case.json> \
        -p executor_causal_conv1d_bwd_dual_benchmark.py \
        --task accuracy
"""
from pathlib import Path
import os
import shutil
import sys
import tempfile

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.triton_base_api import TritonBaseApi

# Import the existing executor to register the CPU golden, PyAclnn API, and
# accuracy standard. Its helpers also keep all three backends on identical data.
from executor_causal_conv1d_bwd import (  # noqa: F401
    DTYPE_MAP,
    _input_to_logical,
    _layout_meta,
    _logical_to_input,
    _normalize_layout,
    causal_conv1d_preactivation,
)


REPO_ROOT = Path(__file__).resolve().parents[8]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRITON_BLOCK_D = 128
TRITON_HEAD_DIM = 128
_TRITON_CANN9_COMPAT_INSTALLED = False


def _round_up(value, alignment):
    return ((int(value) + alignment - 1) // alignment) * alignment


def _activation_name_for_triton(activation):
    activation = int(activation)
    if activation == 0:
        return None
    if activation == 1:
        return "silu"
    if activation == 2:
        return "swish"
    raise ValueError(f"unsupported activation: {activation}")


def _query_start_loc_list(query_start_loc, total_tokens):
    if query_start_loc is None or isinstance(query_start_loc, int):
        return [0, int(total_tokens)]
    if torch.is_tensor(query_start_loc):
        return [
            int(value)
            for value in query_start_loc.detach().cpu().flatten().tolist()
        ]
    return [int(value) for value in query_start_loc]


def _pad_last_dim(tensor, padded_dim):
    pad = int(padded_dim) - int(tensor.shape[-1])
    if pad < 0:
        raise ValueError(
            f"padded_dim={padded_dim} is smaller than D={tensor.shape[-1]}"
        )
    return F.pad(tensor, (0, pad)).contiguous() if pad else tensor.contiguous()


def _install_triton_cann9_compat():
    """Patch the Triton runtime extension source for the CANN 9 enum rename."""
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
        raise RuntimeError(
            "bishengir-compile was not found. Activate the jq311 environment "
            "or add the matching CANN compiler to PATH."
        )
    compiler_dir = str(candidates[0].parent)
    os.environ["PATH"] = compiler_dir + os.pathsep + os.environ["PATH"]


def _load_original_triton_kernel():
    _ensure_bishengir_compiler()
    _install_triton_cann9_compat()
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "npu":
        raise RuntimeError(f"NPU Triton backend is required, got {target.backend}")

    from fla.ops.triton.triton_core.causal_conv1d import (
        causal_conv1d_bwd_kernel,
        get_num_cores,
        prepare_chunk_indices,
    )

    return causal_conv1d_bwd_kernel, get_num_cores, prepare_chunk_indices


@triton.jit
def _causal_conv1d_bwd_state_kernel(
    dy,
    y,
    weight,
    initial_state,
    dht,
    dx,
    dw_state,
    dh0,
    cu_seqlens,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_d = tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        t_len = eos - bos
    else:
        bos = (i_n * T).to(tl.int64)
        t_len = T

    for i_w in tl.static_range(1, W):
        dw_acc = tl.zeros((BD,), dtype=tl.float32)
        for row in tl.static_range(0, W):
            if row < i_w:
                row_valid = row < t_len
                dy_row = tl.load(
                    dy + (bos + row) * D + o_d,
                    mask=row_valid & m_d,
                    other=0.0,
                ).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    y_row = tl.load(
                        y + (bos + row) * D + o_d,
                        mask=row_valid & m_d,
                        other=0.0,
                    ).to(tl.float32)
                    sig = tl.sigmoid(y_row)
                    dy_row *= sig * (1.0 + y_row * (1.0 - sig))
                state_row = tl.load(
                    initial_state
                    + i_n * D * W
                    + o_d * W
                    + W
                    - i_w
                    + row,
                    mask=row_valid & m_d,
                    other=0.0,
                ).to(tl.float32)
                dw_acc += dy_row * state_row

        tl.store(
            dw_state + i_n * W * D + (W - 1 - i_w) * D + o_d,
            dw_acc,
            mask=m_d,
        )

    for slot in tl.static_range(1, W):
        dh0_acc = tl.zeros((BD,), dtype=tl.float32)
        for row in tl.static_range(0, W):
            if row < slot:
                row_valid = row < t_len
                dy_row = tl.load(
                    dy + (bos + row) * D + o_d,
                    mask=row_valid & m_d,
                    other=0.0,
                ).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    y_row = tl.load(
                        y + (bos + row) * D + o_d,
                        mask=row_valid & m_d,
                        other=0.0,
                    ).to(tl.float32)
                    sig = tl.sigmoid(y_row)
                    dy_row *= sig * (1.0 + y_row * (1.0 - sig))
                weight_row = tl.load(
                    weight + (slot - 1 - row) * D + o_d,
                    mask=m_d,
                    other=0.0,
                ).to(tl.float32)
                dh0_acc += dy_row * weight_row
        tl.store(
            dh0 + i_n * D * W + o_d * W + slot,
            dh0_acc,
            mask=m_d,
        )

    start_tok = tl.maximum(0, t_len - (W - 1))
    for row in tl.static_range(0, W - 1):
        token = start_tok + row
        token_valid = token < t_len
        dx_offset = (bos + token) * D + o_d
        dx_row = tl.load(
            dx + dx_offset,
            mask=token_valid & m_d,
            other=0.0,
        ).to(tl.float32)
        dht_row = tl.load(
            dht + i_n * D * W + o_d * W + row + 1,
            mask=token_valid & m_d,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            dx + dx_offset,
            (dx_row + dht_row).to(dx.dtype.element_ty),
            mask=token_valid & m_d,
        )


def causal_conv1d_bwd_triton_reference(
    x,
    y,
    weight,
    dy,
    initial_state,
    dht,
    query_start_loc,
    activation,
    input_layout,
):
    """Run the repository's original NPU Triton backward kernel."""
    layout = _normalize_layout(input_layout)
    input_dtype = x.dtype
    original_dim = int(x.shape[-1])
    padded_dim = _round_up(original_dim, TRITON_BLOCK_D)
    num_heads = padded_dim // TRITON_HEAD_DIM

    x_logical = x.contiguous()
    y_logical = _input_to_logical(y, layout)
    dy_logical = _input_to_logical(dy, layout)

    is_varlen = x_logical.dim() == 2
    if is_varlen:
        total_tokens = int(x_logical.shape[0])
        x_logical = x_logical.unsqueeze(0)
        y_logical = y_logical.unsqueeze(0)
        dy_logical = dy_logical.unsqueeze(0)
        qsl = _query_start_loc_list(query_start_loc, total_tokens)
        if len(qsl) < 2 or qsl[0] != 0 or qsl[-1] != total_tokens:
            raise ValueError(
                "queryStartLoc must start at 0 and end at totalTokens, "
                f"got {qsl} for totalTokens={total_tokens}"
            )
        cu_seqlens = torch.tensor(qsl, dtype=torch.int64, device=x.device)
    else:
        cu_seqlens = None

    batch, seqlen, _ = x_logical.shape
    width = int(weight.shape[0])

    x_pad = _pad_last_dim(x_logical, padded_dim)
    y_pad = _pad_last_dim(y_logical, padded_dim)
    dy_pad = _pad_last_dim(dy_logical, padded_dim)
    weight_pad = _pad_last_dim(weight, padded_dim)

    # The original kernel consumes dy as [B, H, T, K]. K=128 keeps
    # HEADS_PER_BLOCK integral.
    dy_triton = (
        dy_pad.reshape(batch, seqlen, num_heads, TRITON_HEAD_DIM)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    # Public states are [N, W, D]. The original Triton kernel indexes
    # [N, D, W], so state-only corrections use the transposed representation.
    state_triton = (
        _pad_last_dim(initial_state, padded_dim).transpose(1, 2).contiguous()
    )
    dht_triton = (
        _pad_last_dim(dht, padded_dim).transpose(1, 2).contiguous()
    )

    kernel, get_num_cores, prepare_chunk_indices = _load_original_triton_kernel()
    num_cores = get_num_cores()
    block_t = min(
        4,
        triton.next_power_of_2(
            triton.cdiv(max(16, batch * seqlen), num_cores)
        ),
    )
    num_blocks_d = triton.cdiv(padded_dim, TRITON_BLOCK_D)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, block_t)
        num_chunks = len(chunk_indices)
    else:
        chunk_indices = None
        num_chunks = triton.cdiv(seqlen, block_t) * batch

    dx = torch.empty_like(x_pad)
    dw_partial = torch.empty(
        (num_chunks, width, padded_dim),
        dtype=torch.float32,
        device=x.device,
    )
    db_partial = torch.empty(
        (num_chunks, padded_dim),
        dtype=torch.float32,
        device=x.device,
    )

    kernel[(num_cores,)](
        x=x_pad,
        y=y_pad,
        weight=weight_pad,
        initial_state=None,
        dh0=None,
        dht=None,
        dy=dy_triton,
        dx=dx,
        dw=dw_partial,
        db=db_partial,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=batch,
        T=seqlen,
        D=padded_dim,
        H=num_heads,
        W=width,
        BT=block_t,
        BD=TRITON_BLOCK_D,
        ACTIVATION=_activation_name_for_triton(activation),
        NUM_Blk_D=num_blocks_d,
        NUM_CHKS=num_chunks,
        multibuffer=False,
    )

    state_batch = int(state_triton.shape[0])
    dw_state = torch.zeros(
        (state_batch, width, padded_dim),
        dtype=torch.float32,
        device=x.device,
    )
    dh0 = torch.zeros_like(state_triton, dtype=torch.float32)
    _causal_conv1d_bwd_state_kernel[
        (state_batch, triton.cdiv(padded_dim, TRITON_BLOCK_D))
    ](
        dy=dy_pad,
        y=y_pad,
        weight=weight_pad,
        initial_state=state_triton,
        dht=dht_triton,
        dx=dx,
        dw_state=dw_state,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        T=seqlen,
        D=padded_dim,
        W=width,
        BD=TRITON_BLOCK_D,
        ACTIVATION=_activation_name_for_triton(activation),
        IS_VARLEN=cu_seqlens is not None,
        multibuffer=False,
    )

    dw = dw_partial.sum(0) + dw_state.sum(0)
    dw = dw.to(input_dtype)[:, :original_dim].contiguous()
    db = db_partial.sum(0).to(input_dtype)[:original_dim].contiguous()
    dx = dx[..., :original_dim]
    if is_varlen:
        dx = dx.squeeze(0)
    dx = dx.contiguous()

    dh0 = (
        dh0.to(input_dtype)
        .transpose(1, 2)[..., :original_dim]
        .contiguous()
    )

    return dx, dw, db, dh0


@register("triton_causal_conv1d_bwd")
class CausalConv1dBwdTritonApi(TritonBaseApi):
    def __init__(self, task_result: TaskResult):
        # The reference imports the repository kernel directly, so ATK does
        # not need triton_name/triton_ut_path based function discovery.
        BaseApi.__init__(self, task_result)

    def _target_input_dtype(self):
        case_config = getattr(self.task_result, "case_config", None)
        inputs = getattr(case_config, "inputs", None) or []
        for input_config in inputs:
            if getattr(input_config, "name", None) == "x":
                dtype = DTYPE_MAP.get(
                    str(getattr(input_config, "dtype", "")).lower()
                )
                if dtype is not None:
                    return dtype
        return torch.float32

    def init_by_input_data(self, input_data: InputDataset):
        dtype = self._target_input_dtype()
        layout = _normalize_layout(
            input_data.kwargs.get("inputLayout", "BSND")
        )
        x = input_data.kwargs["x"].to(dtype).contiguous()
        weight = input_data.kwargs["weight"].to(dtype).contiguous()
        dy = input_data.kwargs["dy"].to(dtype).contiguous()
        _, t, _, n_heads, head_dim = _layout_meta(dy, layout)
        activation = int(input_data.kwargs["activation"])

        initial_state = (
            input_data.kwargs["initial_state"].to(dtype).contiguous()
        )
        dht = input_data.kwargs["dht"].to(dtype).contiguous()

        if activation == 0:
            y = torch.zeros_like(dy)
        else:
            if x.dim() == 2:
                y_logical = causal_conv1d_preactivation(
                    x.detach().cpu().unsqueeze(0),
                    weight.detach().cpu(),
                    initial_state.detach().cpu(),
                ).squeeze(0)
            else:
                y_logical = causal_conv1d_preactivation(
                    x.detach().cpu(),
                    weight.detach().cpu(),
                    initial_state.detach().cpu(),
                )
            y = _logical_to_input(
                y_logical.to(dtype), layout, n_heads, head_dim
            )

        input_data.kwargs["x"] = x.npu()
        input_data.kwargs["y"] = y.to(dtype).contiguous().npu()
        input_data.kwargs["weight"] = weight.npu()
        input_data.kwargs["dy"] = dy.npu()
        input_data.kwargs["initial_state"] = initial_state.npu()
        input_data.kwargs["dht"] = dht.npu()
        input_data.kwargs["queryStartLoc"] = [0, int(t)]
        input_data.kwargs["activation"] = activation
        input_data.kwargs["inputLayout"] = layout

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        return causal_conv1d_bwd_triton_reference(
            input_data.kwargs["x"],
            input_data.kwargs["y"],
            input_data.kwargs["weight"],
            input_data.kwargs["dy"],
            input_data.kwargs["initial_state"],
            input_data.kwargs["dht"],
            input_data.kwargs["queryStartLoc"],
            input_data.kwargs["activation"],
            input_data.kwargs.get("inputLayout", "BSND"),
        )
