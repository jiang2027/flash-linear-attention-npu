# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
"""ATK executor for CPU golden, CUDA causal-conv1d, and torch NPU paths.

This file registers the normal ``executor_causal_conv1d_bwd`` api_type, so the
existing generated json can be reused with ``-p``.  PyAclnn support is kept by
importing the default executor, which registers ``aclnn_causal_conv1d_bwd``.
"""

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

from executor_causal_conv1d_bwd import (  # noqa: F401
    CausalConv1dBwdAclnnApi as _CausalConv1dBwdAclnnApi,
    DTYPE_MAP,
    _input_to_logical,
    _layout_meta,
    _logical_to_input,
    _normalize_layout,
    causal_conv1d_bwd_cpu,
    causal_conv1d_preactivation,
)


def _query_start_loc_list(query_start_loc, total_tokens):
    if query_start_loc is None or isinstance(query_start_loc, int):
        return [0, int(total_tokens)]
    if torch.is_tensor(query_start_loc):
        return [int(v) for v in query_start_loc.detach().cpu().flatten().tolist()]
    return [int(v) for v in query_start_loc]


def _target_torch_device(api):
    device = str(getattr(api, "device", "cpu")).lower()
    device_id = getattr(api, "device_id", None)

    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("ATK backend is gpu, but torch.cuda is not available")
        if device_id is not None:
            torch.cuda.set_device(int(device_id))
            return torch.device(f"cuda:{int(device_id)}")
        return torch.device("cuda")

    if device == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("ATK backend is npu, but torch_npu is not importable") from exc
        if device_id is not None:
            torch.npu.set_device(int(device_id))
            return torch.device(f"npu:{int(device_id)}")
        return torch.device("npu")

    return torch.device("cpu")


def _to_device_tensor(tensor, dtype, device):
    return tensor.to(device=device, dtype=dtype).contiguous()


def _make_preactivation_y(x, weight, initial_state, query_start_loc, input_layout):
    layout = _normalize_layout(input_layout)
    if layout not in ("TND", "NTD"):
        return causal_conv1d_preactivation(x, weight, initial_state)

    total_tokens = int(x.shape[0])
    qsl = _query_start_loc_list(query_start_loc, total_tokens)
    y = torch.empty((total_tokens, x.shape[-1]), dtype=torch.float32, device=x.device)

    for batch_idx in range(len(qsl) - 1):
        start = int(qsl[batch_idx])
        end = int(qsl[batch_idx + 1])
        if end <= start:
            continue
        state_b = initial_state[batch_idx : batch_idx + 1] if initial_state is not None else None
        y[start:end] = causal_conv1d_preactivation(
            x[start:end].unsqueeze(0), weight, state_b
        ).squeeze(0)

    return y


def _activation_is_silu(activation):
    activation = int(activation)
    if activation == 0:
        return False
    if activation in (1, 2):
        return True
    raise ValueError(f"unsupported activation: {activation}")


def _state_to_cuda(state):
    if state is None:
        return None
    return state[:, 1:, :].permute(0, 2, 1)


def _dh0_from_cuda(dinitial_states, state_template):
    dh0 = torch.zeros_like(state_template)
    if dinitial_states is not None and state_template.shape[1] > 1:
        dh0[:, 1:, :] = dinitial_states.permute(0, 2, 1).contiguous()
    return dh0


def _cuda_bwd_fixed(x, weight, dy, initial_state, dht, activation):
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function

    x_cuda = x.permute(0, 2, 1)
    dy_cuda = dy.permute(0, 2, 1)
    weight_cuda = weight.transpose(0, 1).contiguous()
    bias = torch.zeros((weight.shape[1],), dtype=weight.dtype, device=weight.device)
    initial_states = _state_to_cuda(initial_state)
    dfinal_states = _state_to_cuda(dht)

    dx_cuda, dweight_cuda, dbias, dinitial_states = causal_conv1d_bwd_function(
        x=x_cuda,
        weight=weight_cuda,
        bias=bias,
        dout=dy_cuda,
        seq_idx=None,
        initial_states=initial_states,
        dfinal_states=dfinal_states,
        dx=None,
        return_dinitial_states=True,
        silu_activation=_activation_is_silu(activation),
    )

    dx = dx_cuda.permute(0, 2, 1).contiguous()
    dw = dweight_cuda.transpose(0, 1).contiguous().to(weight.dtype)
    db = dbias.contiguous().to(weight.dtype)
    dh0 = _dh0_from_cuda(dinitial_states, initial_state)
    return dx, dw, db, dh0


def _run_cuda_causal_conv1d(input_data):
    layout = _normalize_layout(input_data.kwargs.get("inputLayout", "BSND"))
    x = input_data.kwargs["x"]
    weight = input_data.kwargs["weight"]
    dy_logic = _input_to_logical(input_data.kwargs["dy"], layout)
    initial_state = input_data.kwargs["initial_state"]
    dht = input_data.kwargs["dht"]
    activation = input_data.kwargs["activation"]

    if x.dim() == 3:
        return _cuda_bwd_fixed(x, weight, dy_logic, initial_state, dht, activation)

    total_tokens, dim = x.shape
    qsl = _query_start_loc_list(input_data.kwargs.get("queryStartLoc"), total_tokens)
    width = int(weight.shape[0])
    dtype = x.dtype
    dx = torch.empty_like(x)
    dw_acc = torch.zeros((width, dim), dtype=torch.float32, device=x.device)
    db_acc = torch.zeros((dim,), dtype=torch.float32, device=x.device)
    dh0 = torch.zeros((len(qsl) - 1, width, dim), dtype=dtype, device=x.device)

    for batch_idx in range(len(qsl) - 1):
        start = int(qsl[batch_idx])
        end = int(qsl[batch_idx + 1])
        if end <= start:
            continue
        dx_b, dw_b, db_b, dh0_b = _cuda_bwd_fixed(
            x[start:end].unsqueeze(0),
            weight,
            dy_logic[start:end].unsqueeze(0),
            initial_state[batch_idx : batch_idx + 1],
            dht[batch_idx : batch_idx + 1],
            activation,
        )
        dx[start:end] = dx_b.squeeze(0)
        dw_acc += dw_b.to(torch.float32)
        db_acc += db_b.to(torch.float32)
        dh0[batch_idx : batch_idx + 1] = dh0_b

    return dx.contiguous(), dw_acc.to(weight.dtype).contiguous(), db_acc.to(weight.dtype).contiguous(), dh0


def _run_npu_fla_op(input_data):
    import fla_npu  # noqa: F401

    if not hasattr(torch.ops.npu, "npu_causal_conv1d_bwd"):
        raise RuntimeError("torch.ops.npu.npu_causal_conv1d_bwd is not registered")

    return torch.ops.npu.npu_causal_conv1d_bwd(
        input_data.kwargs["x"],
        input_data.kwargs["y"],
        input_data.kwargs["weight"],
        input_data.kwargs["dy"],
        input_data.kwargs["initial_state"],
        input_data.kwargs["dht"],
        input_data.kwargs["queryStartLoc"],
        input_data.kwargs["activation"],
        input_data.kwargs["inputLayout"],
    )


@register("executor_causal_conv1d_bwd")
class CausalConv1dBwdTorchDeviceApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(CausalConv1dBwdTorchDeviceApi, self).__init__(task_result)

    def _target_input_dtype(self):
        case_config = getattr(self.task_result, "case_config", None)
        inputs = getattr(case_config, "inputs", None) or []
        for input_config in inputs:
            if getattr(input_config, "name", None) == "x":
                dtype = DTYPE_MAP.get(str(getattr(input_config, "dtype", "")).lower())
                if dtype is not None:
                    return dtype
        return torch.float32

    def init_by_input_data(self, input_data: InputDataset):
        dtype = self._target_input_dtype()
        device = _target_torch_device(self)
        input_layout = _normalize_layout(input_data.kwargs.get("inputLayout", "BSND"))

        x = _to_device_tensor(input_data.kwargs["x"], dtype, device)
        weight = _to_device_tensor(input_data.kwargs["weight"], dtype, device)
        dy = _to_device_tensor(input_data.kwargs["dy"], dtype, device)
        initial_state = _to_device_tensor(input_data.kwargs["initial_state"], dtype, device)
        dht = _to_device_tensor(input_data.kwargs["dht"], dtype, device)
        activation = int(input_data.kwargs["activation"])

        b, t, _, n_heads, head_dim = _layout_meta(dy, input_layout)
        total_tokens = x.shape[0] if x.dim() == 2 else x.shape[1]
        query_start_loc = _query_start_loc_list(
            input_data.kwargs.get("queryStartLoc"), total_tokens
        )

        if activation == 0:
            y = torch.zeros_like(dy)
        else:
            x_logic = x if x.dim() == 2 else x.reshape(b, t, -1).contiguous()
            y_logic = _make_preactivation_y(
                x_logic, weight, initial_state, query_start_loc, input_layout
            ).to(dtype)
            y = _logical_to_input(y_logic, input_layout, n_heads, head_dim).contiguous()

        input_data.kwargs["x"] = x
        input_data.kwargs["y"] = y
        input_data.kwargs["weight"] = weight
        input_data.kwargs["dy"] = dy
        input_data.kwargs["initial_state"] = initial_state
        input_data.kwargs["dht"] = dht
        input_data.kwargs["queryStartLoc"] = query_start_loc
        input_data.kwargs["activation"] = activation
        input_data.kwargs["inputLayout"] = input_layout

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            return _run_cuda_causal_conv1d(input_data)
        if self.device == "npu":
            return _run_npu_fla_op(input_data)
        return causal_conv1d_bwd_cpu(
            input_data.kwargs["x"],
            input_data.kwargs["y"],
            input_data.kwargs["weight"],
            input_data.kwargs["dy"],
            input_data.kwargs["initial_state"],
            input_data.kwargs["dht"],
            input_data.kwargs["queryStartLoc"],
            input_data.kwargs["activation"],
            input_data.kwargs.get("inputLayout", "BSND"),
            self._target_input_dtype(),
        )

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND
