'''
Modify from conda/envs/fluxinpaint/lib/python3.10/site-packages/diffusers/models/attention_processor.py
class FluxAttnProcessor2_0:

'''
import inspect
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..image_processor import IPAdapterMaskProcessor
from ..utils import deprecate, logging
from ..utils.import_utils import is_torch_npu_available, is_xformers_available
from ..utils.torch_utils import is_torch_version, maybe_allow_in_graph


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_npu_available():
    import torch_npu

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        input_ndim = hidden_states.ndim  # 3
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim  # 3
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]  # 2
        # import pdb;pdb.set_trace()

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]  # 3072
        head_dim = inner_dim // attn.heads  # 24个头, =128

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # torch.Size([2, 24, 6624, 128])
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # TODO修改部分，加入λ
        # 划分左图和右图
        left_indices = []
        right_indices = []  
        for i in range(72):  # 行数  height // 16
            start_idx = i * 92  # 当前行的起始索引  weight // 16
            left_indices.extend(range(start_idx, start_idx + 46))  # 左图索引
            right_indices.extend(range(start_idx + 46, start_idx + 92))  # 右图索引

        # 确保所有格子都已被覆盖
        assert len(left_indices) + len(right_indices) == query.shape[-2], f"预期的索引数量之和（{len(left_indices) + len(right_indices)}）与查询张量query的倒数第二个维度（{query.shape[-2]}）不匹配！"  # 6624个格子
        # 转换为张量
        left_indices = torch.tensor(left_indices)
        right_indices = torch.tensor(right_indices)

        # 提取左图和右图的 query 和 key 部分
        left_query = query[:, :, left_indices, :]
        right_query = query[:, :, right_indices, :]
        left_key = key[:, :, left_indices, :]
        right_key = key[:, :, right_indices, :]
        # 假设 sqrt_lambda 已经定义，作为标量
        lambda_value = 1.3
        sqrt_lambda = torch.sqrt(torch.tensor(lambda_value))
        # 对左图的 key 部分乘以 sqrt_lambda
        key[:, :, left_indices, :] *= sqrt_lambda
        # 对右图的 query 部分乘以 sqrt_lambda
        query[:, :, right_indices, :] *= sqrt_lambda
        
        
        
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)  
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(  # torch.Size([2, 24, 512, 128])
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)  # 对 query 和 key 张量进行旋转位置编码的变换
        
        
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states