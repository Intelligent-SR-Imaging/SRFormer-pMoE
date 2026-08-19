import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.nn import functional as F

from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange

import math
import numpy as np
from model.SFFB3D import SFFB3D, SFFB3D_complex_conv, SFFB2D

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class PixelShuffle3d(nn.Module):
    def __init__(self,z_scale, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.z_scale = z_scale

    def forward(self, x):
        B, C, Z, X, Y = x.size()
        outC = C // (self.x_scale * self.y_scale * self.z_scale)
        outX = X * self.x_scale
        outY = Y * self.y_scale
        outZ = Z * self.z_scale
        x_view = x.contiguous().view(B, outC, self.z_scale, self.x_scale, self.y_scale, Z, X, Y)
        out = x_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return out.view(B, outC, outZ, outX, outY).contiguous()

class PixelUnshuffle3d(nn.Module):
    def __init__(self,z_scale, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.z_scale = z_scale

    def forward(self, x):
        B, C, Z, X, Y = x.size()
        outC = C * self.x_scale * self.y_scale * self.z_scale
        outX = X // self.x_scale
        outY = Y // self.y_scale
        outZ = Z // self.z_scale
        x_view = x.contiguous().view(B, C, self.z_scale, outZ, self.x_scale, outX, self.y_scale, outY)
        out = x_view.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()

        return out.view(B, outC, outZ, outX, outY).contiguous()


def img2windows(img, Z_sp, H_sp, W_sp):
    """
    Input: Image (B, C, Z, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, Z, H, W = img.shape
    img_reshape = img.view(B, C, Z // Z_sp, Z_sp, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().reshape(-1, Z_sp * H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_zhw, Z_sp, H_sp, W_sp, Z, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, Z, H, W, C)
    """
    B = int(img_splits_zhw.shape[0] / (Z * H * W / Z_sp / H_sp / W_sp))

    img = img_splits_zhw.view(B, Z // Z_sp, H // H_sp, W // W_sp, Z_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Z, H, W, -1)
    return img


class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

    def forward(self, x, Z, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        # TODO changed tested
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, Z, H, W)).flatten(2).transpose(-1,
                                                                                                                 -2).contiguous()

        return x1 * x2

class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int, drop: float):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor,Z,H,W) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, Z, H, W,train=False):
        """
        Input: x: (B, Z*H*W, C),Z, H, W
        Output: x: (B, Z*H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, Z, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class Expert_conv(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features,3,1,1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features,3,1,1)
        self.act2 = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x, Z, H, W,train=False):
        """
        Input: x: (B, Z*H*W, C),Z, H, W
        Output: x: (B, Z*H*W, C)
        """
        B, N, C = x.shape
        x = self.fc1(x.transpose(1, 2).contiguous().view(B, C, Z, H, W))
        x = self.act1(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop(x).flatten(2).transpose(-1,-2).contiguous()
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(3, self.pos_dim) #TODO changed to test
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, dim,n_activated_experts,score_func,route_scale,n_routed_experts,normal_moe_weight,
                aux_free_loss,aux_free_rate,aux_free_batch):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = dim
        self.topk = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.n_routed_experts = n_routed_experts
        self.normal_moe_weight = normal_moe_weight

        self.aux_free_loss = aux_free_loss
        self.aux_free_batch = aux_free_batch
        self.counts_p = 0
        self.counts = nn.Parameter(torch.empty(aux_free_batch,n_routed_experts)) if self.aux_free_loss else None
        self.aux_free_rate=aux_free_rate

        self.linear = nn.Linear(dim,n_routed_experts,bias=False)
        self.bias = nn.Parameter(torch.empty(n_routed_experts)) if self.aux_free_loss else None

        if self.aux_free_loss:
            nn.init.constant_(self.bias,0)
            nn.init.constant_(self.counts, 0)

    def forward(self, x: torch.Tensor,train=False):
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        # scores = F.linear(x, self.weight, None)
        # x = x.squeeze(1)
        scores = self.linear(x)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias

        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid" or self.normal_moe_weight:
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale

        ## aux free bias update
        if train and self.aux_free_loss:
            with torch.no_grad():
                self.counts[self.counts_p] = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)#.item()
                self.bias += self.aux_free_rate * torch.sign(torch.mean(self.counts) - torch.mean(self.counts,axis=0))
                self.counts_p = (self.counts_p+1) % self.aux_free_batch

        return weights.type_as(x), indices


class MoE_SGFN(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, dim, reso,moe_inter_dim,n_shared_experts,n_routed_experts,n_activated_experts,normal_moe_weight,
                 score_func,route_scale,act_layer,sgfn_drop,aux_free_loss,aux_free_rate,aux_free_batch,expert_type,gate_feature_dim,gate_f_type):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.gate_f_type= gate_f_type

        if gate_f_type == 'AvgPool':
            self.gate_head = nn.AdaptiveAvgPool3d(1)
            gate_dim = dim
        elif  gate_f_type == 'conv':
            self.gate_head = nn.Sequential(nn.Conv3d(dim,(dim+gate_feature_dim)//2,3,2,1),
                                            nn.Conv3d((dim+gate_feature_dim)//2, gate_feature_dim, 3, 2, 1))
            gate_dim = gate_feature_dim * reso[0] * reso[1] * reso[2] //(2**6)
        else:
            print("warning gate_f_type do not supported")
            self.gate_head = nn.Sequential(nn.Conv3d(dim,(dim+gate_feature_dim)//2,3,2,1),
                                            nn.Conv3d((dim+gate_feature_dim)//2, gate_feature_dim, 3, 2, 1))
            gate_dim = gate_feature_dim * reso[0] * reso[1] * reso[2] //(2**6)
        self.gate = Gate(dim=gate_dim,n_activated_experts=n_activated_experts,score_func=score_func,
                         route_scale=route_scale,n_routed_experts=n_routed_experts,normal_moe_weight=normal_moe_weight
                         ,aux_free_loss=aux_free_loss,aux_free_rate=aux_free_rate,aux_free_batch=aux_free_batch)
        if expert_type == 'SGFN':
            self.experts = nn.ModuleList([SGFN(dim, moe_inter_dim, out_features=dim, act_layer=act_layer,
                                       drop=sgfn_drop)for i in range(self.n_routed_experts)])
        elif expert_type == 'Expert':
            self.experts = nn.ModuleList([Expert(dim, moe_inter_dim,drop=sgfn_drop)
                                          for i in range(self.n_routed_experts)])
        elif expert_type == 'Expert_conv':
            self.experts = nn.ModuleList([Expert_conv(dim, moe_inter_dim, drop=sgfn_drop)
                                          for i in range(self.n_routed_experts)])
        else:
            print("warning expert_type do not supported")
            self.experts = nn.ModuleList([Expert(dim, moe_inter_dim,drop=sgfn_drop)
                                          for i in range(self.n_routed_experts)])

        self.shared_experts = SGFN(dim, n_shared_experts * moe_inter_dim, out_features=dim, act_layer=act_layer,
                                   drop=sgfn_drop)

        if expert_type == 'SGFN':
            self.shared_experts = SGFN(dim, n_shared_experts * moe_inter_dim, out_features=dim, act_layer=act_layer,
                                   drop=sgfn_drop)
        elif expert_type == 'Expert':
            self.shared_experts = Expert(dim, n_shared_experts*moe_inter_dim,drop=sgfn_drop)
        elif expert_type == 'Expert_conv':
            self.shared_experts = Expert_conv(dim, n_shared_experts * moe_inter_dim, drop=sgfn_drop)
        else:
            print("warning expert_type do not supported")
            self.shared_experts = Expert(dim, n_shared_experts*moe_inter_dim,drop=sgfn_drop)

    def forward(self, x, Z, H, W, train=False):
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor. B H C

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        B,N,C = x.size()
        context = (self.gate_head(x.transpose(1, 2).view(B, C, Z, H, W).contiguous())
                   .view(B, -1).contiguous())

        res = self.gate(context, train=train)

        weights, indices = res
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(0, self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx],Z,H,W) * weights[idx, top, None, None]
        y = y.view(B,N,C)
        z = self.shared_experts(x, Z, H, W)


        return (y + z)

class MoE_attn(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, b_idx,dim, reso,num_heads,split_size,shift_size,qkv_bias,
                 qk_scale,drop,attn_drop,rg_idx,bn,
                 n_routed_experts,n_activated_experts,normal_moe_weight,
                 score_func,route_scale,aux_free_loss,aux_free_rate,aux_free_batch,
                 gate_feature_dim,gate_f_type):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.gate_f_type= gate_f_type

        if gate_f_type == 'AvgPool':
            self.gate_head = nn.AdaptiveAvgPool3d(1)
            gate_dim = dim
        elif  gate_f_type == 'conv':
            self.gate_head = nn.Sequential(nn.Conv3d(dim,(dim+gate_feature_dim)//2,3,2,1),
                                            nn.Conv3d((dim+gate_feature_dim)//2, gate_feature_dim, 3, 2, 1))
            gate_dim = gate_feature_dim * reso[0] * reso[1] * reso[2] //(2**6)
        else:
            print("warning gate_f_type do not supported")
            self.gate_head = nn.Sequential(nn.Conv3d(dim,(dim+gate_feature_dim)//2,3,2,1),
                                            nn.Conv3d((dim+gate_feature_dim)//2, gate_feature_dim, 3, 2, 1))
            gate_dim = gate_feature_dim * reso[0] * reso[1] * reso[2] //(2**6)
        self.gate = Gate(dim=gate_dim,n_activated_experts=n_activated_experts,score_func=score_func,
                         route_scale=route_scale,n_routed_experts=n_routed_experts,normal_moe_weight=normal_moe_weight
                         ,aux_free_loss=aux_free_loss,aux_free_rate=aux_free_rate,aux_free_batch=aux_free_batch)
        if b_idx % 2 == 0:
            # DSTB
            self.experts = nn.ModuleList([Adaptive_Spatial_Attention(
                dim, bn=bn,num_heads=num_heads,
            reso=reso, split_size=split_size, shift_size=shift_size, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, rg_idx=rg_idx, b_idx=b_idx
            ) for i in range(self.n_routed_experts)])
            self.shared_experts = Adaptive_Spatial_Attention(
                dim, bn=bn, num_heads=num_heads,
                reso=reso, split_size=split_size, shift_size=shift_size, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, rg_idx=rg_idx, b_idx=b_idx
            )
        else:
            # DCTB
            self.experts = nn.ModuleList([Adaptive_Channel_Attention(
                dim, bn=bn, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop
            ) for i in range(self.n_routed_experts)])
            self.shared_experts =Adaptive_Channel_Attention(
                dim, bn=bn, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop
            )

    def forward(self, x, Z, H, W, train=False):
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor. B H C

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        B,N,C = x.size()
        context = (self.gate_head(x.transpose(1, 2).view(B, C, Z, H, W).contiguous())
                   .view(B, -1).contiguous())

        res = self.gate(context, train=train)

        weights, indices = res
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(0, self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx],Z,H,W) * weights[idx, top, None, None]
        y = y.view(B,N,C)
        z = self.shared_experts(x, Z, H, W)


        return (y + z)

def load_balancing_loss_func2(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2) -> float:
    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        stacked_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    scores = torch.nn.functional.softmax(stacked_gate_logits, dim=-1)  # [num_layers, num_tokens, num_experts]
    _, indices = torch.topk(scores, top_k, dim=-1)  # [num_layers, num_tokens, top_k]
    expert_mask = torch.nn.functional.one_hot(indices,num_experts)  # [num_layers, num_tokens, top_k, num_experts]
    # For a given token, determine if it was routed to a given expert. Think of this as a collection of top_k-hot vectors.
    expert_mask = torch.max(expert_mask, dim=-2).values.float()  # [num_layers, num_tokens, num_experts]
    tokens_per_layer_and_expert = torch.mean(expert_mask, dim=-2)  # [num_layers, num_experts]
    router_prob_per_layer_and_expert = torch.mean(scores, dim=-2)  # [num_layers, num_experts]
    return torch.mean(tokens_per_layer_and_expert * router_prob_per_layer_and_expert) * num_experts ** 2


class Spatial_Attention(nn.Module):
    """ Spatial Window Self-Attention.
    It supports rectangle window (containing square window).
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """

    def __init__(self, dim, idx, split_size=[8, 8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0.,
                 qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            Z_sp, H_sp, W_sp = self.split_size[0], self.split_size[1], self.split_size[2]
        elif idx == 1:
            Z_sp, W_sp, H_sp = self.split_size[0], self.split_size[1], self.split_size[2]
        else:
            print("ERROR MODE", idx)
            exit(0)

        self.Z_sp = Z_sp
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            # TODO changed to test
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_z = torch.arange(1 - self.Z_sp, self.Z_sp)
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_z, position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_z = torch.arange(self.Z_sp)
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_z, coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.Z_sp - 1
            relative_coords[:, :, 1] += self.H_sp - 1
            relative_coords[:, :, 2] += self.W_sp - 1
            relative_coords[:, :, 0] *= (2 * self.H_sp - 1) * (2 * self.W_sp - 1)
            relative_coords[:, :, 1] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, Z, H, W):
        # TODO changed to test
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, Z, H, W)
        x = img2windows(x, self.Z_sp, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.Z_sp * self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, Z, H, W, mask=None):
        """
        Input: qkv: (3, B, L, C), Z, H, W, mask: (B, N, N), N is the window size # TODO fixed to test
        Output: x (B, Z, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == Z * H * W, "flatten img_tokens has wrong size"
        # TODO changed to test
        # partition the q,k,v, image to window
        q = self.im2win(q, Z, H, W)
        k = self.im2win(k, Z, H, W)
        v = self.im2win(v, Z, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.Z_sp * self.H_sp * self.W_sp, self.Z_sp * self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.Z_sp * self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.Z_sp, self.H_sp, self.W_sp, Z, H, W)  # B H' W' C

        return x


class Adaptive_Spatial_Attention(nn.Module):
    # The implementation builds on CAT code https://github.com/Zhengchen1999/CAT
    """ Adaptive Spatial Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        split_size (tuple(int)): Height and Width of spatial window.
        shift_size (tuple(int)): Shift size for spatial window.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Attention dropout rate. Default: 0.0
        rg_idx (int): The indentix of Residual Group (RG)
        b_idx (int): The indentix of Block in each RG
    """

    def __init__(self, dim, num_heads,bn = True,
                 reso=[3, 64, 64], split_size=[8, 8], shift_size=[1, 2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., rg_idx=0, b_idx=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"
        assert 0 <= self.shift_size[2] < self.split_size[2], "shift_size must in 0-split_size2"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            Spatial_Attention(
                dim // 2, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)])

        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
                self.rg_idx % 2 != 0 and self.b_idx % 4 == 0):
            attn_mask = self.calculate_mask(self.patches_resolution[0], self.patches_resolution[1],
                                            self.patches_resolution[2])
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        # TODO changed to test
        if bn:
            self.dwconv = nn.Sequential(
                nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                nn.BatchNorm3d(dim),
                nn.GELU()
            )
            self.channel_interaction = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(dim, dim // 8, kernel_size=1),
                nn.BatchNorm3d(dim // 8),
                nn.GELU(),
                nn.Conv3d(dim // 8, dim, kernel_size=1),
            )
            self.spatial_interaction = nn.Sequential(
                nn.Conv3d(dim, dim // 16, kernel_size=1),
                nn.BatchNorm3d(dim // 16),
                nn.GELU(),
                nn.Conv3d(dim // 16, 1, kernel_size=1)
            )
        else:
            self.dwconv = nn.Sequential(
                nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                nn.GELU()
            )
            self.channel_interaction = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(dim, dim // 8, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(dim // 8, dim, kernel_size=1),
            )
            self.spatial_interaction = nn.Sequential(
                nn.Conv3d(dim, dim // 16, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(dim // 16, 1, kernel_size=1)
            )

    def calculate_mask(self, Z, H, W):  # TODO
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for shift window
        # TODO changed to test
        img_mask_0 = torch.zeros((1, Z, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, Z, H, W, 1))  # 1 H W 1 idx=1

        z_slices_0 = z_slices_1 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))

        w_slices_1 = h_slices_0 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))
        h_slices_1 = w_slices_0 = (slice(0, -self.split_size[2]),
                      slice(-self.split_size[2], -self.shift_size[2]),
                      slice(-self.shift_size[2], None))

        cnt = 0
        for z in z_slices_0:
            for h in h_slices_0:
                for w in w_slices_0:
                    img_mask_0[:, z, h, w, :] = cnt
                    cnt += 1
        cnt = 0
        for z in z_slices_1:
            for h in h_slices_1:
                for w in w_slices_1:
                    img_mask_1[:, z, h, w, :] = cnt
                    cnt += 1

        # calculate mask for window-0
        img_mask_0 = img_mask_0.view(1, Z // self.split_size[0], self.split_size[0]
                                     , H // self.split_size[1], self.split_size[1]
                                     , W // self.split_size[2], self.split_size[2], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()\
            .view(-1, self.split_size[0], self.split_size[1], self.split_size[2], 1)  # nW, sw[0], sw[1], sw[2], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1] * self.split_size[2])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for window-1
        img_mask_1 = img_mask_1.view(1, Z // self.split_size[0], self.split_size[0]
                                     , H // self.split_size[1], self.split_size[1]
                                     , W // self.split_size[2], self.split_size[2], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()\
            .view(-1, self.split_size[0], self.split_size[1], self.split_size[2], 1)  # nW, sw[0], sw[1], sw[2], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[0] * self.split_size[1] * self.split_size[2])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, Z, H, W):
        """
        Input: x: (B, Z*H*W, C), H, W
        Output: x: (B, Z*H*W, C)
        """
        B, L, C = x.shape
        assert L == Z * H * W, "flatten img_tokens has wrong size"

        # TODO changed to test
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, ZHW, C
        # V without partition
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, Z, H, W)

        # image padding
        max_split_size = max(self.split_size[0], self.split_size[1], self.split_size[2])
        pad_zl = pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size
        pad_zh = (self.split_size[0] - Z % self.split_size[0]) % self.split_size[0]

        qkv = qkv.reshape(3 * B, Z, H, W, C).permute(0, 4, 1, 2, 3)  # 3B C Z H W
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b, pad_zl, pad_zh)).reshape(3, B, C, -1).transpose(-2, -1)  # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _Z = pad_zh + Z
        _L = _H * _W * _Z

        # window-0 and window-1 on split channels [C/2, C/2]; for square windows (e.g., 8x8), window-0 and window-1 can be merged
        # shift in block: (0, 4, 8, ...), (2, 6, 10, ...), (0, 4, 8, ...), (2, 6, 10, ...), ...
        # if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (
        #         self.rg_idx % 2 != 0 and self.b_idx % 4 == 0):
        # TODO changed in UDAT3D
        # shift in block: (2, 6, 10, ...) ...
        if self.b_idx > 0 and (self.b_idx - 2) % 4 == 0:
            qkv = qkv.view(3, B, _Z, _H, _W, C)

            qkv_0 = torch.roll(qkv[:, :, :, :, :, :C // 2], shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                               dims=(2, 3, 4))
            qkv_0 = qkv_0.view(3, B, _L, C // 2)
            qkv_1 = torch.roll(qkv[:, :, :, :, :, C // 2:], shifts=(-self.shift_size[0], -self.shift_size[2],-self.shift_size[1]),
                               dims=(2, 3, 4))
            qkv_1 = qkv_1.view(3, B, _L, C // 2)

            # TODO changed to test
            if self.patches_resolution[0] != _Z or self.patches_resolution[1] != _H or self.patches_resolution[2] != _W:
                mask_tmp = self.calculate_mask(_Z, _H, _W)
                x1_shift = self.attns[0](qkv_0, _Z, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, _Z, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _Z, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _Z, _H, _W, mask=self.attn_mask_1)


            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[0], self.shift_size[2], self.shift_size[1]), dims=(1, 2, 3))
            x1 = x1[:, :Z, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :Z, :H, :W, :].reshape(B, L, C // 2)
            # attention output
            attened_x = torch.cat([x1, x2], dim=2)

        else:
            x1 = self.attns[0](qkv[:, :, :, :C // 2], _Z, _H, _W)[:, :Z, :H, :W, :].reshape(B, L, C // 2)
            x2 = self.attns[1](qkv[:, :, :, C // 2:], _Z, _H, _W)[:, :Z, :H, :W, :].reshape(B, L, C // 2)
            # attention output
            attened_x = torch.cat([x1, x2], dim=2)

        # convolution output
        conv_x = self.dwconv(v)

        # TODO changed to test
        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 4, 1).contiguous().view(B, 1, C)
        # S-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, Z, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)

        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 4, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Adaptive_Channel_Attention(nn.Module):
    # The implementation builds on XCiT code https://github.com/facebookresearch/xcit
    """ Adaptive Channel Self-Attention
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set.
        attn_drop (float): Attention dropout rate. Default: 0.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, bn=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # TODO changed tested
        # TODO check if need BatchNorm
        if bn:
            self.dwconv = nn.Sequential(
                nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                nn.BatchNorm3d(dim),
                nn.GELU()
            )
            self.channel_interaction = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(dim, dim // 8, kernel_size=1),
                nn.BatchNorm3d(dim // 8),
                nn.GELU(),
                nn.Conv3d(dim // 8, dim, kernel_size=1),
            )
            self.spatial_interaction = nn.Sequential(
                nn.Conv3d(dim, dim // 16, kernel_size=1),
                nn.BatchNorm3d(dim // 16),
                nn.GELU(),
                nn.Conv3d(dim // 16, 1, kernel_size=1)
            )
        else:
            self.dwconv = nn.Sequential(
                nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                nn.GELU()
            )
            self.channel_interaction = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(dim, dim // 8, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(dim // 8, dim, kernel_size=1),
            )
            self.spatial_interaction = nn.Sequential(
                nn.Conv3d(dim, dim // 16, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(dim // 16, 1, kernel_size=1)
            )


    def forward(self, x, Z, H, W):
        """
        Input: x: (B, Z*H*W, C), x_size: (Z, H, W)
        Output: x: (B, Z*H*W, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # TODO changed tested
        v_ = v.reshape(B, C, N).contiguous().view(B, C, Z, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention output
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        # convolution output
        conv_x = self.dwconv(v_)

        # TODO changed tested
        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, Z, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 4, 1).contiguous().view(B, N, 1)

        # S-I
        attened_x = attened_x * torch.sigmoid(spatial_map)
        # C-I
        conv_x = conv_x * torch.sigmoid(channel_map)
        # TODO changed tested
        conv_x = conv_x.permute(0, 2, 3, 4, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DATB(nn.Module):
    def __init__(self, dim, num_heads,bn=True, reso=[24, 64, 64], split_size=[2, 4],
                 shift_size=[1, 2], expansion_factor=4.,
                 qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., sgfn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, rg_idx=0, b_idx=0,moe_attn_cw=False,moe_attn_sw=False,
                 moe_ffn_cw = False,moe_ffn_sw = False, n_shared_experts=2, n_routed_experts = 64, n_activated_experts=6,
                 normal_moe_weight = False,channel_map="default",
                 score_func="softmax", route_scale=1.,aux_free_loss=False,aux_free_batch=8,aux_free_rate=0.1,
                 expert_type='Expert',gate_feature_dim=1,gate_f_type='conv'
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if b_idx % 2 == 0:
            self.moe_ffn = moe_ffn_sw
        else:
            self.moe_ffn = moe_ffn_cw
        self.moe_attn_sw = moe_attn_sw
        self.moe_attn_cw = moe_attn_cw

        if b_idx % 2 == 0:
            if self.moe_attn_sw:
                self.attn = MoE_attn(b_idx=b_idx,
                                     dim=dim, bn=bn, num_heads=num_heads,
                                     reso=reso, split_size=split_size, shift_size=shift_size, qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop, rg_idx=rg_idx,
                                     n_routed_experts=n_routed_experts,
                                     n_activated_experts=n_activated_experts, normal_moe_weight=normal_moe_weight,
                                     score_func=score_func, route_scale=route_scale, aux_free_loss=aux_free_loss,
                                     aux_free_batch=aux_free_batch, aux_free_rate=aux_free_rate,
                                     gate_feature_dim=gate_feature_dim, gate_f_type=gate_f_type
                                     )
            # DSTB
            else:
                self.attn = Adaptive_Spatial_Attention(
                    dim, bn=bn,num_heads=num_heads,
                reso=reso, split_size=split_size, shift_size=shift_size, qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop, rg_idx=rg_idx, b_idx=b_idx
                )
        else:
            if self.moe_attn_cw:
                self.attn = MoE_attn(b_idx=b_idx,
                                     dim=dim, bn=bn, num_heads=num_heads,
                                     reso=reso, split_size=split_size, shift_size=shift_size, qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop, rg_idx=rg_idx,
                                     n_routed_experts=n_routed_experts,
                                     n_activated_experts=n_activated_experts, normal_moe_weight=normal_moe_weight,
                                     score_func=score_func, route_scale=route_scale, aux_free_loss=aux_free_loss,
                                     aux_free_batch=aux_free_batch, aux_free_rate=aux_free_rate,
                                     gate_feature_dim=gate_feature_dim, gate_f_type=gate_f_type
                                     )
            else:
                # DCTB
                self.attn = Adaptive_Channel_Attention(
                    dim, bn=bn, num_heads=num_heads,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                    proj_drop=drop
                )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        ffn_hidden_dim = int(dim * expansion_factor)
        if self.moe_ffn:
            self.ffn = MoE_SGFN(dim=dim, reso=reso, moe_inter_dim=ffn_hidden_dim // n_shared_experts,
                            n_shared_experts=n_shared_experts, n_routed_experts=n_routed_experts,
                            n_activated_experts=n_activated_experts,normal_moe_weight=normal_moe_weight,
                            score_func=score_func, route_scale=route_scale, act_layer=act_layer,
                            sgfn_drop=sgfn_drop, aux_free_loss=aux_free_loss,
                            aux_free_batch=aux_free_batch,expert_type=expert_type,aux_free_rate=aux_free_rate,
                            gate_feature_dim=gate_feature_dim,gate_f_type=gate_f_type)
        elif expert_type == 'SGFN':
            self.ffn = SGFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer,
                                 drop=sgfn_drop)
        elif expert_type == 'Expert':
            self.ffn = Expert(dim, ffn_hidden_dim,drop=sgfn_drop)
        elif expert_type == 'Expert_conv':
            self.ffn = Expert_conv(dim, ffn_hidden_dim,drop=sgfn_drop)
        else:
            print("warning expert_type do not supported")
            self.ffn = SGFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer,
                            drop=sgfn_drop)

        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size,train=False):
        """
        Input: x: (B, Z*H*W, C), x_size: (Z, H, W)
        Output: x: (B, Z*H*W, C)
        """
        # TODO changed tested
        Z, H, W = x_size
        x = x + self.drop_path(self.attn(self.norm1(x), Z, H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), Z, H, W, train))
        return x

class ResidualGroup(nn.Module):
    """ ResidualGroup
    Args:
        dim (int): Number of input channels.
        reso (int): Input resolution.
        num_heads (int): Number of attention heads.
        split_size (tuple(int)): Height and Width of spatial window.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop (float): Dropout rate. Default: 0
        attn_drop(float): Attention dropout rate. Default: 0
        drop_paths (float | None): Stochastic depth rate.
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        depth (int): Number of dual aggregation Transformer blocks in residual group.
        use_chk (bool): Whether to use checkpointing to save memory.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 dim,
                 bn,
                 reso,
                 num_heads,
                 split_size=[2, 4],
                 expansion_factor=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_paths=None,
                 sgfn_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 depth=2,
                 use_chk=False,
                 resi_connection='1conv',
                 rg_idx=0,moe_attn_cw=False,moe_attn_sw=False,
                 moe_ffn_cw=False,moe_ffn_sw=False, n_shared_experts=2, n_routed_experts=64, n_activated_experts=6,
                 normal_moe_weight=False,
                 score_func="softmax", route_scale=1., aux_free_loss=False,
                 aux_free_batch=8,aux_free_rate=0.1,expert_type='Expert',gate_feature_dim=1,gate_f_type='conv'
                 ):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso
        # self.moe_ffn = moe_ffn
        self.blocks = nn.ModuleList([
            DATB(
                dim=dim,
                bn = bn,
                num_heads=num_heads,
                reso=reso,
                split_size=split_size,
                shift_size=[split_size[0] // 2, split_size[1] // 2, split_size[2] // 2],
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                sgfn_drop=sgfn_drop,
                act_layer=act_layer,
                norm_layer=norm_layer,
                rg_idx=rg_idx,
                b_idx=i,
                moe_attn_cw=moe_attn_cw,
                moe_attn_sw=moe_attn_sw,
                moe_ffn_cw=moe_ffn_cw,
                moe_ffn_sw=moe_ffn_sw,
                n_shared_experts=n_shared_experts,
                n_routed_experts=n_routed_experts,
                n_activated_experts=n_activated_experts,
                normal_moe_weight = normal_moe_weight,
                score_func=score_func,
                route_scale=route_scale,
            aux_free_batch=aux_free_batch,
                aux_free_rate=aux_free_rate,
                expert_type=expert_type,
                gate_feature_dim=gate_feature_dim,
                                        aux_free_loss=aux_free_loss,
                                    gate_f_type=gate_f_type
            ) for i in range(depth)])

        # TODO changed tested
        self.moe_ffn_cw = moe_ffn_cw
        self.moe_ffn_sw = moe_ffn_sw
        if resi_connection == '1conv':
            self.conv = nn.Conv3d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim, 3, 1, 1))
        elif resi_connection == 'SFFB3D':
            self.conv = SFFB3D(dim)
        elif resi_connection == 'SFFB2D':
            self.conv = SFFB2D(dim)
        elif resi_connection == 'SFFB3D_complex_conv':
            self.conv = SFFB3D_complex_conv(dim)

    def forward(self, x, x_size,train=False):
        """
        Input: x: (B, C, Z, H, W), x_size: (Z, H, W)
        Output: x: (B, C, Z, H, W)
        """
        # TODO changed tested
        Z, H, W = x_size
        res = x

        x = rearrange(x, "b c z h w -> b (z h w) c")
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size,train)
            else:
                x = blk(x, x_size,train)
        x = rearrange(x, "b (z h w) c -> b c z h w", z=Z, h=H, w=W)
        x = self.conv(x)

        x = res + x

        return x


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        # TODO changed tested
        z_scale, x_scale, y_scale = scale
        assert x_scale == y_scale
        if (x_scale & (x_scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(x_scale, 2))):
                m.append(nn.Conv3d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(PixelShuffle3d(1, 2, 2))
        elif x_scale == 3:
            m.append(nn.Conv3d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(PixelShuffle3d(1, 3, 3))
        else:
            raise ValueError(f'xscale {x_scale} is not supported. ' 'Supported scales: 2^n and 3.')

        if (z_scale & (z_scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(z_scale, 2))):
                m.append(nn.Conv3d(num_feat, 2 * num_feat, 3, 1, 1))
                m.append(PixelShuffle3d(2, 1, 1))
        elif z_scale == 3:
            m.append(nn.Conv3d(num_feat, 3 * num_feat, 3, 1, 1))
            m.append(PixelShuffle3d(3, 1, 1))
        else:
            raise ValueError(f'z scale {z_scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleLarge(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        # TODO changed tested
        z_scale, x_scale, y_scale = scale
        assert x_scale == y_scale
        if (x_scale & (x_scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(x_scale, 2))):
                m.append(nn.Conv3d(num_feat, num_feat, 3, 1, 1))
                m.append(PixelShuffle3d(1, 2, 2))
                num_feat = num_feat // 4
        elif x_scale == 3:
            m.append(nn.Conv3d(num_feat, num_feat, 3, 1, 1))
            m.append(PixelShuffle3d(1, 3, 3))
            num_feat = num_feat // 9
        else:
            raise ValueError(f'xscale {x_scale} is not supported. ' 'Supported scales: 2^n and 3.')

        if (z_scale & (z_scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(z_scale, 2))):
                m.append(nn.Conv3d(num_feat, num_feat, 3, 1, 1))
                m.append(PixelShuffle3d(2, 1, 1))
                num_feat = num_feat // 2
        elif z_scale == 3:
            m.append(nn.Conv3d(num_feat, num_feat, 3, 1, 1))
            m.append(PixelShuffle3d(3, 1, 1))
            num_feat = num_feat // 3
        else:
            raise ValueError(f'z scale {z_scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(UpsampleLarge, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        # TODO changed tested
        self.num_feat = num_feat
        # self.input_resolution = input_resolution
        z_scale, x_scale, y_scale = scale
        m = []
        m.append(nn.Conv3d(num_feat, x_scale * y_scale * z_scale * num_out_ch, 3, 1, 1))
        m.append(PixelShuffle3d(x_scale, y_scale, z_scale))
        super(UpsampleOneStep, self).__init__(*m)

    # def flops(self):
    #     h, w = self.input_resolution
    #     flops = h * w * self.num_feat * 3 * 9
    #     return flops

class DownsamplePixelShuffle(nn.Module):
    def __init__(self, scale, in_nfeat, out_nfeat):
        super(DownsamplePixelShuffle, self).__init__()
        z_scale, x_scale, y_scale = scale
        self.body = nn.Sequential(nn.Conv3d(in_nfeat, out_nfeat //z_scale//x_scale//y_scale, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelUnshuffle3d(z_scale, x_scale, y_scale))

    def forward(self, x):
        return self.body(x)


class UpsamplePixelShuffle(nn.Module):
    def __init__(self, scale, in_nfeat, out_nfeat):
        super(UpsamplePixelShuffle, self).__init__()
        z_scale, x_scale, y_scale = scale
        self.body = nn.Sequential(nn.Conv3d(in_nfeat, out_nfeat * z_scale * x_scale * y_scale, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelShuffle3d(z_scale, x_scale, y_scale))

    def forward(self, x):
        return self.body(x)

class SRFormer_pMoE(nn.Module):
    """ Dual Aggregation Transformer
    Args:
        img_size (int): Input image size. Default: 64
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 180
        depths (tuple(int)): Depth of each residual group (number of DATB in each RG).
        split_size (tuple(int)): Height and Width of spatial window.
        num_heads (tuple(int)): Number of attention heads in different residual groups.
        expansion_factor (float): Ratio of ffn hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        use_chk (bool): Whether to use checkpointing to save memory.
        upscale: Upscale factor. 2/3/4 for image SR
        img_range: Image range. 1. or 255.
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=[24, 64, 64],
                 in_chans=1,
                 embed_dim=180,
                 split_size=[2, 4],  # TODO
                 depth=[2, 2, 2, 2],
                 num_heads=[2, 2, 2, 2],
                 expansion_factor=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 sgfn_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_chk=False,
                 upscale=2,  # TODO changeto [x y z]
                 img_range=1.,
                 bn=True,
                 resi_connection='1conv',
                 out_proj = '1conv',
                 upsampler='pixelshuffle',
                 bayesian = False,
                 moe_attn_cw=False,moe_attn_sw=False,
                 moe_ffn_cw=False,
                 moe_ffn_sw=False, n_shared_experts=2, n_routed_experts=64, n_activated_experts=6,
                 normal_moe_weight=False,
                 score_func="softmax", route_scale=1.,
                 aux_free_loss=False,
                 aux_free_batch=8,
                aux_free_rate=0.1,
                expert_type='SGFN',
                 gate_feature_dim=1,
                 gate_f_type='conv',
                 **kwargs):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_res = nn.Conv3d(embed_dim, embed_dim *2, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads

        # TODO changed tested
        self.before_RG = nn.Sequential(
            Rearrange('b c z h w -> b (z h w) c'),
            nn.LayerNorm(embed_dim),
        )
        self.moe_ffn = moe_ffn_cw or moe_ffn_sw

        curr_dim = embed_dim*2
        # TODO check dpr
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        img_size_l2 = [img_size[0], img_size[1]//2, img_size[2]//2]
        self.encoder_l1 = ResidualGroup(dim=embed_dim,
                                    bn = bn,
                                    num_heads=heads[0],
                                    reso=img_size,
                                    split_size=split_size[0],
                                    expansion_factor=expansion_factor,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    sgfn_drop=sgfn_drop,
                                    drop_paths=dpr[sum(depth[:0]):sum(depth[:1])],
                                    act_layer=act_layer,
                                    norm_layer=norm_layer,
                                    depth=depth[0],
                                    use_chk=use_chk,
                                    resi_connection=resi_connection,
                                    rg_idx=0,
                                        moe_attn_cw=moe_attn_cw,
                                        moe_attn_sw=moe_attn_sw,
                                    moe_ffn_cw=moe_ffn_cw,
                                    moe_ffn_sw=moe_ffn_sw,
                                    n_shared_experts=n_shared_experts,
                                    n_routed_experts=n_routed_experts,
                                    n_activated_experts=n_activated_experts,
                                        normal_moe_weight=normal_moe_weight,
                                    score_func=score_func,
                                    route_scale=route_scale,
                                        aux_free_batch=aux_free_batch,
                                        aux_free_rate=aux_free_rate,
                                        expert_type=expert_type,
                                        gate_feature_dim=gate_feature_dim,
                                        aux_free_loss=aux_free_loss,
                                        gate_f_type=gate_f_type

                                        )
        self.down_l1 = DownsamplePixelShuffle([1, 2, 2], embed_dim, embed_dim * 2)

        self.latent = ResidualGroup(dim=embed_dim * 2,
                                    bn=bn,
                                      num_heads=heads[1],
                                      reso=img_size_l2,
                                      split_size=split_size[1],
                                      expansion_factor=expansion_factor,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      drop=drop_rate,
                                      attn_drop=attn_drop_rate,
                                    sgfn_drop=sgfn_drop,
                                      drop_paths=dpr[sum(depth[:1]):sum(depth[:2])],
                                      act_layer=act_layer,
                                      norm_layer=norm_layer,
                                      depth=depth[1],
                                      use_chk=use_chk,
                                      resi_connection=resi_connection,
                                      rg_idx=0,
                                    moe_attn_cw=moe_attn_cw,
                                    moe_attn_sw=moe_attn_sw,
                                    moe_ffn_cw=moe_ffn_cw,
                                    moe_ffn_sw=moe_ffn_sw,
                                    n_shared_experts=n_shared_experts,
                                    n_routed_experts=n_routed_experts,
                                    n_activated_experts=n_activated_experts,
                                    normal_moe_weight=normal_moe_weight,
                                    score_func=score_func,
                                    route_scale=route_scale,
                                    aux_free_batch=aux_free_batch,
                                    aux_free_rate=aux_free_rate,
                                    expert_type=expert_type,
                                    gate_feature_dim=gate_feature_dim,
                                        aux_free_loss=aux_free_loss,
                                    gate_f_type=gate_f_type
                                    )

        self.up_l1 = UpsamplePixelShuffle([1, 2, 2], embed_dim * 2, embed_dim)
        self.decoder_l1 = ResidualGroup(dim=embed_dim*2,
                                        bn=bn,
                                      num_heads=heads[2],
                                      reso=img_size,
                                      split_size=split_size[0],
                                      expansion_factor=expansion_factor,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      drop=drop_rate,
                                      attn_drop=attn_drop_rate,
                                        sgfn_drop=sgfn_drop,
                                      drop_paths=dpr[sum(depth[:2]):sum(depth[:3])],
                                      act_layer=act_layer,
                                      norm_layer=norm_layer,
                                      depth=depth[2],
                                      use_chk=use_chk,
                                      resi_connection=resi_connection,
                                      rg_idx=0,
                                        moe_attn_cw=moe_attn_cw,
                                        moe_attn_sw=moe_attn_sw,
                                        moe_ffn_cw=moe_ffn_cw,
                                        moe_ffn_sw=moe_ffn_sw,
                                        n_shared_experts=n_shared_experts,
                                        n_routed_experts=n_routed_experts,
                                        n_activated_experts=n_activated_experts,
                                        normal_moe_weight=normal_moe_weight,
                                        score_func=score_func,
                                        route_scale=route_scale,
                                        aux_free_batch=aux_free_batch,
                                        aux_free_rate=aux_free_rate,
                                        expert_type=expert_type,
                                        gate_feature_dim=gate_feature_dim,
                                        aux_free_loss=aux_free_loss,
                                    gate_f_type=gate_f_type
                                        )

        self.refinement = ResidualGroup(dim=embed_dim*2,
                                        bn=bn,
                                      num_heads=heads[3],
                                      reso=img_size,
                                      split_size=split_size[0],
                                      expansion_factor=expansion_factor,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qk_scale,
                                      drop=drop_rate,
                                      attn_drop=attn_drop_rate,
                                        sgfn_drop=sgfn_drop,
                                      drop_paths=dpr[sum(depth[:3]):sum(depth[:4])],
                                      act_layer=act_layer,
                                      norm_layer=norm_layer,
                                      depth=depth[3],
                                      use_chk=use_chk,
                                      resi_connection=resi_connection,
                                      rg_idx=0,
                                        moe_attn_cw=moe_attn_cw,
                                        moe_attn_sw=moe_attn_sw,
                                        moe_ffn_cw=moe_ffn_cw,
                                        moe_ffn_sw=moe_ffn_sw,
                                        n_shared_experts=n_shared_experts,
                                        n_routed_experts=n_routed_experts,
                                        n_activated_experts=n_activated_experts,
                                        normal_moe_weight=normal_moe_weight,
                                        score_func=score_func,
                                        route_scale=route_scale,
                                        aux_free_batch=aux_free_batch,
                                        aux_free_rate=aux_free_rate,
                                        expert_type=expert_type,
                                        gate_feature_dim=gate_feature_dim,
                                        aux_free_loss=aux_free_loss,
                                    gate_f_type=gate_f_type
                                        )

        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        # TODO changed tested!
        if out_proj == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim*2, embed_dim*2, 3, 1, 1)
        elif out_proj == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv3d(embed_dim*2, embed_dim // 2, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(embed_dim // 2, embed_dim // 2, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(embed_dim // 2, embed_dim * 2 , 3, 1, 1))

        # ------------------------- 3, Reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            # TODO changed conv before upsample
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim *2, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        if self.upsampler == 'pixelshufflelarge':
            # for classical SR
            # TODO changed conv before upsample and upsample
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim*2, embed_dim*2, 3, 1, 1), nn.LeakyReLU(inplace=True))
            z_scale, x_scale, y_scale = upscale
            assert (embed_dim *2 ) % (x_scale * y_scale * z_scale) == 0
            num_feat = (embed_dim *2 ) // (x_scale * y_scale * z_scale)
            self.upsample = UpsampleLarge(upscale, embed_dim*2)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim*2, num_out_ch)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Parameter):
            nn.init.constant_(m.data, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, train=False):
        # TODO changed tested
        _, _, Z, H, W = x.shape
        x_size_l1 = [Z, H, W]
        x_size_l2 = [x_size_l1[0], x_size_l1[1]//2, x_size_l1[2]//2]
        # x_size_l3 = [a // 4 for a in x_size_l1]
        # x_size_l4 = [a // 8 for a in x_size_l1]

        x = self.before_RG(x)
        x = rearrange(x, "b (z h w) c -> b c z h w", z=Z, h=H, w=W)
        # for layer in self.layers:
        #     x = layer(x, x_size)

        x_l1 = self.encoder_l1(x, x_size_l1,train)

        latent = self.down_l1(x_l1)

        latent = self.latent(latent, x_size_l2,train)

        x_up_l1 = self.up_l1(latent)
        x_up_l1 = torch.cat([x_up_l1, x_l1], 1)


        x_up_l1 = self.decoder_l1(x_up_l1, x_size_l1,train)
        x_out = self.refinement(x_up_l1, x_size_l1,train)

        x_out = rearrange(x_out, "b c z h w -> b (z h w) c")
        x_out = self.norm(x_out)
        x_out = rearrange(x_out, "b (z h w) c -> b c z h w", z=Z, h=H, w=W)


        return x_out

    def forward(self, x, train=False):
        """
        Input: x: (B, C, H, W)
        """
        # self.mean = self.mean.type_as(x)
        # x = (x - self.mean) * self.img_range

        # TODO changed tested
        x = x.unsqueeze(1)

        if self.upsampler == 'pixelshuffle' or self.upsampler == 'pixelshufflelarge':
            # for image SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x,train)) + self.conv_res(x)
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x,train)) + self.conv_res(x)
            x = self.upsample(x)


        # TODO changed tested
        x = x.squeeze(1)

        return x

if __name__ == '__main__':
    # import os

    # os.chdir('/data/CHY/PyProjectOnServer')
    torch.cuda.set_device(1)

    depth = 12
    height = 64
    width = 64
    aux_loss_opt= False
    aux_free_loss_opt=True
    model = SRFormer_pMoE(
        upscale=[3, 2, 2],
        in_chans=1,
        img_size=[depth, height, width],
        img_range=1.,
        depth=[8,8,8,8],#[4,4,4,4],#[12,12,12,12],#[4,4,4,4],#[6,6,6,6],
        embed_dim=64,#64,#90,#64,#32,#,90,#64,#90,#64,#96,
        num_heads=[4,4,4,4],#[4,4,4,4],#[6,8,6,6],
        expansion_factor=2,
        bn=False,
        resi_connection='SFFB3D',
        out_proj = '1conv',
        split_size=[[4, 16, 16],[4, 16, 16]],  # [z h w]
        upsampler='pixelshuffle',
        moe_attn_cw=False,
        moe_attn_sw=False,
        moe_ffn=True,
        moe_ffn_cw=True,
        moe_ffn_sw=True,
        n_shared_experts=2, n_routed_experts=24, n_activated_experts=24,
        normal_moe_weight=False,
        score_func="softmax", route_scale=1.,
        aux_free_loss=aux_free_loss_opt,
        aux_free_batch=256,aux_free_rate=0.1,expert_type='Expert_conv',gate_feature_dim=2,
        gate_f_type = 'AvgPool'
    ).cuda().eval()

    print(depth, height, width)

    # x = torch.randn(1, depth, height, width).half().cuda()

    # from contextlib import contextmanager
    # import time
    # @contextmanager
    # def timeblock(label, debug=1):
    #     start = time.perf_counter()
    #     try:
    #         yield
    #     finally:
    #         end = time.perf_counter()
    #         if debug:
    #             print('{} : {}'.format(label, end - start))
    #
    #
    # with timeblock("****************\n time for predicting : "):
    #     y = model(x, train=False)
    #
    # print(y.shape)

    # from torchsummary import summary
    # summary(model,input_size=(depth, height, width))

    from thop import profile
    inputs = torch.randn(1,depth, height, width).cuda()
    mac, params = profile(model,inputs=(inputs,))
    print("mac:%.2f, params:%.2f"%(2*mac,params))