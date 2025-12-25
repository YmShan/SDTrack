import torch
import torchinfo
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

import os
import math

def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
        # 这里用 heaviside 函数作为示例，你可以替换成你实际用的前向函数
        return torch.heaviside(x, torch.tensor(1.0, device=x.device))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = 4
        return sigmoid_backward(grad_output, x, alpha)

class LIF(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, tau=2.0):
        super(LIF, self).__init__()

        self.threshold = v_threshold
        self.reset = v_reset
        self.tau = tau
        self.surrogate_function = sigmoid
        self.v = None
        # self.fr = []
    # 设置v的device
    def init_v(self, x):

        self.v = torch.zeros_like(x[0])  # 默认初始化
        
        self.t = x.shape[0]  # [T, B, C, H, W]

    # 泄露
    def to_leakage(self, i):
        """
        :param i: 脉冲前神经元输入
        :return: 电压
        """
        self.v = self.v + (i - self.v)/ self.tau

    def to_firing(self):
        """
        :param u: 电压
        :return: spike
        """
        spike_t = self.surrogate_function.apply(self.v - self.threshold)
        return spike_t

    def to_reset(self, spike):
        """
        :param u: 电压
        :param s: spike
        :return: 当前膜电位
        """
        spike_d = spike.detach()                  # hard_reset
        self.v = (1.0 - spike_d) * self.v
        # self.v.data = self.v - self.threshold * spike_d

    def to_zero(self):

        self.v = None

    def forward(self, x):
        """
        :param x: 输入
        :return: spike
        """
        if self.v == None:
            self.init_v(x)

        spike = []
        for t in range(self.t):
            self.to_leakage(x[t])
            spike_t = self.to_firing()
            self.to_reset(spike_t)
            spike.append(spike_t)

        # self.fr.append(torch.stack(spike).mean().item())
        return torch.stack(spike)




class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = LIF()
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.spike2 = LIF()
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        
        x = self.spike1(x)
            
        x = self.bn1(self.pwconv1(x))
        
        x = self.spike2(x)
            
        x = self.dwconv(x)
        x = self.bn2(self.pwconv2(x))
        return x

class SepConv_Spike(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.spike1 = LIF()

        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)

        self.spike2 = LIF()

        self.dwconv = nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels, bias=bias)
        self.bn2 = nn.BatchNorm2d(med_channels)

        self.spike3 = LIF()

        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(dim)


    def forward(self, x):
        
        x = self.spike1(x)
        
        T, B, C, H, W = x.shape

        x = self.pwconv1(x.flatten(0, 1))
        TB, C, H, W = x.shape
        x = self.bn1(x).reshape(T, B, C, H, W).contiguous()
        
        x = self.spike2(x)
        
        x = self.dwconv(x.flatten(0, 1))
        TB, C, H, W = x.shape
        x = self.bn2(x).reshape(T, B, C, H, W).contiguous()

        x = self.spike3(x)

        x = self.pwconv2(x.flatten(0, 1))
        TB, C, H, W = x.shape
        x = self.bn3(x).reshape(T, B, C, H, W).contiguous()
        return x



class MS_ConvBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = LIF()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = LIF()
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.spike1(x)
        x = self.bn1(self.conv1(x)).reshape(B, self.mlp_ratio * C, H, W)
        x = self.spike2(x)
        x = self.bn2(self.conv2(x)).reshape(B, C, H, W)
        x = x_feat + x

        return x

class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv_Spike(dim=dim)

        self.mlp_ratio = mlp_ratio

        self.spike1 = LIF()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio) 
        self.spike2 = LIF()
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)  

    def forward(self, x):

        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        T, B, C, H, W = x.shape
        x_feat = x.reshape(T, B, C, H, W).contiguous()
        x = self.spike1(x)
        x = self.bn1(self.conv1(x.flatten(0, 1))).reshape(T, B, self.mlp_ratio * C, H, W).contiguous()
        x = self.spike2(x)
        x = self.conv2(x.flatten(0, 1))
        TB, C, H, W = x.shape
        x = self.bn2(x).reshape(T, B, C, H, W).contiguous()
        x = x_feat + x

        return x



class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = LIF()

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = LIF()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_spike(x)
        T, B, C, N = x.shape
        x = self.fc1_conv(x.flatten(0, 1))
        TB, C, N = x.shape
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)
        T, B, C, N = x.shape
        x = self.fc2_conv(x.flatten(0, 1))
        TB, C, N = x.shape
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x



class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim//num_heads) ** -0.5

        self.head_spike = LIF()

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = LIF()
        
        self.k_spike = LIF()
        
        self.v_spike = LIF()

        self.attn_spike = LIF()

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

        # self.proj_conv = nn.Sequential(
        #     nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        # )


    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        x = self.head_spike(x)

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        q = self.q_spike(q)
        q = q.flatten(2)
        q = (
            q.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        k = self.k_spike(k)            
        k = k.flatten(2)
        k = (
            k.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v = self.v_spike(v)
        v = v.flatten(2)
        v = (
            v.transpose(-1, -2)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(2, 3).reshape(B, C, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(B, C, H, W)
        x = self.proj_conv(x).reshape(B, C, H, W)

        return x

class MS_Attention_linear(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        lamda_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim//num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio

        self.head_spike = LIF()

        self.q_conv = nn.Conv2d(dim, dim, 1, 1, bias=False)
        self.q_BN = nn.BatchNorm2d(dim)

        self.q_spike = LIF()

        self.k_conv = nn.Conv2d(dim, dim, 1, 1, bias=False)
        self.k_BN = nn.BatchNorm2d(dim)

        self.k_spike = LIF()

        self.v_conv = nn.Conv2d(dim, int(dim*lamda_ratio), 1, 1, bias=False)
        self.v_BN = nn.BatchNorm2d(int(dim*lamda_ratio))
        
        self.v_spike = LIF()

        self.attn_spike = LIF()


        self.proj_conv = nn.Conv2d(dim*lamda_ratio, dim, 1, 1, bias=False)
        self.proj_BN = nn.BatchNorm2d(dim)
        self.qk = []
        self.qkv = []


    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        C_v = int(C*self.lamda_ratio)

        x = self.head_spike(x)

        T, B, C, H, W = x.shape
        q = self.q_conv(x.flatten(0, 1))
        k = self.k_conv(x.flatten(0, 1))
        v = self.v_conv(x.flatten(0, 1))
        qTB, qC, qH, qW = q.shape
        kTB, kC, kH, kW = k.shape
        vTB, vC, vH, vW = v.shape

        q = self.q_BN(q).reshape(T, B, qC, qH, qW).contiguous()
        k = self.k_BN(k).reshape(T, B, kC, kH, kW).contiguous()
        v = self.v_BN(v).reshape(T, B, vC, vH, vW).contiguous()

        q = self.q_spike(q)
        q = q.flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
      
        k = self.k_spike(k)
        k = k.flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_spike(v)
        v = v.flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C_v // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = q @ k.transpose(-2, -1)
        # x = (x @ v) * (self.scale*2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # qk_cache的firing rate

        total_sum = torch.sum(x)
        self.qk.append(float(total_sum))

        x = (x @ v)

        total_sum = torch.sum(x)
        self.qkv.append(float(total_sum))
        x = x * (self.scale*2)        # [2, 8, 324, 128]

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        x = x.transpose(3, 4).reshape(T, B, C_v, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(T, B, C_v, H, W)


        x = self.proj_conv(x.flatten(0, 1))
        TB, C, H, W = x.shape
        x = self.proj_BN(x).reshape(T, B, C, H, W)

        return x




class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x

class MS_Block_Spike_SepConv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        init_values = 1e-6
    ):
        super().__init__()

        # self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)

        self.attn = MS_Attention_linear(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            lamda_ratio=4,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        
    def forward(self, x):
        # x = x + self.conv(x)
        x = x + self.attn(x) 
        x = x + self.mlp(x) 

        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        T=None,
        two_input=False
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.two_input = two_input
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = LIF()
            self.encode_spike2 = LIF()
            self.encode_spike3 = LIF()
    def forward(self, x):
        if hasattr(self, "encode_spike"):
            if self.two_input == False:
                x = self.encode_spike(x)
            elif x.shape[-1] == 16:
                x = self.encode_spike2(x)
            elif x.shape[-1] == 2:
                x = self.encode_spike3(x)
            else:
                x = self.encode_spike(x)

        T, B, C, H, W = x.shape
        x = self.encode_conv(x.flatten(0, 1))
        TB, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, C, H, W).contiguous()

        return x



class Spiking_vit_MetaFormer_Spike_SepConv(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dim=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    
                )
                for j in range(int(depths * 0.75))
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            two_input=True
        )

        self.block4 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    
                )
                for j in range(int(depths * 0.25))
            ]
        )
        
        self.head = (
            nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        )
        self.spike = LIF()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def create_graph(self, x, z, scale):
        if scale == 'center':
            B, C, XH, _ = x.shape
            _, _, ZH, _ = z.shape
            start_h = (XH - ZH) // 2  # 中央区域起始行
            end_h = start_h + ZH  # 中央区域结束行
            x[:, :, start_h:end_h, start_h:end_h] += z
            return x
        else:
            T, B, C, XH, _ = x.shape
            T, _, _, ZH, _ = z.shape
            out = torch.zeros((T, B, C, XH+ZH-scale, XH+ZH-scale), device='cuda')
            out[:, :, :, :XH, :XH] += x
            start = XH-scale
            out[:, :, :, start:, start:] += z
            return out

    def split_graph(self, x, x_size, z_size, scale):
        if scale == 'center':
            start_h = (x_size - z_size) // 2  # 中央区域起始行
            end_h = start_h + z_size  # 中央区域结束行
            z = x[:, :, :, start_h:end_h, start_h:end_h]
            return x, z 
        else:
            return x[:, :, :, :x_size, :x_size], x[:, :, :, -z_size:, -z_size:]


    def forward_features(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        return x

    def forward_features_transformer(self, x):          # x[B, 256, 18, 18]

        for blk in self.block3:
            x = blk(x)                  # output[B, 256, 18, 18]

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # 在这里应该拆分，先把最后四个token提取出来，再将template和search从对应位置还原，分别过downsample再拼接
        T, B, C, H, W = x.shape
        x_all = x.reshape(T, B, C, H*W)            # [1, 256, 324]

        z = x_all[:, :, :, :64]            # [1, 256, 64]
        _, _, _, N = z.shape
        z = z.reshape(T, B, C, math.isqrt(N), math.isqrt(N))

        x = x_all[:, :, :, 64:320]         # [1, 256, 256]
        _, _, _, N = x.shape
        x = x.reshape(T, B, C, math.isqrt(N), math.isqrt(N))

        appendix_token = x_all[:, :, :, 320:]          # [1, 256, 4]  
        _, _, _, N = appendix_token.shape
        appendix_token = appendix_token.reshape(T, B, C, math.isqrt(N), math.isqrt(N))

        x_z = self.downsample4(z)                       # input[4, 9, 256, 8, 8]
        x_x = self.downsample4(x)                       # input[4, 9, 256, 16, 16]
        x_appendix_token = self.downsample4(appendix_token)     # [1, 360, 2, 2]

        T, B, C, H, W = x_z.shape
        z = x_z.reshape(T, B, C, H*W)                      # [1, 360, 64]
        _, _, _, H, W = x_x.shape      
        x = x_x.reshape(T, B, C, H*W)                      # [1, 360, 256]
        _, _, _, H, W = x_appendix_token.shape
        appendix_token = x_appendix_token.reshape(T, B, C, H*W)        # [1, 360, 4]

        x = torch.cat((z, x, appendix_token), dim=3)            # [B, 360, 324]

        T, B, C, N = x.shape

        x = x.reshape(T, B, C, math.isqrt(N), math.isqrt(N))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        for blk in self.block4:
            x = blk(x)

        return x  # T,B,C,N

    def forward(self, z, x):
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print(z.shape)
        # print(x.shape)
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # x [B, 3, 256, 256]            z [B, 3, 128, 128]
        # print(x.shape)
        scale = 0           # 相交的大小        [0, 32, 64, 96, 128, 'center']
        x = self.create_graph(x, z, scale)

        # x = x + self.pos_encode             # [B, 3, 384, 384]

        x = self.forward_features(x)    # [B, 256, 16, 16]

        x, z = self.split_graph(x, 16, 8, scale)            # [B, 256, 16, 16]    [B, 256, 8, 8]

        z = z.flatten(start_dim=3)              # [B, 256, 64]
        x = x.flatten(start_dim=3)              # [B, 256, 256]

        # z = z.flatten(start_dim=2) + self.pos_encode_z              # [B, 256, 64]
        # x = x.flatten(start_dim=2) + self.pos_encode_x              # [B, 256, 256]
        
        x = torch.cat((z, x), dim=3)            # [B, 256, 320]
        T, B, C, HW = x.shape
        x = torch.cat([x, torch.zeros(T, B, C, 4, device='cuda')], dim=3)
        T, B, C, HW = x.shape
        x = x.reshape(T, B, C, math.isqrt(HW), math.isqrt(HW))

        x = self.forward_features_transformer(x)                # [B, 360, 18, 18]

        T, B, C, H, W = x.shape
        x = x.reshape(T, B, C, H * W)
        x = x[:, :, :, :320]
        aux_dict = {"attn": None}

        return x, aux_dict                      # [B, 360, 320]




def SDTrack_tiny_T4D1(**kwargs):
    #19.0M
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=1000,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=8,
        sr_ratios=1,
        **kwargs,
    )
    return model

