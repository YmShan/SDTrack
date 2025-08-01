import torch
import torchinfo
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
import math
import os

# class Quant(torch.autograd.Function):
#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def forward(ctx, i, min_value, max_value):
#         ctx.min = min_value
#         ctx.max = max_value
#         ctx.save_for_backward(i)
#         return torch.round(torch.clamp(i, min=min_value, max=max_value))

#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         i, = ctx.saved_tensors
#         grad_input[i < ctx.min] = 0
#         grad_input[i > ctx.max] = 0
#         return grad_input, None, None

# class MultiSpike(nn.Module):
#     def __init__(
#         self,
#         min_value=0,
#         max_value=4,
#         Norm=None,
#         ):
#         super().__init__()
#         if Norm == None:
#             self.Norm = max_value
#         else:
#             self.Norm = Norm
#         self.min_value = min_value
#         self.max_value = max_value
    
#     @staticmethod
#     def spike_function(x, min_value, max_value):
#         return Quant.apply(x, min_value, max_value)
        
#     def __repr__(self):
#         return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"     

#     def forward(self, x): # B C H W
#         return self.spike_function(x, min_value=self.min_value, max_value=self.max_value) / (self.Norm)
    
# 计算firing_rate的神经元
class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None
    
class MultiSpike(nn.Module):
    def __init__(
        self,
        min_value=0,
        max_value=4,
        Norm=None,
        ):
        super().__init__()
        if Norm == None:
            self.Norm = max_value
        else:
            self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value
        self.fr = []

    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)
        
    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"     

    def forward(self, x): # B C H W
        
        spike = self.spike_function(x, min_value=self.min_value, max_value=self.max_value)
        count_025 = torch.sum((spike == 1 / self.Norm).to(torch.int)).item()
        count_050 = torch.sum((spike == 2 / self.Norm).to(torch.int)).item()
        count_075 = torch.sum((spike == 3 / self.Norm).to(torch.int)).item()
        count_1 = torch.sum((spike == 4 / self.Norm).to(torch.int)).item()

        self.fr.append((count_025 * 1 + count_050 * 2 + count_075 * 3 + count_1 * 4) / (spike.numel() * self.Norm))

        return spike / (self.Norm)

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
        self.spike1 = MultiSpike()
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.spike2 = MultiSpike()
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
        self.spike1 = MultiSpike()
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(med_channels)
            )
        self.spike2 = MultiSpike()
        self.dwconv = nn.Sequential(
            nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size, padding=padding, groups=med_channels, bias=bias),
            nn.BatchNorm2d(med_channels)
        )
        self.spike3 = MultiSpike()
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        
        x = self.spike1(x)
            
        x = self.pwconv1(x)
        
        x = self.spike2(x)
            
        x = self.dwconv(x)

        x = self.spike3(x)

        x = self.pwconv2(x)
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

        self.spike1 = MultiSpike()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = MultiSpike()
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

        self.spike1 = MultiSpike()
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.spike2 = MultiSpike()
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



class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = MultiSpike()

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = MultiSpike()

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2)
        x = self.fc1_spike(x)
        x = self.fc1_conv(x)
        x = self.fc1_bn(x).reshape(B, self.c_hidden, N).contiguous()
        x = self.fc2_spike(x)
        x = self.fc2_conv(x)
        x = self.fc2_bn(x).reshape(B, C, H, W).contiguous()

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

        self.head_spike = MultiSpike()

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = MultiSpike()
        
        self.k_spike = MultiSpike()
        
        self.v_spike = MultiSpike()

        self.attn_spike = MultiSpike()

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

        self.head_spike = MultiSpike()

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = MultiSpike()

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = MultiSpike()

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim*lamda_ratio), 1, 1, bias=False), nn.BatchNorm2d(int(dim*lamda_ratio)))
        
        self.v_spike = MultiSpike()

        self.attn_spike = MultiSpike()

        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim*lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )
        self.qk = []
        self.qkv = []

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        C_v = int(C*self.lamda_ratio)

        x = self.head_spike(x)          # [1, 256, 18, 18]
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
            .reshape(B, N, self.num_heads, C_v // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # q,k shape [2, 8, 324, 45]    [2, 8, 324, 32]
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        x = q @ k.transpose(-2, -1)
        # x = (x @ v) * (self.scale*2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # qk_cache的firing rate

        total_sum = torch.sum(x) * 4
        self.qk.append(float(total_sum))

        x = (x @ v)

        total_sum = torch.sum(x) * 4
        self.qkv.append(float(total_sum))
        x = x * (self.scale*2)        # [2, 8, 324, 128]

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        x = x.transpose(2, 3).reshape(B, C_v, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(B, C_v, H, W)
        x = self.proj_conv(x).reshape(B, C, H, W)

        return x


class Cross_MS_Attention_linear(nn.Module):
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

        self.head_spike = MultiSpike()

        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.q_spike = MultiSpike()

        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

        self.k_spike = MultiSpike()

        self.v_conv = nn.Sequential(nn.Conv2d(dim, int(dim*lamda_ratio), 1, 1, bias=False), nn.BatchNorm2d(int(dim*lamda_ratio)))
        
        self.v_spike = MultiSpike()

        self.attn_spike = MultiSpike()


        self.proj_conv = nn.Sequential(
            nn.Conv2d(dim*lamda_ratio, dim, 1, 1, bias=False), nn.BatchNorm2d(dim)
        )


    def forward(self, x):
        B, C, H, W = x.shape            # [1, 256, 18, 18]
        N = H * W
        C_v = int(C*self.lamda_ratio)

        x = x.view(B, C, H * W)      # [B, 256, 324]
        x = x[:, :, :320]           # [B, 256, 320]

        template = x[:, :, :64]     # [1, 256, 64]
        search = x[:, :, 64:320]    # [1, 256, 256] 

        template = template.view(B, C, 8, 8)
        search = search.view(B, C, 16, 16)

        template = self.head_spike(template)          # [1, 256, 8, 8]
        search = self.head_spike(search)              # [1, 256, 16, 16] 

        N_template = 64
        N_search = 256

        q_template = self.q_conv(template)
        k_template = self.k_conv(template)
        v_template = self.v_conv(template)

        q_template = self.q_spike(q_template)
        q_template = q_template.flatten(2)          # [1, 256, 64]
        q_template = (                                                   
            q_template.transpose(-1, -2)
            .reshape(B, N_template, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k_template = self.k_spike(k_template)
        k_template = k_template.flatten(2)
        k_template = (
            k_template.transpose(-1, -2)
            .reshape(B, N_template, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v_template = self.v_spike(v_template)
        v_template = v_template.flatten(2)
        v_template = (
            v_template.transpose(-1, -2)
            .reshape(B, N_template, self.num_heads, C_v // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )


        q_search = self.q_conv(search)
        k_search = self.k_conv(search)
        v_search = self.v_conv(search)

        q_search = self.q_spike(q_search)
        q_search = q_search.flatten(2)
        q_search = (                                                   
            q_search.transpose(-1, -2)
            .reshape(B, N_search, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k_search = self.k_spike(k_search)
        k_search = k_search.flatten(2)
        k_search = (
            k_search.transpose(-1, -2)
            .reshape(B, N_search, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v_search = self.v_spike(v_search)
        v_search = v_search.flatten(2)
        v_search = (
            v_search.transpose(-1, -2)
            .reshape(B, N_search, self.num_heads, C_v // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print(q_template.shape)         # [2, 8, 64, 32]
        # print(k_template.shape)         # [2, 8, 64, 32]
        # print(v_template.shape)         # [2, 8, 64, 128]
        # print(q_search.shape)         # [2, 8, 256, 32]
        # print(k_search.shape)         # [2, 8, 256, 32]
        # print(v_search.shape)         # [2, 8, 256, 32]
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


        search_qk_cache = q_template @ k_search.transpose(-2, -1)
        search = (search_qk_cache @ v_search) * (self.scale*2)

        template_qk_cache = q_search @ k_template.transpose(-2, -1)
        template = (template_qk_cache @ v_template) * (self.scale*2)

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print(template.shape)           # [2, 8, 256, 128]
        # print(search.shape)             # [2, 8, 64, 128]
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxx')
        
        x = torch.cat((template, search), dim=2)
        _, _, _, C_padding = x.shape
        padding = torch.zeros(B, self.num_heads, 4, C_padding, device='cuda')
        x = torch.cat((x, padding), dim=2)

        x = x.transpose(2, 3).reshape(B, C_v, N).contiguous()
        x = self.attn_spike(x)
        x = x.reshape(B, C_v, H, W)
        x = self.proj_conv(x).reshape(B, C, H, W)

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
        init_values = 1e-6,
        cross=False
    ):
        super().__init__()

        # self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)
        if cross == False:
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
        else:
            self.attn = Cross_MS_Attention_linear(
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

        x = x + self.attn(x)        # input:[2, 256, 18, 18]     [2, 360, 18, 18]
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
    ):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = MultiSpike()

    def forward(self, x):

        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        x = self.encode_conv(x)
        x = self.encode_bn(x)

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

        # self.block3 = nn.ModuleList(
        #     [
        #         MS_Block_Spike_SepConv(
        #             dim=embed_dim[2],
        #             num_heads=num_heads,
        #             mlp_ratio=mlp_ratios,
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,
        #             drop=drop_rate,
        #             attn_drop=attn_drop_rate,
        #             drop_path=dpr[j],
        #             norm_layer=norm_layer,
        #             sr_ratio=sr_ratios,
        #             cross=False                     # 只改了最后一个参数
        #         )if j % 2 == 0 else MS_Block_Spike_SepConv(
        #             dim=embed_dim[2],
        #             num_heads=num_heads,
        #             mlp_ratio=mlp_ratios,
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,
        #             drop=drop_rate,
        #             attn_drop=attn_drop_rate,
        #             drop_path=dpr[j],
        #             norm_layer=norm_layer,
        #             sr_ratio=sr_ratios,
        #             cross=True                      # 只改了最后一个参数
        #         )
        #         for j in range(6)
        #     ]
        # )

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
                cross=False
            )
            for j in range(6)
        ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
        )

        # self.block4 = nn.ModuleList(
        #     [
        #         MS_Block_Spike_SepConv(
        #             dim=embed_dim[3],
        #             num_heads=num_heads,
        #             mlp_ratio=mlp_ratios,
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,
        #             drop=drop_rate,
        #             attn_drop=attn_drop_rate,
        #             drop_path=dpr[j],
        #             norm_layer=norm_layer,
        #             sr_ratio=sr_ratios,
        #             cross=False
        #         )if j % 2 == 0 else MS_Block_Spike_SepConv(
        #             dim=embed_dim[3],
        #             num_heads=num_heads,
        #             mlp_ratio=mlp_ratios,
        #             qkv_bias=qkv_bias,
        #             qk_scale=qk_scale,
        #             drop=drop_rate,
        #             attn_drop=attn_drop_rate,
        #             drop_path=dpr[j],
        #             norm_layer=norm_layer,
        #             sr_ratio=sr_ratios,
        #             cross=True
        #         )
        #         for j in range(2)
        #     ]
        # )

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
                    cross=False
                )
                for j in range(2)
            ]
        )


        # **************************************************************************
        # split之后加
        # # 可学习的位置编码
        # self.pos_encode_z = nn.Parameter(torch.zeros(1, 256, 64))
        # self.pos_encode_x = nn.Parameter(torch.zeros(1, 256, 256))
        # trunc_normal_(self.pos_encode_z, std=.02)
        # trunc_normal_(self.pos_encode_x, std=.02)
        # 不可学习的余弦位置编码
        # self.pos_encode_z = self.create_positional_encoding(256, 64).cuda()
        # self.pos_encode_x = self.create_positional_encoding(256, 256).cuda()
        # **************************************************************************
        # 合并成一张图之后加
        # 可学习的
        # self.pos_encode = nn.Parameter(torch.zeros(1, 3, 384, 384))
        # trunc_normal_(self.pos_encode, std=.02)
        # 不可学习的
        # self.pos_encode = generate_2D_position_encoding(1, 3, 384, 384).cuda()

        self.apply(self._init_weights)

    def generate_2D_position_encoding(batch_size, channels, height, width):
        # 创建位置编码的网格
        y_position = torch.arange(height).float()  # 高度的索引
        x_position = torch.arange(width).float()   # 宽度的索引
        
        # 生成一个网格，形状为[height, width]
        y_position, x_position = torch.meshgrid(y_position, x_position, indexing='ij')
        
        # 位置编码函数（可以通过sin和cos来生成编码）
        y_position = y_position.unsqueeze(0).unsqueeze(0)  # 变为[1, 1, height, width]
        x_position = x_position.unsqueeze(0).unsqueeze(0)  # 变为[1, 1, height, width]
        
        # 位置编码的周期
        freq = torch.pow(10000, torch.arange(0, channels // 2, 2).float() / channels)
        
        # 扩展频率为[1, channels//2, 1, 1]，以便广播
        freq = freq.view(1, -1, 1, 1)  # 调整频率的形状

        # 生成正弦和余弦位置编码
        sin_y = torch.sin(y_position / freq)
        cos_y = torch.cos(y_position / freq)
        sin_x = torch.sin(x_position / freq)
        cos_x = torch.cos(x_position / freq)

        # 选择其中一部分生成位置编码，以保持3个通道
        # 这里我们只选取 `sin_y`、`cos_y` 和 `sin_x`，以保持通道数为3
        encoding = torch.cat([
            sin_y,
            cos_y,
            sin_x
        ], dim=1)  # 合并为一个编码张量
        

        return encoding


    def create_positional_encoding(self, num_patches, embedding_dim):
        # 创建位置编码
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)  # [num_patches, 1]
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))  # [embedding_dim/2]

        # 计算正弦和余弦编码
        pe = torch.zeros(num_patches, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度
        return pe.unsqueeze(0)  # 形状变为 [1, num_patches, embedding_dim]
    

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
            B, C, XH, _ = x.shape
            _, _, ZH, _ = z.shape
            out = torch.zeros((B, C, XH+ZH-scale, XH+ZH-scale), device='cuda')
            out[:, :, :XH, :XH] += x
            start = XH-scale
            out[:, :, start:, start:] += z
            return out

    def split_graph(self, x, x_size, z_size, scale):
        if scale == 'center':
            start_h = (x_size - z_size) // 2  # 中央区域起始行
            end_h = start_h + z_size  # 中央区域结束行
            z = x[:, :, start_h:end_h, start_h:end_h]
            return x, z 
        else:
            return x[:, :, :x_size, :x_size], x[:, :, -z_size:, -z_size:]


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
        B, C, H, W = x.shape
        x_all = x.reshape(B, C, H*W)            # [1, 256, 324]

        z = x_all[:, :, :64]            # [1, 256, 64]
        _, _, N = z.shape
        z = z.reshape(B, C, math.isqrt(N), math.isqrt(N))

        x = x_all[:, :, 64:320]         # [1, 256, 256]
        _, _, N = x.shape
        x = x.reshape(B, C, math.isqrt(N), math.isqrt(N))

        appendix_token = x_all[:, :, 320:]          # [1, 256, 4]  
        _, _, N = appendix_token.shape
        appendix_token = appendix_token.reshape(B, C, math.isqrt(N), math.isqrt(N))

        x_z = self.downsample4(z)                       # [1, 360, 8, 8]
        x_x = self.downsample4(x)                       # [1, 360, 16, 16]
        x_appendix_token = self.downsample4(appendix_token)     # [1, 360, 2, 2]

        B, C, H, W = x_z.shape
        z = x_z.reshape(B, C, H*W)                      # [1, 360, 64]
        _, _, H, W = x_x.shape      
        x = x_x.reshape(B, C, H*W)                      # [1, 360, 256]
        _, _, H, W = x_appendix_token.shape
        appendix_token = x_appendix_token.reshape(B, C, H*W)        # [1, 360, 4]

        x = torch.cat((z, x, appendix_token), dim=2)            # [B, 360, 324]

        B, C, N = x.shape

        x = x.reshape(B, C, math.isqrt(N), math.isqrt(N))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        for blk in self.block4:
            x = blk(x)

        return x  # T,B,C,N

    def forward(self, z, x):
        # x [B, 3, 256, 256]            z [B, 3, 128, 128]
        
        scale = 0           # 相交的大小        [0, 32, 64, 96, 128, 'center']
        x = self.create_graph(x, z, scale)

        # x = x + self.pos_encode             # [B, 3, 384, 384]

        x = self.forward_features(x)    # [B, 256, 16, 16]

        x, z = self.split_graph(x, 16, 8, scale)            # [B, 256, 16, 16]    [B, 256, 8, 8]

        z = z.flatten(start_dim=2)              # [B, 256, 64]
        x = x.flatten(start_dim=2)              # [B, 256, 256]

        # z = z.flatten(start_dim=2) + self.pos_encode_z              # [B, 256, 64]
        # x = x.flatten(start_dim=2) + self.pos_encode_x              # [B, 256, 256]
        
        x = torch.cat((z, x), dim=2)            # [B, 256, 320]
        B, C, HW = x.shape
        x = torch.cat([x, torch.zeros(B, C, 4, device='cuda')], dim=2)
        B, C, HW = x.shape
        x = x.reshape(B, C, math.isqrt(HW), math.isqrt(HW))

        x = self.forward_features_transformer(x)                # [B, 360, 18, 18]

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = x[:, :, :320]
        aux_dict = {"attn": None}

        return x, aux_dict                      # [B, 360, 320]




def SDTrack_tiny(**kwargs):
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

