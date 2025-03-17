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
    
    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)
        
    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"     

    def forward(self, x): # B C H W
        return self.spike_function(x, min_value=self.min_value, max_value=self.max_value) / (self.Norm)

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

        x = q @ k.transpose(-2, -1)
        x = (x @ v) * (self.scale*2)

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
                for j in range(2)
            ]
        )
        # self.pos_encode_z = nn.Parameter(torch.zeros(256, 64))
        # self.pos_encode_x = nn.Parameter(torch.zeros(256, 256))


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features_siamase(self, x):
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

    def forward_features_transformer(self, x):          # x[B, 256, 14, 14]

        for blk in self.block3:
            x = blk(x)                  # output[B, 256, 14, 14]

        x = self.downsample4(x)     # output[B, 360, 14, 14]

        for blk in self.block4:
            x = blk(x)

        return x  # T,B,C,N

    def forward(self, z, x):
        z = self.forward_features_siamase(z)    # [B, 256, 8, 8]
        x = self.forward_features_siamase(x)    # [B, 256, 16, 16]

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




def spike_driven_transformer_v3(**kwargs):
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

def Efficient_Spiking_Transformer_m(**kwargs):
    #10.0M
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[48, 96, 192, 240],
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


def Efficient_Spiking_Transformer_s(**kwargs):
    #5.1M
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[32, 64, 128, 192],
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

def Efficient_Spiking_Transformer_t(**kwargs):
    model = Spiking_vit_MetaFormer_Spike_SepConv(
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        embed_dim=[24, 48, 96, 128],
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




from timm.models import create_model
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    model = Efficient_Spiking_Transformer_s()
    print(model)
    x = torch.randn(1,3,224,224)
    y = model(x)
    torchinfo.summary(model, (1, 3, 224, 224),device='cpu')
