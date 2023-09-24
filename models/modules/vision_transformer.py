# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
add mask for vision transformer class
"""
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pruner import GateLayer

class Mul(nn.Module):
    '''add operation for pytorch-flops-counter 
    '''
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        return q, k

def mul_flops_counter_hook(module, input, output):
    '''add flops counter for pytorch-flops-counter 
    '''
    q, k = input
    batch,heads,tgt_len,head_dim = q.shape
    # multiply batch because batch is always 1
    # and swin reshape number of windows into batch
    mul_flops = 2 * heads * tgt_len * head_dim * tgt_len * batch 
    module.__flops__ += int(mul_flops)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.hidden_gate1 = GateLayer(hidden_features, hidden_features, [1, 1, -1])
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.hidden_gate1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim 
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gate =  GateLayer(num_heads, num_heads, [1, -1, 1, 1])
        self.mul_flops = Mul()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        self.mul_flops(q, k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x= attn @ v
        x = self.attn_gate(x)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, tgt_len, remained_fc_out):
        flops = 0
        # qkv project
        remained_head = sum(self.attn_gate.weight) / self.attn_gate.weight.size(0)
        flops += tgt_len * self.dim * remained_fc_out * self.dim * 3 * remained_head
        # qkv attention
        flops += 2 * self.num_heads * tgt_len * self.head_dim *tgt_len * remained_head
        # output project
        flops += tgt_len * self.dim * remained_head * self.dim * remained_fc_out

        return flops

class Block(nn.Module):

    def __init__(self, tgt_len, res_gate, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.tgt_len = tgt_len
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.dim = dim 
        self.mlp_ratio = mlp_ratio
        self.res_gate = res_gate

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.res_gate(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.res_gate(x)
        return x 
    
    def flops(self, pruning_layers):
        flops = 0
        remained_fc_out = sum(self.res_gate.weight)/self.res_gate.weight.size(0)
        remained_fc_hidden = sum(self.mlp.hidden_gate1.weight)/self.mlp.hidden_gate1.weight.size(0)
        if pruning_layers == 2: 
            # norm 
            flops += self.dim* self.tgt_len
            # attn 
            flops += self.attn.flops(self.tgt_len, remained_fc_out)
            # norm2 
            flops += self.dim* self.tgt_len
            # mlp
            flops += 2 * self.tgt_len * self.dim * self.dim * self.mlp_ratio * remained_fc_hidden * remained_fc_out
        elif pruning_layers == 1:
            flops += 2 * self.tgt_len * self.dim * self.dim * self.mlp_ratio * remained_fc_hidden * remained_fc_out
        else:
            flops += self.attn.flops(self.tgt_len, remained_fc_out)
        return flops

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
    def flops(self):
        flops = self.num_patches * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        return flops


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class VisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_heads = num_heads

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches + 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.res_gate = GateLayer(embed_dim, embed_dim, [1, 1, -1])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(tgt_len = num_patches, res_gate = self.res_gate,
                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.res_gate(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def forward_dummy(self,img):
        x = self.forward_features(img)
        x = self.head(x)
        return x

    def flops(self, pruning_layers):
        flops = 0
        if pruning_layers > 1:
            remained_fc_out = sum(self.res_gate.weight)/self.res_gate.weight.size(0)
            flops += self.patch_embed.flops() * remained_fc_out
        for i, block in enumerate(self.blocks):
            flops += block.flops(pruning_layers)
        # head
        flops += self.num_features * self.num_classes
        return flops