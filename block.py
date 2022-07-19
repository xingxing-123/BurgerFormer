import torch
import torch.nn as nn
from search_config import CONFIG
from operation import OPS
from timm.models.layers import DropPath
from operation import *


class SkipBlock(nn.Module):
    def __init__(self, meso_arch_str):
        super().__init__()
        self.op = OPS[meso_arch_str]({})

    def forward(self, x):
        out = self.op(x)
        return out


class NormMixerActBlock(nn.Module):
    def __init__(
        self,
        meso_arch_str,
        dim,
        H,
        W,
        num_heads=5,
        mlp_ratio=4,
        act_layer=nn.GELU,
        drop=0.,
        drop_path=0.,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        use_expand=False,
        expand_ratio=4,
    ):
        # num_heads: stage 3:4 stage 4:8
        super().__init__()

        self.use_expand = use_expand
        if self.use_expand:
            self.expand_start = nn.Sequential(OPS['GN']({'dim': dim}), nn.Conv2d(dim, expand_ratio * dim, 1), act_layer())
            self.expand_end = nn.Sequential(nn.Conv2d(expand_ratio * dim, dim, 1))
            dim = expand_ratio * dim

        ops_list = []
        ops_str = meso_arch_str.split('-')
        for op in ops_str:
            if op in ['skip']:
                continue
            if op in CONFIG['normopact_module']['act']:
                ops_list.append(OPS[op]({}))
            if op in CONFIG['normopact_module']['norm']:
                ops_list.append(OPS[op]({'dim': dim, 'norm_track': True}))
            if op in CONFIG['normopact_module']['meat_op' if use_expand else 'bread_op'] or op == 'channel_mlp':
                ops_list.append(OPS[op]({
                    'in_channels': dim,  # conv_1x1, dwise_3x3
                    'out_channels': dim,  # conv_1x1, dwise_3x3
                    'groups': dim,  # dwise_3x3
                    'dim': dim,  # self-atten
                    'head': num_heads,  # self-atten
                    'seq_len': int(H * W),  # spatial_mlp
                    'in_features': dim,  # channel_mlp
                    'hidden_features': mlp_ratio * dim,  # channel_mlp,
                    'out_features': dim,  # channel_mlp
                    'act_layer': act_layer,  # channel_mlp
                    'drop': drop,  # channel_mlp
                }))
        if len(ops_list) == 0:
            ops_list.append(OPS['skip']({}))
        self.ops = nn.Sequential(*ops_list)

        if self.use_expand:
            dim = dim // expand_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale = nn.Identity()

    def forward(self, x):
        if self.use_expand:
            x = self.expand_start(x)
        x = self.ops(x)
        if self.use_expand:
            x = self.expand_end(x)
        if self.use_layer_scale:
            x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x
        x = self.drop_path(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        meso_arch,
        dim,
        H,
        W,
        num_heads=5,
        mlp_ratio=4,
        act_layer=nn.GELU,
        drop=0.,
        drop_path=0.,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        multiplier=3,
        expand_ratio=4,
    ):
        super().__init__()

        norm_mixer_act_block_list = []
        skip_block_list = []

        for i in range(multiplier):
            norm_mixer_act_block_list.append(
                NormMixerActBlock(
                    meso_arch["{}".format(i + 1)],
                    dim,
                    H,
                    W,
                    num_heads,
                    mlp_ratio,
                    act_layer,
                    int(meso_arch["is_drop"][i]) * drop,
                    int(meso_arch["is_drop"][i]) * drop_path,
                    meso_arch["use_layer_scale"][i],
                    layer_scale_init_value,
                    use_expand=True if i == 1 else False,
                    expand_ratio=expand_ratio,
                ))
            for j in range(i + 1):
                skip_block_list.append(SkipBlock(meso_arch["{}->{}".format(j, i + 1)]))

        self.norm_mixer_act_blocks = nn.ModuleList(norm_mixer_act_block_list)
        self.skip_blocks = nn.ModuleList(skip_block_list)
        self.multiplier = multiplier

    def forward(self, x):
        outs = [x]
        pos = 0
        for i in range(self.multiplier):
            out = self.norm_mixer_act_blocks[i](outs[i])

            for j in range(i + 1):
                if isinstance(self.skip_blocks[pos + j].op, nn.Identity):
                    out += self.skip_blocks[pos + j](outs[j])
            pos += i + 1
            outs.append(out)

        return outs[-1]


if __name__ == "__main__":
    meso_arch = {
        "1": "skip-GN-avgpool-skip-skip",
        "2": "skip-GN-conv1x1-skip-gelu",
        "3": "skip-skip-conv1x1-skip-skip",
        "0->1": "skip",
        "0->2": "none",
        "1->2": "none",
        "0->3": "none",
        "1->3": "skip",
        "2->3": "none",
    }
    a = Block(meso_arch, 320)
    x = torch.randn(128, 320, 14, 14)

    b = a(x)
    print(b.size())