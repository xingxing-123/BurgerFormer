def cal_flops_parameters(net_config, stage=4, vit_stem=True, nb_class=1000):
    def cal_item_flops_parameters(type, input_size, output_size, num_heads=8, mlp_ratios=4, bias=True):
        if type in ['skip']:
            flops, params = 0, 0
        elif type in ['BN', 'GN', 'LN']:
            flops = 2 * input_size[0] * input_size[1] * input_size[2] * input_size[3]
            params = 2 * input_size[1]
        elif type in ['relu6', 'gelu', 'silu']:  # approximation
            flops = input_size[0] * input_size[1] * input_size[2] * input_size[3]
            params = 0
        elif type in ['avgpool']:
            flops = output_size[0] * output_size[1] * output_size[2] * output_size[3]
            params = 0
        elif type in ['conv_1x1', 'dwise_3x3', 'conv_3x3', 'conv_7x7']:
            if type == 'conv_1x1':
                group = 1
                ks = 1
            elif type == 'conv_3x3':
                group = 1
                ks = 3
            elif type == 'conv_7x7':
                group = 1
                ks = 7
            elif type == 'dwise_3x3':
                group = input_size[1]
                ks = 3
            flops = output_size[0] * output_size[1] * output_size[2] * output_size[3] * (input_size[1] / group * ks * ks + (1 if bias else 0))
            params = output_size[1] * (input_size[1] / group * ks * ks + (1 if bias else 0))
        elif type in ['self_atten']:
            flops = 0
            embed_dim = input_size[1]
            n_seq = input_size[2] * input_size[3]
            num_heads = num_heads
            head_dim = embed_dim // num_heads
            # embedding -> qkv
            flops += embed_dim * num_heads * head_dim * 3 * n_seq
            # attention score
            flops += n_seq * n_seq * num_heads * head_dim
            flops += n_seq * n_seq * head_dim * num_heads  # weighted average
            flops += n_seq * (head_dim * num_heads * embed_dim)
            flops *= output_size[0]
            params = embed_dim * 3 * embed_dim + embed_dim * (embed_dim + 1)
        elif type in ['spatial_mlp']:
            flops = 0
            embed_dim = input_size[1]
            n_seq = input_size[2] * input_size[3]
            flops += n_seq * embed_dim * n_seq + n_seq
            flops *= output_size[0]
            params = n_seq * (n_seq + 1)
            # flops = 0
            # embed_dim = input_size[1]
            # n_seq = input_size[2] * input_size[3]
            # hidden_size = mlp_ratios * embed_dim
            # flops += n_seq * embed_dim * hidden_size + hidden_size
            # flops += n_seq * embed_dim * hidden_size + hidden_size
            # flops *= output_size[0]
            # params = 2 * embed_dim * (hidden_size + 1)
        elif type in ['channel_mlp']:
            flops = 0
            in_channel = input_size[1]
            hidden_size = mlp_ratios * in_channel
            out_channel = output_size[1]
            flops += hidden_size * (in_channel * output_size[2] * output_size[3] + 1)
            flops += out_channel * (hidden_size * output_size[2] * output_size[3] + 1)
            act_size = [input_size[0], hidden_size, input_size[1], input_size[2]]
            flops += cal_item_flops_parameters('gelu', act_size, act_size)[0]
            flops *= output_size[0]
            params = in_channel * (hidden_size + 1) + hidden_size * (out_channel + 1)
        return flops, params

    def cal_stem_flops_parameters(input_size, output_size):
        if vit_stem:
            flops, params = 0, 0
            B, C, H, W = input_size
            mid_dim = 24
            f1, p1 = cal_item_flops_parameters('conv_3x3', input_size=[B, C, H, W], output_size=[B, mid_dim, H // 2, W // 2], bias=False)
            f2, p2 = cal_item_flops_parameters('BN', input_size=[B, mid_dim, H // 2, W // 2], output_size=[B, mid_dim, H // 2, W // 2])
            f3, p3 = cal_item_flops_parameters('relu6', input_size=[B, mid_dim, H // 2, W // 2], output_size=[B, mid_dim, H // 2, W // 2])
            flops += f1 + f2 + f3
            params += p1 + p2 + p3
            f1, p1 = cal_item_flops_parameters('conv_3x3', input_size=[B, mid_dim, H // 2, W // 2], output_size=[B, mid_dim, H // 2, W // 2], bias=False)
            f2, p2 = cal_item_flops_parameters('BN', input_size=[B, mid_dim, H // 2, W // 2], output_size=[B, mid_dim, H // 2, W // 2])
            f3, p3 = cal_item_flops_parameters('relu6', input_size=[B, mid_dim, H // 2, W // 2], output_size=[B, mid_dim, H // 2, W // 2])
            flops += f1 + f2 + f3
            params += p1 + p2 + p3
            f1, p1 = cal_item_flops_parameters('conv_3x3', input_size=[B, mid_dim, H // 2, W // 2], output_size=[B, mid_dim, H // 2, W // 2], bias=False)
            f2, p2 = cal_item_flops_parameters('BN', input_size=[B, mid_dim, H // 2, W // 2], output_size=[B, mid_dim, H // 2, W // 2])
            f3, p3 = cal_item_flops_parameters('relu6', input_size=[B, mid_dim, H // 2, W // 2], output_size=[B, mid_dim, H // 2, W // 2])
            flops += f1 + f2 + f3
            params += p1 + p2 + p3
            f, p = cal_item_flops_parameters('conv_3x3', input_size=[B, mid_dim, H // 2, W // 2], output_size=output_size, bias=True)
            flops += f
            params += p

            total_flops = flops
            total_params = params
        else:
            total_flops, total_params = cal_patch_embedding_flops_parameters(input_size, output_size, ks=7)

        return total_flops, total_params

    def cal_patch_embedding_flops_parameters(input_size, output_size, ks=3):
        f1, p1 = cal_item_flops_parameters('conv_{}x{}'.format(ks, ks), input_size=input_size, output_size=output_size, bias=True)
        f2, p2 = cal_item_flops_parameters('GN', input_size=input_size, output_size=output_size)
        return f1 + f2, p1 + p2

    def cal_block_flops_parameters(config, input_size, output_size, multiplier=3, num_heads=8, mlp_ratios=4, expand_ratio=4):
        total_flops, total_params = 0, 0
        for i in range(multiplier):
            if i == 1:
                tmp_input_size = [input_size[0], expand_ratio * input_size[1], input_size[2], input_size[3]]
                tmp_output_size = tmp_input_size
                f1, p1 = cal_item_flops_parameters('GN', input_size, output_size)
                f2, p2 = cal_item_flops_parameters('conv_1x1', input_size, tmp_output_size)
                f3, p3 = cal_item_flops_parameters('gelu', tmp_input_size, tmp_output_size)
                f4, p4 = cal_item_flops_parameters('conv_1x1', tmp_output_size, output_size)
                total_flops += f1 + f2 + f3 + f4
                total_params += p1 + p2 + p3 + p4
            else:
                tmp_input_size = input_size
                tmp_output_size = output_size
            ops = config[str(i + 1)].split('-')
            for op in ops:
                f, p = cal_item_flops_parameters(op, tmp_input_size, tmp_output_size, num_heads, mlp_ratios)
                total_flops += f
                total_params += p
        return total_flops, total_params

    def cal_other_flops_parameters(input_size, output_size):
        total_flops, total_params = 0, 0
        # norm
        f1, p1 = cal_item_flops_parameters('GN', input_size, input_size)
        # head
        B, C, _, _ = input_size
        B, C_out, _, _ = output_size
        f2, p2 = cal_item_flops_parameters('conv_1x1', [B, C, 1, 1], [B, C_out, 1, 1])

        total_flops = f1 + f2
        total_params = p1 + p2
        return total_flops, total_params

    depths = []
    widths = []
    ratios = []
    block_configs = []

    for i in range(stage):
        depths.append(net_config['stage-{}'.format(i + 1)]['macro']['depth'])
        widths.append(net_config['stage-{}'.format(i + 1)]['macro']['width'])
        ratios.append(net_config['stage-{}'.format(i + 1)]['macro']['ratio'])
        block_configs.append(net_config['stage-{}'.format(i + 1)]['meso'])

    num_heads = [-1, -1, 4, 8]
    mlp_ratios = [4, 4, 4, 4]
    H, W = 224, 224
    nb_classes = nb_class
    inout_size = {
        'stem': {
            'in': [1, 3, H, W],
            'out': [1, widths[0], H // 4, W // 4]
        },
        'stage-1': {
            'in': [1, widths[0], H // 4, W // 4],
            'out': [1, widths[1], H // 8, W // 8]
        },
        'stage-2': {
            'in': [1, widths[1], H // 8, W // 8],
            'out': [1, widths[2], H // 16, W // 16]
        },
        'stage-3': {
            'in': [1, widths[2], H // 16, W // 16],
            'out': [1, widths[3], H // 32, W // 32]
        },
        'stage-4': {
            'in': [1, widths[3], H // 32, W // 32],
            'out': [1, widths[3], H // 32, W // 32]
        },
        'other': {
            'in': [1, widths[3], H // 32, W // 32],
            'out': [1, nb_classes, 1, 1]
        },
    }

    total_flops = 0
    total_params = 0

    flops, params = cal_stem_flops_parameters(inout_size['stem']['in'], inout_size['stem']['out'])
    total_flops += flops
    total_params += params

    for i in range(stage):
        if i < stage - 1:
            flops, params = cal_patch_embedding_flops_parameters(inout_size['stage-' + str(i + 1)]['in'], inout_size['stage-' + str(i + 1)]['out'])
            total_flops += flops
            total_params += params
        flops, params = cal_block_flops_parameters(block_configs[i], inout_size['stage-' + str(i + 1)]['in'], inout_size['stage-' + str(i + 1)]['in'], num_heads=num_heads[i], mlp_ratios=mlp_ratios[i], expand_ratio=ratios[i])
        total_flops += depths[i] * flops
        total_params += depths[i] * params

    flops, params = cal_other_flops_parameters(inout_size['other']['in'], inout_size['other']['out'])
    total_flops += flops
    total_params += params

    return total_flops, total_params


if __name__ == "__main__":
    import arch

    f, p = cal_flops_parameters(eval("arch.burgerformer_tiny"), vit_stem=True, nb_class=1000)
    print(f / 1e9, p / 1e6)
