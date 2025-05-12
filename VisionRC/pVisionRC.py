# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib.ESG_backbone_revise import Grapher, act_layer


# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .9, 'interpolation': 'bicubic',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'patch_embed.proj', 'classifier': 'head',
#         **kwargs
#     }
#
#
# default_cfgs = {
#     'vig_224_gelu': _cfg(
#         mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#     ),
#     'vig_b_224_gelu': _cfg(
#         crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#     ),
# }


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x  # .reshape(B, C, N, 1)


class Stem(nn.Module):
    """ Image to Visual Embedding
    用于将一定面积的块编码成一个像素，这个编码的像素将作为后面建图使用的节点
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            # pyramid 结构里的Stem只使用3层卷积
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """

    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class VisionRC(torch.nn.Module):
    def __init__(self, opt):
        super(VisionRC, self).__init__()
        #print(opt)
        res_units = opt.res_units
        leaky = opt.leaky
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        spectral_radius = 0.9

        blocks = opt.blocks  # 每个stage的vig块个数
        self.n_blocks = sum(blocks)
        channels = opt.channels  # 每个stage的使用channel数
        reduce_ratios = [2, 2, 1, 1]  # 定义每个stage的reduce_ratio, 这个值会影响相对位置编码和建图(r>1,将会在原输入x和平均池化后的输入y之间建图，以减少计算量)
        dpr = [x.item() for x in torch.linspace(0, drop_path,
                                                self.n_blocks)]  # stochastic depth decay rule;如果使用了droppath, 这里是每个块的输出不被启用的概率; 否则这里都是0
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 16 // max(num_knn)  # 如果是imagenet, 分子是49(最后一层输入图是7x7); 这里换成CIFAR-10,分子是16(最后一层输入图是4x4)

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 16,16))  # （注意要和stem模块的输出长宽相同！）参与训练的positional embedding
        HW = 16 * 16  # original size: 因为CIFAR-10的图像数据太小了，所以去掉stem层

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            # reservoir parameters:
            wi = torch.rand(channels[i], res_units[i])
            wh = torch.rand(res_units[i], res_units[i])
            wh *= spectral_radius / max(abs(torch.linalg.eigvals(wh)))
            if i > 0:  # 跳过第一个stage
                self.backbone.append(Downsample(channels[i - 1], channels[i]))  # 扩展通道数，但特征图的高和宽各减半
                HW = HW // 4
            for j in range(blocks[i]):
                # cifar-10 has size 32, where k's range may exceed
                k = num_knn[idx] if num_knn[idx] < HW else HW
                self.backbone += [
                    Seq(Grapher(channels[i], res_units[i],wi,wh , k, min(idx // 4 + 1, max_dilation), leaky, conv, act, norm,
                                bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                relative_pos=True),  # pyramid结构使用了相对位置编码
                        nn.Dropout(opt.dropout),
                        FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx]),
                        nn.Dropout(opt.dropout)
                        )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()
        size = 0
        size_require_grad = 0
        for p in self.parameters():
            if p.requires_grad:
                size_require_grad += p.nelement()
            size += p.nelement()
        print('param size required training: {}'.format(size_require_grad))
        print('Total param size: {}'.format(size))  # 衡量模型的参数数量以判断它的大小
        self.param_size = size
        self.grad_param_size = size_require_grad
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        # x = inputs + self.pos_embed
        B, C, H, W = x.shape
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


def pvision_rc_ti_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, leaky=0.2, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.leaky = leaky
            self.k = 9  # neighbor num (default:9),pyramid结构里面邻居都设置为9
            self.conv = 'esg'  # graph conv layer {esg, mr}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.dropout = drop_rate  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = kwargs.get("blocks")  # number of basic blocks in the backbone
            self.channels = kwargs.get("channels")#[48, 96, 240, 384]  # number of channels of deep features
            self.res_units = kwargs.get("res_units") # number of units in reservoir
            self.n_classes = num_classes  # Dimension of out_channels
            self.emb_dims = 1024  # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = VisionRC(opt)
    # model.default_cfg = default_cfgs['vig_224_gelu']
    return model


# def pvig_s_224_gelu(pretrained=False, **kwargs):
#     class OptInit:
#         def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
#             self.k = 9  # neighbor num (default:9)
#             self.conv = 'mr'  # graph conv layer {edge, mr}
#             self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
#             self.norm = 'batch'  # batch or instance normalization {batch, instance}
#             self.bias = True  # bias of conv layer True or False
#             self.dropout = 0.0  # dropout rate
#             self.use_dilation = True  # use dilated knn or not
#             self.epsilon = 0.2  # stochastic epsilon for gcn
#             self.use_stochastic = False  # stochastic for gcn, True or False
#             self.drop_path = drop_path_rate
#             self.blocks = [2, 2, 6, 2]  # number of basic blocks in the backbone
#             self.channels = [80, 160, 400, 640]  # number of channels of deep features
#             self.n_classes = num_classes  # Dimension of out_channels
#             self.emb_dims = 1024  # Dimension of embeddings
#
#     opt = OptInit(**kwargs)
#     model = VisionRC(opt)
#     # model.default_cfg = default_cfgs['vig_224_gelu']
#     return model


# def pvig_m_224_gelu(pretrained=False, **kwargs):
#     class OptInit:
#         def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
#             self.k = 9  # neighbor num (default:9)
#             self.conv = 'mr'  # graph conv layer {edge, mr}
#             self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
#             self.norm = 'batch'  # batch or instance normalization {batch, instance}
#             self.bias = True  # bias of conv layer True or False
#             self.dropout = 0.0  # dropout rate
#             self.use_dilation = True  # use dilated knn or not
#             self.epsilon = 0.2  # stochastic epsilon for gcn
#             self.use_stochastic = False  # stochastic for gcn, True or False
#             self.drop_path = drop_path_rate
#             self.blocks = [2, 2, 16, 2]  # number of basic blocks in the backbone
#             self.channels = [96, 192, 384, 768]  # number of channels of deep features
#             self.n_classes = num_classes  # Dimension of out_channels
#             self.emb_dims = 1024  # Dimension of embeddings
#
#     opt = OptInit(**kwargs)
#     model = VisionRC(opt)
#     # model.default_cfg = default_cfgs['vig_224_gelu']
#     return model
#
#
# def pvig_b_224_gelu(pretrained=False, **kwargs):
#     class OptInit:
#         def __init__(self, num_classes=1000, drop_path_rate=0.0, **kwargs):
#             self.k = 9  # neighbor num (default:9)
#             self.conv = 'mr'  # graph conv layer {edge, mr}
#             self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
#             self.norm = 'batch'  # batch or instance normalization {batch, instance}
#             self.bias = True  # bias of conv layer True or False
#             self.dropout = 0.0  # dropout rate
#             self.use_dilation = True  # use dilated knn or not
#             self.epsilon = 0.2  # stochastic epsilon for gcn
#             self.use_stochastic = False  # stochastic for gcn, True or False
#             self.drop_path = drop_path_rate
#             self.blocks = [2, 2, 18, 2]  # number of basic blocks in the backbone
#             self.channels = [128, 256, 512, 1024]  # number of channels of deep features
#             self.n_classes = num_classes  # Dimension of out_channels
#             self.emb_dims = 1024  # Dimension of embeddings
#
#     opt = OptInit(**kwargs)
#     model = VisionRC(opt)
#     # model.default_cfg = default_cfgs['vig_b_224_gelu']
#     return model
