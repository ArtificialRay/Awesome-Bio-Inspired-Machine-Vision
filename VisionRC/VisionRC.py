# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib.ESG_backbone_revise import Grapher, act_layer

# from timm.models.layers import DropPath


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
#     'gnn_patch16_224': _cfg(
#         crop_pct=0.9, input_size=(3, 224, 224),
#         mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#     ),
# }


class FFN(nn.Module):
    """根据论文，FFN的作用是node feature transformation, 缓解传统GNN中over-smoothing的现象"""

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
        x = self.drop_path(x) + shortcut  # 仍然保持残差连接
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding,将输入图像转为视觉词嵌入
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            # 每个卷积层扩充输出维度，若输入大小224，最后会形成(out_dim,14,14)的图片
            nn.Conv2d(in_dim, out_dim // 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 4),
            act_layer(act),
            nn.Conv2d(out_dim // 4, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            # nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            # nn.BatchNorm2d(out_dim),
            # act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class VisionRC(torch.nn.Module):
    def __init__(self, opt):
        super(VisionRC, self).__init__()
        channels = opt.n_filters
        res_unit = opt.res_units
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        leaky = opt.leaky
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        spectral_radius = 0.9

        # reservoir parameters:
        wi = torch.rand(channels,res_unit)
        wh = torch.rand(res_unit,res_unit)
        wh *= spectral_radius / max(abs(torch.linalg.eigvals(wh)))

        self.stem = Stem(out_dim=channels, act=act)  # Stem

        dpr = [x.item() for x in
               torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule，是否关闭张量某些前向传播的部分，这里都不关闭
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in
                   torch.linspace(k, 2 * k - 2, self.n_blocks)]  # number of knn's k，原论文中k随着深度逐渐增加，int(x.item())表示向下取整
        print('num_knn', num_knn)
        max_dilation = 16 // max(num_knn)  # 196//18

        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 4, 4))  # sequence 14x14=196 inputs对应的位置编码，初始化为0，自动参与反向传播

        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels,res_unit,wi,wh, num_knn[i], min(i // 4 + 1, max_dilation),leaky, conv, act, norm,
                                              bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      nn.Dropout(opt.dropout),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i]),
                                      nn.Dropout(opt.dropout)
                                      # FFN的hidden feature是in_channelx4
                                      ) for i in range(self.n_blocks)])  # 一个Grapher相当于一个ViG block，default r=1, r=2用于调试
            # dilation最小是1，最大是196//max(num_knn)=10
            # 对于ti, dilation最大是4; s和b最大是5
        else:
            self.backbone = Seq(*[Seq(Grapher(channels,res_unit,wi,wh, num_knn[i], 1, leaky,conv, act, norm,
                                              bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      nn.Dropout(opt.dropout),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i]),
                                      nn.Dropout(opt.dropout)
                                      ) for i in range(self.n_blocks)])
        # 输出层: 1x1卷积层组成, 结构 192x1024 -> 1024xclassNum
        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()  # 初始化模型
        size = 0
        size_require_grad = 0
        for p in self.parameters():
            if p.requires_grad:
                size_require_grad += p.nelement()
            size += p.nelement()
        # print('param size required training: {}'.format(size_require_grad))
        # print('Total param size: {}'.format(size))  # 衡量模型的参数数量以判断它的大小
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
        B, C, H, W = x.shape

        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


def vision_rc_ti_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=10, leaky=0.2,drop_path_rate=0.0, drop_rate=0.0, num_knn=9,use_dilation=True, **kwargs):
            self.leaky = leaky
            self.k = num_knn  # neighbor num (default:9)
            self.conv = 'esg'  # graph conv layer {esg, mr}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.n_blocks = kwargs.get('blocks')  # number of basic blocks in the backbone，网络中有多少网络块
            self.n_filters = 192  # number of channels of deep features
            self.res_units = 127 # number of units in reservoir
            self.n_classes = num_classes  # Dimension of out_channels
            self.dropout = drop_rate  # dropout rate
            self.use_dilation = use_dilation  # use dilated knn or not，建图时取的节点间距
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = False  # stochastic for gcn, True or False，指每次建图的时候随机采样邻居节点,否则按照dilation指定的间距采样点
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = VisionRC(opt)
    # model.default_cfg = default_cfgs['gnn_patch16_224'] # 这一行应该是设置了一个新参数
    return model

def pvision_rc_ti_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, leaky=0.2,drop_path_rate=0.0, drop_rate=0.0, num_knn=9,**kwargs):
            self.leaky = leaky
            self.k = 9 # neighbor num (default:9),pyramid结构里面邻居都设置为9
            self.conv = 'esg' # graph conv layer {esg, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.dropout = 0.0 # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,6,2] # number of basic blocks in the backbone
            self.channels = [48, 96, 240, 384] # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.emb_dims = 1024 # Dimension of embeddings

    opt = OptInit(**kwargs)
    model = VisionRC(opt)
    #model.default_cfg = default_cfgs['vig_224_gelu']
    return model

# a bigger version of visionRC, we can discuss it later

# def vig_s_224_gelu(pretrained=False, **kwargs):
#     class OptInit:
#         def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
#             self.k = num_knn  # neighbor num (default:9)
#             self.conv = 'mr'  # graph conv layer {edge, mr}
#             self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
#             self.norm = 'batch'  # batch or instance normalization {batch, instance}
#             self.bias = True  # bias of conv layer True or False
#             self.n_blocks = 16  # number of basic blocks in the backbone，网络中有多少网络块
#             self.n_filters = 320  # number of channels of deep features
#             self.n_classes = num_classes  # Dimension of out_channels
#             self.dropout = drop_rate  # dropout rate
#             self.use_dilation = True  # use dilated knn or not
#             self.epsilon = 0.2  # stochastic epsilon for gcn
#             self.use_stochastic = False  # stochastic for gcn, True or False
#             self.drop_path = drop_path_rate
#
#     opt = OptInit(**kwargs)
#     model = VisionRC(opt)
#     # model.default_cfg = default_cfgs['gnn_patch16_224']
#     return model
#
#
# def vig_b_224_gelu(pretrained=False, **kwargs):
#     class OptInit:
#         def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
#             self.k = num_knn  # neighbor num (default:9)
#             self.conv = 'mr'  # graph conv layer {edge, mr}
#             self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
#             self.norm = 'batch'  # batch or instance normalization {batch, instance}
#             self.bias = True  # bias of conv layer True or False
#             self.n_blocks = 16  # number of basic blocks in the backbone
#             self.n_filters = 640  # number of channels of deep features
#             self.n_classes = num_classes  # Dimension of out_channels
#             self.dropout = drop_rate  # dropout rate
#             self.use_dilation = True  # use dilated knn or not
#             self.epsilon = 0.2  # stochastic epsilon for gcn
#             self.use_stochastic = False  # stochastic for gcn, True or False
#             self.drop_path = drop_path_rate
#
#     opt = OptInit(**kwargs)
#     model = VisionRC(opt)
#     # model.default_cfg = default_cfgs['gnn_patch16_224']
#     return model
