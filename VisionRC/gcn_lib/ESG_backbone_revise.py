# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from .reservoir import Reservoir_He100


# from timm.models.layers import DropPath


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm,
                            bias)  # 1x1卷积层作为Max-Relative Graph Convolution的更新函数

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])  # 取所有centroid
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])  # 相同的方法取所有neighbor的特征
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)  # -1: 最后一个维度上取最大值，返回有最大值的节点xj
        b, c, n, _ = x.shape  # batch_size, channel, num_vertices
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)  # 原xi与max(xj-xi)在channel维度上合并
        return self.nn(x)  # BasicConv定义的是1x1 conv + batchnorm + gelu, 作为特征更新函数，注意这个函数不改变原特征图特征维度


class ResGraph(nn.Module):
    """
    Echo state Graph Neural Network
    """

    def __init__(self, wi,wh, leaky=0.2):
        super(ResGraph, self).__init__()
        self.leaky = leaky
        self.backend = Reservoir_He100(wi,wh)

    def forward(self, x, edge_index,y=None):
        # number of vertices in the graph
        x = self.backend.forward(x, 'wi')
        x_i = batched_index_select(x, edge_index[1])  # 取所有centroid
        if y is not None:
            y = self.backend.forward(y, 'wi')
            x_j = batched_index_select(y, edge_index[0])  # 相同的方法取所有neighbor的特征
        else:
            x_j = batched_index_select(x, edge_index[0])

        x_j = torch.sort(x_j - x_i,dim=-1)[0] # 按照x_j - x_i(邻居节点到中心的距离)从小到大排序所有邻居节点相对距离
        x_j = self.backend.forward(x_j, 'wh')  # 状态更新矩阵更新所有邻居节点
        new_state = torch.sum(x_j, dim=-1).unsqueeze(-1) + x
        #new_state = self.backend.forward(x, 'wh') + torch.sum(x_j, dim=-1).unsqueeze(-1)
        state = (1-self.leaky) * x + self.leaky * torch.tanh(new_state) # state update
        return state

class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """

    def __init__(self, wi,wh,leaky=0.2, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'esg':
            self.gconv = ResGraph(wi,wh,leaky=leaky)
        # elif conv == 'mr':
        #     self.gconv = MRConv2d(in_channels, out_channels, act=act, norm=norm, bias=bias)

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, wi, wh,leaky, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(wi, wh,leaky, conv, act, norm,
                                            bias)  # 这些参数将传递到GraphConv2d的定义中去
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape  # batch_size, channels, height, width
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)  # r>1, 对x先进行平均池化操作，如果使用xy_pairwise_distance建图用到
            y = y.reshape(B, C, -1, 1).contiguous()  # 重整为(B,C,HxW,1)，并保持该张量在内存中是连续的
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)  # 通过knn建图得到的图节点边关系
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)  # 使用mr conv/res graph进行推理，也就是原文的max-relative graphconv
        return x.reshape(B, -1, H, W).contiguous()  # 重新整理成(B,C,H,W)形式，但维度是放大的


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels,res_unit,wi,wh,kernel_size=9, dilation=1, leaky=0.2,conv='edge', act='relu', norm=None,
                 bias=True, stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        # kernel_size: knn的中k的值，默认k=9
        # dilation: min(i // 4 + 1, max_dilation)
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n  # number of channels of deep features
        self.r = r
        self.graph_conv = DyGraphConv2d(wi,wh,leaky, kernel_size, dilation, conv,
                                        act, norm, bias, stochastic, epsilon, r)
        self.fc = nn.Sequential(
            nn.Conv2d(res_unit, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),  # graphconv输出通道是res_unit
        )  # 输出映射层Wout
        # 如果使用droppath, 则抛弃当前GraphConv的输出，直接使用原输入
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # nn.Identity()：恒等变换，输入直接作为输出返回;
        self.drop_path = nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print(
                'using relative_pos')  # 是否使用相对位置编码，和绝对位置编码不一样，相对位置编码将编码元素之间的距离或偏移量，如果使用除了vig定义的pos_embed,还会加上这里的relative_pos
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                                                                                        int(n ** 0.5)))).unsqueeze(
                0).unsqueeze(1)  # 返回后又向左拓展两个维度, size=(1,1,3136,3136)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(n, n // (r * r)), mode='bicubic',
                align_corners=False)  # 对原位置编码的最后一个维度进行下采样，使用bicubic采样
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos  # 使用自训练的positional encoding
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x  # 残差连接
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        x = self.fc(x)  # xWout, 投影回原channel
        x = self.drop_path(x) + _tmp  # 残差连接
        return x
