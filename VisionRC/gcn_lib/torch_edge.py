# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.计算x中每两个点之间的距离

    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def part_pairwise_distance(x, start_idx=0, end_idx=1):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        x_part = x[:, start_idx:end_idx]
        x_square_part = torch.sum(torch.mul(x_part, x_part), dim=-1, keepdim=True)
        x_inner = -2*torch.matmul(x_part, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square_part + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    """
    Compute pairwise distance of a point cloud.计算点集x和点集y中每两个点之间的距离
    公式:Σ(x-y)^2 = Σx^2 - Σ2xy + Σy^2
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1)) # 计算x和y的矩阵乘积
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True) # 因为前面已经进行在dim=1的L2 normalize, 这里按dim=-1相加元素平方会在每个channel得到1
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1) #上述结果调整到可以相加的维度后加起来;


def dense_knn_matrix(x, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        ### memory efficient implementation ###
        n_part = 10000
        if n_points > n_part: # isotropic结构可能用到,图最多只能有10^4对点，超过这个数量，就分开n个组计算pairwise distance
            nn_idx_list = []
            groups = math.ceil(n_points / n_part)
            for i in range(groups):
                start_idx = n_part * i # 从第多少个节点开始
                end_idx = min(n_points, n_part * (i + 1)) # 索引到第多少个节点
                dist = part_pairwise_distance(x.detach(), start_idx, end_idx)
                if relative_pos is not None:
                    dist += relative_pos[:, start_idx:end_idx]
                _, nn_idx_part = torch.topk(-dist, k=k) # 当前组里所有pair中距离最小的前k个
                nn_idx_list += [nn_idx_part]
            nn_idx = torch.cat(nn_idx_list, dim=1) # 将所有节点的索引concat在一起（dim=1表示节点个数）
        else:
            dist = pairwise_distance(x.detach()) # 计算x中每对点之间的距离
            if relative_pos is not None: # 如果有，加入相对编码位置
                dist += relative_pos
            _, nn_idx = torch.topk(-dist, k=k) # 选前k个距离最小的点作为邻居
        ######
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1) # 中心索引，即x中所有点的索引
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16, relative_pos=None):
    """Get KNN based on the pairwise distance.
    这种情况下，原patches中的14x14=196个块会与经过avg_pool2d(r=2)的块7x7=49来建图
    似乎是可以将更全局/粗糙的信息（经过下采样后）和更局部/全面信息（下采样前）连接，GNN可以同时捕捉全局特征和局部上下文
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points, k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1) # (B,C,HxW,1) -> (B,HxW,C)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach()) # .detach()从反向传播中暂时去除，因为计算distance的过程不需要加入梯度计算
        if relative_pos is not None: #dist: x和y中所有点的距离和
            dist += relative_pos # 计算x和y的distance时还要加上relative_position
        _, nn_idx = torch.topk(-dist, k=k) # 每个在x中的点，选前9个(isotropic 中k会增大，pyramid中k=9)在y中距离最小的点建图,nn_idx是x每一个点(中心)在y中选中点的索引
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1) # center就是x中所有点本身的索引
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    根据vig.py中Grapher的定义，在更深层的网络中，使用KNN建图得到的邻居节点会越少

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation # 扩张因子，控制邻域采样的稀疏程度
        self.stochastic = stochastic # 是否随机采样
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic: # vig.py中如果use_stochastic=True
            if torch.rand(1) < self.epsilon and self.training: # 如果触发，那么随机对已有的邻居节点（最后一个维度值）采样，只保留randnum指向的的邻居节点索引
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else: # 否则按照self.dilation规定的间距采样顶点
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation] #直接按self.dilation规定的间距采样顶点
        return edge_index


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k # kernel_size of conv
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):
        if y is not None: # 如果对x进行过池化了
            #### normalize
            x = F.normalize(x, p=2.0, dim=1) # 将每个通道中的 (HxW, 1) 空间进行 L2 范数归一化
            y = F.normalize(y, p=2.0, dim=1)
            ####
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation, relative_pos) # 返回x中的点和y中的点的邻接关系
        else:
            #### normalize
            x = F.normalize(x, p=2.0, dim=1)
            ####
            edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos) # 返回x中每个像素的邻接关系，edge_index[0]: neighbors, edge_index[1]: center
        return self._dilated(edge_index)
