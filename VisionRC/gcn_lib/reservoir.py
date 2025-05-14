import torch
import torch.nn as nn
import os
import sys
sys.path.append('.')

torch.manual_seed(3407)
class Reservoir_He100(nn.Module):
    def __init__(self, wi,wh,device='cpu'
                 ):
        super(Reservoir_He100,self).__init__()
        self.float_type = torch.double
        self.device = device
        # # Dimension
        # self.dim_input = dim_input
        # self.dim_hidden = dim_hidden
        dim_input = wi.shape[0]
        dim_hidden = wh.shape[1]

        # batchnorm, as the layer goes very deep
        self.batchnorm_in = nn.BatchNorm2d(dim_hidden)
        self.batchnorm_rc = nn.BatchNorm2d(dim_hidden)

        self.wi_ori = nn.Parameter(wi,requires_grad=False)
        self.wh_ori = nn.Parameter(wh,requires_grad=False)

        # if gen_weights:
        #     # 原始wi, wh都是由正态分布构成，wh调整谱半径
        #     self.wi_ori = torch.rand(self.dim_input,self.dim_hidden)
        #     self.wh_ori = torch.rand(self.dim_hidden,self.dim_hidden)
        #     self.wh_ori *= spectral_radius / max(abs(torch.linalg.eigvals(self.wh_ori)))  # 调整谱半径
        #     # 让wi_ori和wh_ori作为计算图的一部分，但是不参与训练
        #     self.wi_ori = nn.Parameter(self.wi_ori,requires_grad=False)
        #     self.wh_ori = nn.Parameter(self.wh_ori,requires_grad=False)
        #     # Bit and base
        #     self.half_level = 2 ** n_bit - 1

        # VMM dynamic scaling origin shift
        # self.wi_sum_1d = torch.sum(self.wi_ori, dim=1)
        # self.wh_sum_1d = torch.sum(self.wh_ori, dim=1)

        # location in HE100
        self.wi_repeat = [576 // dim_input, 1]
        self.wh_repeat = [576 // dim_hidden, 1]
        self.wi_loc = [0, 0, self.wi_repeat[0] * (dim_input + 1), dim_hidden]
        self.wh_loc = [0, dim_hidden, self.wh_repeat[0] * (dim_hidden), dim_hidden]
        # self.deploy_weights()

    def deploy_weights(self, write_repeat=1):
        """
        write_repeat: 重复将参数写入多少次, 最后上版到He100时再使用
        """
        # scale wi and wh to range
        self.wi, self.scale_wi_a, self.scale_wi_b = self.weight_quantization(self.wi_ori)
        self.wh, self.scale_wh_a, self.scale_wh_b = self.weight_quantization(self.wh_ori)
        # input matrix wi and reservoir update matrix wh maps to device
        self.wi_mapping = self.wi.repeat(self.wi_repeat)
        self.wi_ori_mapping = self.wi_ori.repeat(self.wi_repeat)
        self.wh_mapping = self.wh.repeat(self.wh_repeat)
        self.wh_ori_mapping = self.wh_ori.repeat(self.wh_repeat)

    def forward(self, in_mat, weight_type,iter_time=10):
        # Weight multiplication
        # in_mat: input matrix, one sample per column(17个column，17个节点/samples)
        # wi_wh_wphi: 'wi', 'wh'

        assert weight_type in ['wi', 'wh'], 'weight must be wi or wh'
        if weight_type == 'wi':
            w = self.wi_ori
            in_mat = in_mat.permute(0, 3, 2, 1)
            result = torch.matmul(in_mat, w)
            result = result.permute(0, 3, 2, 1)
            result = self.batchnorm_in(result)
            return result
        elif weight_type == 'wh':
            w = self.wh_ori
            in_mat = in_mat.permute(0, 3, 2, 1)
            result = torch.matmul(in_mat, w)
            result = result.permute(0, 3, 2, 1)
            result = self.batchnorm_rc(result)
            return result
        else:
            raise Exception('Must be wi or wh')

    # use when deploy:
        # assert weight_type in ['wi', 'wh'], 'weight must be wi or wh'
        # n_sample_per_batch = in_mat.shape[1]  # Samples per batch
        # batch_size = 32
        # if weight_type == 'wi':
        #     addr = self.wi_loc
        #     repeat = self.wi_repeat
        #     wi_a = self.scale_wi_a
        #     wi_b = self.scale_wi_b
        #     dim_out = self.dim_hidden
        # elif weight_type == 'wh':
        #     addr = self.wh_loc
        #     repeat = self.wh_repeat
        #     wh_a = self.scale_wh_a
        #     wh_b = self.scale_wh_b
        #     dim_out = self.dim_hidden
        # else:
        #     raise Exception('Must be wi or wh')
        #
        # # Initialize output
        # result = torch.zeros(dim_out, n_sample_per_batch).to(self.device)

        # Quantization
        # in_mat,_ = self.data_quantization(in_mat)
        # in_mat_q = in_mat.repeat(repeat)
        # input_sum_1d = torch.sum(in_mat_q.transpose(1,0), dim=1)
        # # multiply and rescale the result
        # # 缩放规则见codenote
        # #result = mvm_bitwise_concat_push_fast_144k(sdk,in_mat_q,addr, repeat, it_time = iter_time)
        # result = torch.mm(in_mat_q.transpose(1,0),self.wi_mapping if weight_type=='wi' else self.wh_mapping)
        # result = torch.tensor(result).to(self.device)
        # if weight_type == 'wi':
        #     result = result * wi_a + torch.outer(input_sum_1d,wi_b)
        # elif weight_type == 'wh':
        #     result = result * wh_a + torch.outer(input_sum_1d,wh_b)
        # result,_ = self.data_quantization(result,isint=1) # scale back the result
        # #return result.transpose(1,0) / 576
        # return result.transpose(1, 0)

    def data_quantization(self,data_float,isint=0):
        # isint = 1 -> return quantized values as integer levels
        # isint = 0 -> return quantized values as float numbers with the same range as input
        if self.half_level <= 0:
            return data_float, 0

        # 根据clamp_std量化：将数据限制在-clamp_std * std ~ clamp_std * std的范围内，目前先不用
        std = data_float.std()
        # if clamp_std != None and clamp_std != 0:
        #     data_float = torch.clamp(data_float, min = -clamp_std * std, max = clamp_std * std)

        scale = abs(data_float).max()

        # 按如下公式进行数据量化
        data_quantized = (data_float / scale * self.half_level).round()
        quant_scale = 1 / scale * self.half_level
        if isint == 0:  # 若isint=0，说明希望返回浮点数，则按照下列公式进行反量化，quant_scale=1
            data_quantized = data_quantized * scale / self.half_level
            quant_scale = 1

        return data_quantized, quant_scale

    def weight_quantization(self,input)->torch.tensor:
        # Column-wise min/max of input
        input_min = torch.min(input, dim=0)[0]
        input_max = torch.max(input, dim=0)[0]
        # Scaling coefficient (see output)
        a = (input_max - input_min) / self.half_level
        b = input_min
        #c = -8

        # 每一个节点的min和max，都复制到该节点label的每一个dimension上
        input_min = input_min.repeat(input.shape[0], 1)
        input_max = input_max.repeat(input.shape[0], 1)

        input_int = (input - input_min) / (
                    input_max - input_min) * self.half_level  # 输入数据缩放公式: input-b/a, 数据会被缩放到[0,self.half_level]的范围内， b和a会在函数最后返回
        #input_int = torch.round(input_int).to(torch.int) + c  # 强制用int表示，调整到[-8,+7]

        return input_int,a,b