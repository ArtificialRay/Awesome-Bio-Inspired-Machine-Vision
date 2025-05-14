import torch
import torch.nn as nn
import torch.nn.functional as F
from VisionRC import vision_rc_ti_gelu
from pVisionRC import pvision_rc_ti_gelu
from vig import vig_ti_224_gelu
from pyramid_vig import pvig_ti_224_gelu

# 超参数
torch.manual_seed(3407)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
epoches = 25
folds = 3
lr = 0.001
# pvisionrc only
blocks = [2,2,4,2]
channels = [48,96,128,240]
res_units = [50,100,127,235]

model_pvig = pvig_ti_224_gelu(num_classes=10,drop_rate=0.1).to(device)
model_pvisionrc = pvision_rc_ti_gelu(num_classes=10,drop_rate=0.1,blocks=blocks,channels=channels,res_units=res_units).to(device)
##########################################
# test FLOPs
input_size = [1,3,32,32]
input = torch.randn(input_size).to(device)
from torchprofile import profile_macs
print('total parameter size:',model_pvig.param_size,'param size required training:',model_pvig.grad_param_size)
model_pvig.eval()
macs = profile_macs(model_pvig,input)
model_pvig.train()
print('vig flops:', macs, 'input_size:', input_size)  # 计算model的FLOPs
##########################################
print('total parameter size:',model_pvisionrc.param_size,'param size required training:',model_pvisionrc.grad_param_size)
model_pvisionrc.eval()
macs = profile_macs(model_pvisionrc,input)
model_pvisionrc.train()
print('visionrc flops:', macs, 'input_size:', input_size)  # 计算model的FLOPs


