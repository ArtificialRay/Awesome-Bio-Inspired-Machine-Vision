import torch
from VisionRC import vision_rc_ti_gelu
from vig import vig_ti_224_gelu

# 超参数
torch.manual_seed(3407)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
epoches = 25
folds = 3
lr = 0.001


model_vig = vig_ti_224_gelu(num_classes=10,drop_rate=0.1,use_dilation=False).to(device)
model_visionrc = vision_rc_ti_gelu(num_classes=10,drop_rate=0.1).to(device)
##########################################
# test FLOPs
input_size = [1,3,32,32]
input = torch.randn(input_size).to(device)
from torchprofile import profile_macs
print('total parameter size:',model_vig.param_size,'param size required training:',model_vig.grad_param_size)
model_vig.eval()
macs = profile_macs(model_vig,input)
model_vig.train()
print('vig flops:', macs, 'input_size:', input_size)  # 计算model的FLOPs
##########################################
print('total parameter size:',model_visionrc.param_size,'param size required training:',model_visionrc.grad_param_size)
model_visionrc.eval()
macs = profile_macs(model_visionrc,input)
model_visionrc.train()
print('visionrc flops:', macs, 'input_size:', input_size)  # 计算model的FLOPs


