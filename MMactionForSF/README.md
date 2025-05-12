### 环境依赖 / 配置
PyTorch  1.8.1  
Python  3.8(ubuntu18.04)  
CUDA  11.1  
GPU V100-32GB(32GB) * 1  
CPU 10 vCPU Intel Xeon Processor (Skylake, IBRS)  
### 部署步骤
1. 安装对应环境依赖
```
    pip install -r requirements.txt
```
2. setup.py 
```
    python setup.py develop
```
### 文件目录结构
- my/
  - MMactionForSF/
    - Checkpionts/
    - configs/
    - Datasets/
    - demo/
    - docker/
    - docs/
    - tests/
    - tools/
    - work_dirs/
    - mmaction/
    - mmaction2.egg-info/
    - mmdet/
    - LICENSE
    - Loss0.0025.pdf
    - Loss0.0075.pdf
    - Loss0.075.pdf
    - Loss0.1.pdf
    - README.md
    - requirements.txt
    - setup.cfg
    - setup.py
### 重要文件
1. my/MMactionForSF/configs/SF/  
**该路径内的文件为包含模型训练参数等配置文件**  
config文件由多个python字典与变量组成，可能根据文件路径不同的原因而需要修改的参数包括：
- 数据集标注与图片路径 (第112行到第116行代码)
  - train_images_root = '/root/autodl-tmp/my/MMactionForSF/Datasets/Interaction/images/train'  
  - train_annotations_root = '/root/autodl-tmp/my/MMactionForSF/Datasets/Interaction/annotations/train'
  - test_images_root = '/root/autodl-tmp/my/MMactionForSF/Datasets/Interaction/images/test'
  - test_annotations_root = '/root/autodl-tmp/my/MMactionForSF/Datasets/Interaction/annotations/test' 
- 训练模型保存路径 (第438行代码):
  - work_dir = './work_dirs/M7-LR0.1'
- batch size 
  - data = dict(videos_per_gpu=16, .......) 第366行代码


2. my/MMactionForSF/Datasets  
**数据集文件**


3. my/MMactionForSF/work_dirs/  
**该路径内存放的是训练完后的权重模型文件**


4. my/MMactionForSF/Checkpionts/  
**该路径为预训练模型存放位置**


5. my/MMactionForSF/mmaction/models/heads/bbox_head.py  
**该文件为此次修改loss function的地方**


6. my/MMactionForSF/mmaction/models/backbones/resnet3d_slowfast.py  
**该文件为主干网络源码**  
### 运行代码
若与上方文件目录结构相同，即文件夹路径为`/root/autodl-tmp/my/MMactionForSF/`，下方代码可直接使用来测试配置完成后的环境依赖是否兼容
- 训练:  
python tools/train.py config文件的路径 --validate
``` 
python tools/train.py configs/SF/MySFConfig_LRTest.py --validate
```
- 测试:   
python tools/test.py config路径 训练后权重文件路径 --eval mAP
```
python tools/test.py configs/SF/MySFConfig_LR0.075.py ./work_dirs/M9-LR0.075/LR0.075.pth --eval mAP
```

