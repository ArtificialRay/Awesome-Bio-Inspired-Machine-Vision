import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
import numpy as np
import torchvision
from VisionRC import vision_rc_ti_gelu
from pVisionRC import pvision_rc_ti_gelu
from sklearn.model_selection import KFold
from timm.scheduler.plateau_lr import PlateauLRScheduler
import benchMarks
import json

torch.manual_seed(3407)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, optimizer, loss_func, epochs,device,batch_size=64,k_fold=3,lr_scheduler=None,print_every=200):
    import copy
    """
    define training process of upon model, 使用K折验证
    return:
    model: the model after training
    train_losses: loss for each epoch in training(最优一折的数据)
    CV_losses: loss for each epoch in validation(最优一折的数据)
    """
    best_val_acc_list = [0.0 for i in range(k_fold)]  # 跟踪每个折最佳的损失
    best_models = []  # 记录每一折训练得到的最优model，然后根据正确率得到一个全局最优model
    best_model_params = copy.deepcopy(model.state_dict())  # 创建当前模型参数的深拷贝
    kf = KFold(n_splits=k_fold) # K折交叉验证
    # 记录每次验证的损失
    train_loss_folds = []
    CV_loss_folds = []
    # 记录每次验证的正确率
    train_acc_folds = []
    CV_acc_folds = []

    for fold, (train_indexes, val_indexes) in enumerate(kf.split(train_loader.dataset)): # enumerate: 将loader中的每个sample和索引配对
        train_sampler = SubsetRandomSampler(train_indexes)  # 告诉dataloader应该加载与len(train_indexes)数量相同，与train_indexes对应的样本
        val_sampler = SubsetRandomSampler(val_indexes)
        curr_train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_loader.dataset, batch_size=batch_size, sampler=val_sampler)
        # 记录训练损失和正确率，用于画图：
        train_loss_epochs = []
        train_acc_epochs = []
        # 记录每次验证的损失和正确率，用于画图：
        CV_loss_epochs = []
        CV_acc_epochs = []

        for epoch in range(epochs):
            print(f" fold{fold+1}, epoch{epoch + 1}:...")
            model.train()  # 设置为训练模式
            loss_val = 0.0
            corrects = 0.0
            for datas,labels in curr_train_loader:
                # datas: (batch_size,18,18)
                # labels: (batch_size,10)
                datas = datas.to(device)
                labels = labels.to(device)

                preds = model(datas)  # 前向传播

                loss = loss_func(preds, labels)  # 计算损失
                optimizer.zero_grad()  # 清除优化器梯度（来自于上一次反向传播）
                loss.backward()  # 反向传播, 计算模型参数梯度
                optimizer.step()  # 根据计算得到的梯度，使用优化器更新模型的参数。

                # 检查准确率
                preds = F.softmax(preds,dim=1)
                preds = torch.argmax(preds, dim=1)
                corrects += torch.sum(preds == labels).item() # item: 将torch张量转为对应的python基本数据类型

                loss_val += loss.item() * datas.size(0)  # 获取loss，并乘以当前批次大小

            train_loss = loss_val / len(train_sampler) # 计算整个模型的总损失
            train_acc = corrects / len(train_sampler) # 计算本次epoch的总正确率
            print(f"Train Loss: {train_loss:.4f}; Train Accuracy: {train_acc:.4f}")
            train_loss_epochs.append(train_loss)
            train_acc_epochs.append(train_acc)
            # 每个epoch都进行评估：
            val_loss,val_acc = validation(model, val_loader, loss_func, device,data_size=len(val_sampler))
            if (best_val_acc_list[fold] < val_acc):  # 出现最优模型时(准确率最大的模型)，保存最优模型
                best_val_acc_list[fold] = val_acc
                best_model_params = copy.deepcopy(model.state_dict())
            # 更新平均loss指标
            if lr_scheduler is not None: # 用validation accuracy的指标更新调度器
                lr_scheduler.step(epoch+1,val_acc)
            CV_loss_epochs.append(val_loss)
            CV_acc_epochs.append(val_acc)


        # 更新每个fold的模型和训练及测试的loss记录
        model.load_state_dict(best_model_params)
        best_models.append(model)
        train_loss_folds.append(np.array(train_loss_epochs))
        CV_loss_folds.append(np.array(CV_loss_epochs))
        train_acc_folds.append(np.array(train_acc_epochs))
        CV_acc_folds.append(np.array(CV_acc_epochs))

    best_val_acc_index = torch.argmax(torch.tensor(best_val_acc_list))
    print(best_val_acc_index)
    model = best_models[best_val_acc_index]
    train_losses = np.concatenate(train_loss_folds)
    train_accs = np.concatenate(train_acc_folds)
    CV_losses = np.concatenate(CV_loss_folds)
    CV_accs = np.concatenate(CV_acc_folds)
    return model, train_losses, train_accs,CV_losses,CV_accs

def validation(model, val_loader, loss_func, device,data_size):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for datas,labels in val_loader:
        # datas: (batch_size,3,64,64)
        # labels: (batch_size,10)
        datas = datas.to(device)
        labels = labels.to(device)

        preds = model(datas)
        loss = loss_func(preds, labels.long())
        loss_val += loss.item() * datas.size(0)

        # 检查准确率
        preds = F.softmax(preds,dim=1)
        preds = torch.argmax(preds, dim=1)
        corrects += torch.sum(preds == labels).item()  # item: 将torch张量转为对应的python基本数据类型

    validation_loss = loss_val / data_size  # 计算整个测试集的总损失
    validation_acc = corrects / data_size
    print(f"validation Loss: {validation_loss:.4f}; validation Accuracy: {validation_acc:.4f}")
    return validation_loss,validation_acc

def save_arrays_in_csv(arrays,path):
    with open(path,"w",newline='') as f:
        np.savetxt(f,arrays,delimiter=",")

def predict_on_test(model,test_loader):
    """
    return predicted values of test_loader, written in ndarray
    :param model:
    :param test_loader:
    :return:
    """
    model.eval()
    predictions = []
    labels = []
    for data,label in test_loader:
        data = data.to(device)
        label = label.to(device)
        preds = model(data)
        preds = F.softmax(preds,dim=1)
        preds = torch.argmax(preds, dim=1)
        predictions.append(preds)
        labels.append(label)
    return torch.cat(predictions,dim=-1).cpu().detach().numpy(),torch.cat(labels,dim=-1).cpu().detach().numpy()

def init_prediction(model,test_loader,loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for data,label in test_loader:
        data = data.to(device)
        label = label.to(device)
        preds = model(data)
        # loss
        loss = loss_func(preds, label.long())
        loss_val += loss.item() * data.size(0)
        # accuracy
        preds = nn.functional.softmax(preds,dim=1)
        preds = torch.argmax(preds, dim=1)
        corrects += torch.sum(preds == label).item()  # item: 将torch张量转为对应的python基本数据类型
    return loss_val / len(test_loader.dataset),corrects/len(test_loader.dataset)

def main():
    # 超参数
    batch_size = 32
    epoches = 25
    folds = 3
    lr = 0.001
    # for pvisionrc:
    # pvisionrc only
    blocks = [2, 2, 4, 2]
    channels = [48, 96, 128, 240]
    res_units = [50, 100, 127, 235]

    # CIFAR10 数据集
    # 加载训练集
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 可以在这里添加其他转换操作，例如归一化
    ])
    trainset = torchvision.datasets.CIFAR10(root='.\\CIFAR10', train=True,download=True,transform=transform)
    #trainset = ImageDataset(trainset)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # 加载测试集
    testset = torchvision.datasets.CIFAR10(root='.\\CIFAR10', train=False, download=True,transform=transform)
    #testset = ImageDataset(testset)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    #model = vision_rc_ti_gelu(num_classes=10,drop_rate=0.1,use_dilation=False).to(device)
    model = pvision_rc_ti_gelu(num_classes=10,drop_rate=0.1,blocks=blocks,channels=channels,res_units=res_units).to(device)

    # ##########################################
    # # test FLOPs
    # input_size = [1,3,32,32]
    # input = torch.randn(input_size).to(device)
    # from torchprofile import profile_macs
    # print(model)
    # model.eval()
    # macs = profile_macs(model,input)
    # model.train()
    # print('model flops:', macs, 'input_size:', input_size)  # 计算model的FLOPs
    # ##########################################

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    # define scheduler:
    lr_noise = 0.01
    noise_range = lr_noise * epoches * folds
    lr_scheduler = PlateauLRScheduler(
        optimizer,
        decay_rate=0.1,
        patience_t=10,
        lr_min=1e-5,
        mode='max',
        warmup_lr_init=1e-6,
        warmup_t=20,
        cooldown_t=0,
        noise_range_t=noise_range,
        noise_seed=42
    )
    init_loss,init_acc = init_prediction(model,trainloader,loss_func)
    print(f"initial loss {init_loss:.4f}; initial accuracy {init_acc:.4f}")
    model, train_loss, train_acc, CV_loss, CV_acc = train(model,
                                                         train_loader=trainloader, loss_func=loss_func,
                                                         optimizer=optimizer, epochs=epoches, device=device,
                                                         batch_size=batch_size, k_fold=folds,lr_scheduler=lr_scheduler)
    # store the model
    torch.save(model.state_dict(),'Vision_rc_ti_3.pth')
    train_loss = np.concatenate((np.array([init_loss]), train_loss))
    train_acc = np.concatenate((np.array([init_acc]), train_acc))
    CV_loss = np.concatenate((np.array([init_loss]), CV_loss))
    CV_acc = np.concatenate((np.array([init_acc]), CV_acc))
    # store train-test data
    save_arrays_in_csv(train_loss, './dataresults/vig_ti_train_loss_CIFAR10_1.csv')
    save_arrays_in_csv(train_acc, './dataresults/vig_ti_train_acc_CIFAR10_1.csv')
    save_arrays_in_csv(CV_loss, './dataresults/vig_ti_test_loss_CIFAR10_1.csv')
    save_arrays_in_csv(CV_acc, './dataresults/vig_ti_test_acc_CIFAR10_1.csv')
    predictions, labels = predict_on_test(model, testloader)
    test_acc_path = "./dataresults/model_test_recording_vig.json"
    test_acc = {"test acc ":
        {
            "parameter size of model": model.param_size,
            "parameter size required grad": model.grad_param_size,
            "accuracy of this model": np.round(benchMarks.accuracy(predictions, labels), 2),
            "macro precision of this model": np.round(benchMarks.precision(predictions, labels), 2),
            "macro recall of this model": np.round(benchMarks.recall(predictions, labels), 2),
            "macro F1 score of this model": np.round(benchMarks.F1(predictions, labels), 2)
        }
    }
    with open(test_acc_path, 'a') as file:
        json.dump(test_acc, file, indent=4)  # 使用indent参数美化输出

    # loss_result_path = "./results/vision_rc_ti_loss_CIFAR10.jpg"
    # acc_result_path = "./results/vision_rc_ti_acc_CIFAR10.jpg"
    # confMatrix_result_path = "./results/vision_rc_ti_conf_matrix_CIFAR10.jpg"
    # benchMarks.plot_performance_all_loss(epoches * folds, train_loss, CV_loss, loss_result_path)
    # benchMarks.plot_performance_all_acc(epoches * folds, train_acc, CV_acc, acc_result_path)
    # predictions, labels = predict_on_test(model, testloader)
    # print(
    #     f"accuracy of this model:{benchMarks.accuracy(predictions, labels):.2f}\n" +
    #     f"macro precision of this model:{benchMarks.precision(predictions, labels):.2f}\n" +
    #     f"macro recall of this model:{benchMarks.recall(predictions, labels):.2f}\n" +
    #     f"macro F1 score of this model:{benchMarks.F1(predictions, labels):.2f}\n"
    # )
    # benchMarks.plot_conf_matrix(predictions, labels, path=confMatrix_result_path)


if __name__ == '__main__':
    main()