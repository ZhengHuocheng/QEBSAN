'''
description: 
Version: 
Author: Zheng Huocheng
Date: 2024-04-08 18:52:24
LastEditors: Zheng Huocheng
LastEditTime: 2024-04-18 14:21:39
'''
# import sys
# sys.path.append("D:\\Project_Design\\")


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ATmodel import HyperModel
from qiskitModule import Q_Model
import random

def predict():
    # device = torch.device("cuda:0") # 如果GPU设备可以，则用GPU；否则用CPU设备

    # 加载文本特征
    textPath = "D:\\Project_Design\\out_txtFea\\"
    textFea = [np.loadtxt(
                    textPath+str(num)+'.txt',
                    dtype=float , delimiter=",") for num in range(4)]
    # 将文本特征增加一个维度
    textFea = [np.expand_dims(item, axis=0) for item in textFea]
    # 将文本特征拼接起来
    textFea = np.concatenate(textFea, axis=0)
    # 将文本特征转换为tensor
    textFea = torch.tensor(textFea).to(torch.float32).cuda()   #4 64 768
    # print("textFea是4，64，768吗",textFea.shape)

    # 加载音频特征
    context_audioPath = "D:\\Project_Design\\out_audioFea\\au_context.txt"
    target_audioPath = "D:\\Project_Design\\out_audioFea\\au_target.txt"
    context_auFea = np.loadtxt(context_audioPath, dtype=float, delimiter=",")
    context_auFea = np.expand_dims(context_auFea, axis=1)
    target_auFea = np.loadtxt(target_audioPath, dtype=float, delimiter=",")
    target_auFea = np.expand_dims(target_auFea, axis=0)
    target_auFea = np.expand_dims(target_auFea, axis=1)
    target_auFea = torch.tensor(target_auFea)
    context_auFea = torch.tensor(context_auFea)

    # data =  [textFea,target_auFea,context_auFea]
    # data_loader = DataLoader(
    #         data,
    #         batch_size=None,
    #         shuffle=True,
    #         pin_memory=True
    #     )

    ED_model = HyperModel(batchsize = 1)
    ED_model = torch.load(f'D:\\Project_Design\\result\\state\\model80.pt')
    ED_model.eval()
    # for batch in data_loader:
    #     print(batch)
    #     textInput = batch[0][0].cuda().to(torch.float32)
    #     tWavPath = batch[0][1].to(torch.float32).cuda()
    #     cWavPath = batch[0][2].to(torch.float32).cuda()
    textFea = torch.unsqueeze(textFea, dim=0).to(torch.float32).cuda()
    # print("后text音频特征是",textFea.shape)
    target_auFea = torch.unsqueeze(target_auFea, dim=0).to(torch.float32).cuda()
    context_auFea = torch.unsqueeze(context_auFea, dim=0).to(torch.float32).cuda()
    # print("后音频特征是",target_auFea.shape,context_auFea.shape)
    sar,sen = ED_model.forward(textFea,target_auFea,context_auFea)
    # print("Encoder-Decoder产生：",sar,sen)
    q_model = Q_Model(bszi=1)
    q_model = torch.load(f'QEBSAN\\QEBSAN\\Quantum_result\\state\\qModel100.pt')
    q_model.eval()
    sar,sen = q_model(sar,sen)
    # print("QNN产生：",sar,sen)

    sar = sar.cpu().numpy().tolist()[0]
    sen = sen.cpu().numpy().tolist()[0]

    def measure(lst):
        print(lst)
        if sum(lst) > 1:
            # 生成随机索引
            # print("函数内",lst)
            tagedOne_List = []
            for  i in range(len(lst)):
                if lst[i] ==1 :
                    tagedOne_List.append(i)    #值为1的元素的在lst中的索引
            define_index = random.randint(0, len(tagedOne_List)-1)   #选择lst中索引为tageOne_List(define_index)
            
            for i in range(len(lst)):
                if i == tagedOne_List[define_index]:
                    lst[i] = 1
                else:
                    lst[i] = 0
            return lst
        elif sum(lst) == 0:
            define_index = random.randint(0, len(lst)-1)
            lst[define_index] = 1
            return lst
        else:
            return lst

    sar = measure(sar)
    sent = measure(sen)

    sar_label = '讽刺' if sar[0] == 1 else '非讽刺'
    sent_label = '积极' if sent[0] == 1 else '中性' if sent[1] == 1 else '消极'
    print(sar,sent)
    print(sar_label,sent_label)
    return sent_label,sar_label


# predict()

