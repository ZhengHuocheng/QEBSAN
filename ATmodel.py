import datetime
from sklearn.preprocessing import MinMaxScaler
from os import TMP_MAX
import os
from ExpDataset import MustardDataset
import torch
from torch import nn, float32, autograd
from torch.nn import Module, BatchNorm1d, functional
from torch.nn import Dropout, Flatten, Linear, \
    Softmax, GRU, AvgPool3d, MultiheadAttention
from torch.utils.data import DataLoader, dataset
from transformers import AlbertModel, AlbertConfig
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from torch.autograd import Variable
import logging
# import qModule
# txt:torch.Size([4, 64, 768])
# audio:(4, 128)
import torch
import torch.nn as nn


class HyperModel(Module):
    def __init__(self, batchsize):
        super(HyperModel, self).__init__()
        # 初始化HyperModel类
        # 初始化vgg线性层和激活函数
        self.vggLinear = Linear(in_features=1, out_features=24, bias=True)
        self.vggAct = nn.Tanh()

        # 初始化文本线性层和激活函数
        self.textLinear = Linear(in_features=64, out_features=24, bias=True)
        self.textAct = nn.Tanh()

        # 初始化batchsize
        self.batchsize = batchsize
        # intra级单模态特征提取
        # 初始化文本BiGRU，BiGRU是一种将GRU神经网络的单向时间步扩展为双向的神经网络结构
        self.textBiGRU = GRU(
            input_size=768,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

        # 初始化音频BiGRU
        self.audioBiGRU = GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True)

        # 自注意力机制将用于编码文本和音频的特征。它们可以帮助模型更好地理解输入的文本和音频，从而更好地进行预测和分类。
        self.textEncodeAtt = MultiheadAttention(embed_dim=128, num_heads=1)
        self.audioEncodeAtt = MultiheadAttention(embed_dim=128, num_heads=1)  # 128->self.batchsize

        # 初始化权重
        self.targetWeight = nn.Parameter(torch.Tensor([0.5]))
        self.contextWeight = nn.Parameter(torch.Tensor([0.5]))

        # inter级多模态特征融合
        # 初始化多头注意力

        self.interTtoA = MultiheadAttention(128, 4)
        self.interAtoT = MultiheadAttention(128, 4)
        # Decoder模块
        # 初始化GRU
        self.sentimentGRU = GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=False)

        self.sarcasmGRU = GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=False)
        # 注意力头类看收藏,第一个是总的输入特征维度,第二个为所用的注意力头数
        self.sentimentAtt = MultiheadAttention(128, 1)

        self.sarcasmAtt = MultiheadAttention(128, 1)
        # 初始化线性层  diret线性层sentiment
        self.directSent = nn.Linear(in_features=6144, out_features=3, device='cuda:0')  # 18432 -》 28672
        self.sentBN = nn.BatchNorm1d(6144, device='cuda:0',eps=1e-25)  # 18432 -》 28672

        self.directSar = nn.Linear(in_features=12800, out_features=2, device='cuda:0')  # 37376- 57856
        self.sarBN = nn.BatchNorm1d(12800, device='cuda:0',eps=1e-25)  # 37376- 57856
        # 初始化线性层
        # self.re_sar=nn.Linear(3,2)
        # 初始化激活函数
        self.sarScaleAct = nn.Tanh()
        self.dropout = nn.Dropout(p=0.4)
        # decoder

    def Qin(self, text, tWavFea, cWavFea):
        contextsize = text.shape[1]  # 4
        # print("长下文长度：",contextsize)
        text = text
        text = torch.transpose(text, dim0=-1, dim1=-2)
        text = self.textAct(self.textLinear(text))
        text = torch.transpose(text, dim0=-1, dim1=-2)

        # vggish
        # 为测试模型先去掉扩充音频特征的线性层
        # print(tWavFea.shape)
        # print(cWavFea.shape)
        tWavPath = torch.transpose(tWavFea, dim0=-1, dim1=-2)
        cWavPath = torch.transpose(cWavFea, dim0=-1, dim1=-2)
        tAudioFeature = self.vggAct(self.vggLinear(tWavPath))
        cAudioFeature = self.vggAct(self.vggLinear(cWavPath))
        # torch.Size([128, 1, 128, 24])
        # torch.Size([128, 3, 128, 24])
        # print(tAudioFeature.shape)
        # print(cAudioFeature.shape)
        tAudioFeature = torch.transpose(tAudioFeature, dim0=-1, dim1=-2)
        cAudioFeature = torch.transpose(cAudioFeature, dim0=-1, dim1=-2)
        # 为测试模型先去掉扩充音频特征的线性层
        audio = torch.squeeze(
            torch.cat([cAudioFeature, tAudioFeature], dim=1), dim=1)
        if torch.isnan(text).sum().item() > 0 or torch.isnan(audio).sum().item() > 0:
            return None, None
        else:
            # 将文本、音频按照contextsize分割成多个块
            textChunks = torch.chunk(text, contextsize, 1)
            audioChunks = torch.chunk(audio, contextsize, 1)
            # 初始化块列表
            tChunkList = []
            aChunkList = []
            # 初始化上下文列表
            thnList = []
            ahnList = []
            # 初始化编码器注意力权重列表
            tEncoderAttWeight = []
            aEncoderAttWeight = []

            # 遍历文本块
            for tChunk in textChunks:
                # 将tChunk的维度从1维转换为2维
                tChunk = torch.squeeze(tChunk, dim=1)  # dim=0

                # 使用textBiGRU对tChunk进行编码
                tChunk, thn = self.textBiGRU(tChunk)

                # 将tChunk的维度从0维转换(对换到)1维
                query = torch.transpose(tChunk, dim0=0, dim1=1)
                key = torch.transpose(tChunk, dim0=0, dim1=1)
                value = torch.transpose(tChunk, dim0=0, dim1=1)

                # (24,b,32)
                textAttnOutput, textAttnOutputWeights = self.textEncodeAtt(query, key, value)

                # 将textAttnOutput的维度从1维转换为2维
                tChunkList.append(textAttnOutput)


            for aChunk in audioChunks:
                aChunk = torch.squeeze(aChunk, dim=1)

                aChunk, ahn = self.audioBiGRU(aChunk)

                ahnList.append(ahn)
                query = torch.transpose(aChunk, dim0=0, dim1=1)
                key = torch.transpose(aChunk, dim0=0, dim1=1)
                value = torch.transpose(aChunk, dim0=0, dim1=1)

                audioAttnOutput, audioAttnOutputWeights = self.audioEncodeAtt(query, key, value)

                aChunkList.append(audioAttnOutput)
                aEncoderAttWeight.append(audioAttnOutputWeights)

            targetText = tChunkList[-1]

            targetAudio = aChunkList[-1]

            '''
            text
            '''
            tConAtt = [torch.exp(
                torch.tanh(targetText + tContext) + 1e-7)
                for tContext in tChunkList[:-1]
            ]
            # 计算tConAttSum
            tConAttSum = tConAtt[0] + tConAtt[1] + tConAtt[2]
            # 计算tIntraAttOutput
            tIntraAttOutput = [
                (tConAtt[i] / tConAttSum) * tChunkList[i] for i in range(3)
            ]
            # 计算tIntraAttOutput
            tIntraAttOutput = tIntraAttOutput[0] + \
                              tIntraAttOutput[1] + tIntraAttOutput[2]

            # 计算tIntraAttOutput
            tIntraAttOutput = targetText + \
                              torch.tanh(
                                  self.targetWeight * targetText
                                  + self.contextWeight * tIntraAttOutput
                              )

            '''
            text
            '''

            '''
            audio
            '''
            aConAtt = [torch.exp(
                torch.tanh(targetAudio + aContext) + 1e-7)
                for aContext in aChunkList[:-1]
            ]

            aConAttSum = aConAtt[0] + aConAtt[1] + aConAtt[2]

            aIntraAttOutput = [
                (aConAtt[i] / aConAttSum) * aChunkList[i] for i in range(3)
            ]

            aIntraAttOutput = aIntraAttOutput[0] + \
                              aIntraAttOutput[1] + aIntraAttOutput[2]

            aIntraAttOutput = targetAudio + \
                              torch.tanh(
                                  self.targetWeight * targetAudio
                                  + self.contextWeight * aIntraAttOutput
                              )

            '''
            audio
            '''

            textKey, textValue, textQuery = \
                tIntraAttOutput, tIntraAttOutput, tIntraAttOutput
            audioKey, audioValue, audioQuery = \
                aIntraAttOutput, aIntraAttOutput, aIntraAttOutput


            interTtoA, _ = self.interTtoA(
                query=textQuery, key=audioKey, value=audioValue)
            interAtoT, _ = self.interAtoT(
                query=audioQuery, key=textKey, value=textValue)

            ATCat = torch.cat([interTtoA, interAtoT], dim=0)


            catFeature = torch.cat([ATCat], dim=0)

            flatten = nn.Flatten()
            # 以上为多模态ENCODER部分
            # 以下为DECODER部分架构
            # 获取情感特征
            sentGRU, sentHidden = self.sentimentGRU(catFeature)
            # 获取情感注意力
            sentAtt, _ = self.sentimentAtt(sentGRU, sentGRU, sentGRU)
            # 转置情感注意力
            sentAtt = torch.transpose(sentAtt, dim0=0, dim1=1)
            # 将情感注意力展开
            sentAttFlatten = flatten(sentAtt)

            # 将情感注意力展开的输入过批规范化
            sentOutput = self.sentBN(sentAttFlatten)
            # 将情感注意力展开的输入过全连接
            sentOutput = self.directSent(sentOutput)

            # 获取情感注意力top1
            sentTop1 = torch.topk(sentOutput, 1)[1]

            # 将情感注意力top1重复128次
            sentTop1 = sentTop1.repeat(1, 128)
            # 将情感注意力top1添加维度
            sentTop1 = torch.unsqueeze(sentTop1, 1)
            # 将情感注意力top1转置
            sentTop1 = torch.transpose(sentTop1, dim0=0, dim1=1)

            # 将情感特征，情感隐藏层，情感注意力top1拼接
            sarInput = torch.cat([catFeature, sentHidden, sentTop1], dim=0)
            # 获取情感特征
            sarGRU, sarHidden = self.sarcasmGRU(sarInput)
            # 获取情感注意力
            sarAtt, _ = self.sarcasmAtt(sarGRU, sarGRU, sarGRU)
            # 转置情感注意力
            sarAtt = torch.transpose(sarAtt, dim0=0, dim1=1)
            # 将情感注意力展开和情感注意力展开拼接
            sarAttFlatten = torch.cat([flatten(sarAtt), flatten(sarAtt)], dim=1)

            # 将情感注意力展开的输入过bn
            sarOutput = self.sarBN(sarAttFlatten)
            # 将情感注意力展开的输入过全连接
            sarOutput = self.directSar(sarOutput)

            return sarOutput, sentOutput

    def forward(self, text, tWavFea, cWavFea):

        contextsize = text.shape[1]
        # print(text.shape)  #torch.Size([489, 4, 64, 768])
        # print(audio.shape) #torch.Size([489, 4, 128])
        text = text

        text = torch.transpose(text, dim0=-1, dim1=-2)
        text = self.textAct(self.textLinear(text))
        text = torch.transpose(text, dim0=-1, dim1=-2)

        # vggish
        # print(tWavFea.shape)
        # print(cWavFea.shape)
        tWavPath = torch.transpose(tWavFea, dim0=-1, dim1=-2)
        cWavPath = torch.transpose(cWavFea, dim0=-1, dim1=-2)
        tAudioFeature = self.vggAct(self.vggLinear(tWavPath))
        cAudioFeature = self.vggAct(self.vggLinear(cWavPath))
        # torch.Size([128, 1, 128, 24])
        # torch.Size([128, 3, 128, 24])
        # print(tAudioFeature.shape)
        # print(cAudioFeature.shape)
        tAudioFeature = torch.transpose(tAudioFeature, dim0=-1, dim1=-2)
        cAudioFeature = torch.transpose(cAudioFeature, dim0=-1, dim1=-2)
        # torch.Size([1, 24, 128]) torch.Size([3, 24, 128])
        # 为测试模型先去掉扩充音频特征的线性层
        audio = torch.squeeze(
            torch.cat([cAudioFeature, tAudioFeature], dim=1), dim=1)   #预测时,内外两部dim由1-》0
        # print("音频特征形状是",audio.shape)  128 4 1 128

        # print("text",torch.isnan(text).sum().item())
        # print("audio",torch.isnan(audio).sum().item())
        if torch.isnan(text).sum().item() > 0 or torch.isnan(audio).sum().item() > 0:
            return None, None
        # print("audio",audio)
        else:
            # print(text,"\n",image,"\n",audio)
            textChunks = torch.chunk(text, contextsize, 1)  # ([489, 1, 64, 768])
            audioChunks = torch.chunk(audio, contextsize, 1)
            tChunkList = []
            aChunkList = []
            thnList = []
            ahnList = []
            tEncoderAttWeight = []
            aEncoderAttWeight = []

            # print("=============文本领域形状\n")
            for tChunk in textChunks:
                tChunk = torch.squeeze(tChunk, dim=1)  # ([489, 64, 768])
                tChunk, thn = self.textBiGRU(tChunk)  # tChunk  torch.Size([489, 64, 128])
                thnList.append(thn)
                # tChunk, thn = self.textBiGRU2(tChunk)
                # tChunk=tChunk.reshape(tChunk.shape[0],3,-1)

                query = torch.transpose(tChunk, dim0=0, dim1=1)
                key = torch.transpose(tChunk, dim0=0, dim1=1)
                value = torch.transpose(tChunk, dim0=0, dim1=1)

                # (24,b,32)
                textAttnOutput, textAttnOutputWeights = self.textEncodeAtt(query, key, value)
                # textAttnOutput torch.Size([64, 489, 128])
                tChunkList.append(textAttnOutput)
                tEncoderAttWeight.append(textAttnOutputWeights)


            # print("=============声音领域形状\n")
            for aChunk in audioChunks:
                aChunk = torch.squeeze(aChunk, dim=1)

                aChunk, ahn = self.audioBiGRU(aChunk)

                ahnList.append(ahn)
                query = torch.transpose(aChunk, dim0=0, dim1=1)
                key = torch.transpose(aChunk, dim0=0, dim1=1)
                value = torch.transpose(aChunk, dim0=0, dim1=1)

                audioAttnOutput, audioAttnOutputWeights = \
                    self.audioEncodeAtt(query, key, value)

                aChunkList.append(audioAttnOutput)
                aEncoderAttWeight.append(audioAttnOutputWeights)

            targetText = tChunkList[-1]
            targetAudio = aChunkList[-1]

            '''
            text
            '''
            tConAtt = [torch.exp(
                torch.tanh(targetText + tContext) + 1e-7)
                for tContext in tChunkList[:-1]
            ]
            tConAttSum = tConAtt[0] + tConAtt[1] + tConAtt[2]
            tIntraAttOutput = [
                (tConAtt[i] / tConAttSum) * tChunkList[i] for i in range(3)
            ]
            tIntraAttOutput = tIntraAttOutput[0] + \
                              tIntraAttOutput[1] + tIntraAttOutput[2]

            tIntraAttOutput = targetText + \
                              torch.tanh(
                                  self.targetWeight * targetText
                                  + self.contextWeight * tIntraAttOutput
                              )

            '''
            text
            '''

            '''
            audio
            '''
            aConAtt = [torch.exp(
                torch.tanh(targetAudio + aContext) + 1e-7)
                for aContext in aChunkList[:-1]
            ]

            aConAttSum = aConAtt[0] + aConAtt[1] + aConAtt[2]

            aIntraAttOutput = [
                (aConAtt[i] / aConAttSum) * aChunkList[i] for i in range(3)
            ]

            aIntraAttOutput = aIntraAttOutput[0] + \
                              aIntraAttOutput[1] + aIntraAttOutput[2]

            aIntraAttOutput = targetAudio + \
                              torch.tanh(
                                  self.targetWeight * targetAudio
                                  + self.contextWeight * aIntraAttOutput
                              )

            '''
            audio
            '''

            textKey, textValue, textQuery = \
                tIntraAttOutput, tIntraAttOutput, tIntraAttOutput
            audioKey, audioValue, audioQuery = \
                aIntraAttOutput, aIntraAttOutput, aIntraAttOutput

            interTtoA, _ = self.interTtoA(
                query=textQuery, key=audioKey, value=audioValue)
            interAtoT, _ = self.interAtoT(
                query=audioQuery, key=textKey, value=textValue)

            # ITCat = torch.cat([interTtoI, interItoT], dim=0)
            ATCat = torch.cat([interTtoA, interAtoT], dim=0)
            # IACat = torch.cat([interAtoI, interItoA], dim=0)

            catFeature = torch.cat([ATCat], dim=0)

            flatten = nn.Flatten()
            print("flatten",flatten)
            sentGRU, sentHidden = self.sentimentGRU(catFeature)

            sentAtt, _ = self.sentimentAtt(sentGRU, sentGRU, sentGRU)
            sentAtt = torch.transpose(sentAtt, dim0=0, dim1=1)

            # print(sentAtt) 不是这里NAN
            # print(torch.isnan(sentAtt))

            # faltten应该含有NAN
            sentAttFlatten = flatten(sentAtt)
            # print("sentATT",torch.isnan(sentAttFlatten))
            # print("sentAttFlatten:", sentAttFlatten)

            # 将张量转换为numpy数组，进行归一化处理
            input_array = sentAttFlatten.detach().cpu().numpy()

            # 归一化处理
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_array = scaler.fit_transform(input_array)

            # 将归一化后的数组转换回张量
            sentAttFlatten = torch.tensor(normalized_array,device='cuda:0')

            # 下句开始出现nan  #1\冻结参数， 2、float32
            sentAttFlatten =  sentAttFlatten
            sentOutput = self.sentBN(sentAttFlatten)
            # print("sentoutput1:", sentOutput)
            # print(self.sentBN.running_mean)
            # print(self.sentBN.running_var)

            sentOutput = self.directSent(sentOutput)  # sentOutput -》sentAttFlatten
            # sentOutput = torch.where(torch.isnan(sentOutput), torch.tensor(1e-6), sentOutput)  # 将NaN值替换为0.0
            # sentOutput = torch.clamp(sentOutput, 1e-6, 10)
            # print(torch.isnan(sentOutput))
            # print("sentoutput2:", sentOutput)
            # sentment

            # sarcasm
            sentTop1 = torch.topk(sentOutput, 1)[1]
            sentTop1 = sentTop1.repeat(1, 128)
            sentTop1 = torch.unsqueeze(sentTop1, 1)
            sentTop1 = torch.transpose(sentTop1, dim0=0, dim1=1)

            sarInput = torch.cat([catFeature, sentHidden, sentTop1], dim=0)
            sarGRU, sarHidden = self.sarcasmGRU(sarInput)
            sarAtt, _ = self.sarcasmAtt(sarGRU, sarGRU, sarGRU)
            sarAtt = torch.transpose(sarAtt, dim0=0, dim1=1)
            sarAttFlatten = torch.cat([flatten(sarAtt), flatten(sarAtt)], dim=1)

            # torch.nn.init.constant_(self.sarBN.running_mean, 0)
            # torch.nn.init.constant_(self.sarBN.running_var, 1)
            # sarAttFlatten = 100 * sarAttFlatten
            sarOutput = self.sarBN(sarAttFlatten)

            # print(sarOutput)
            sarOutput = self.directSar(sarOutput)  # sarOutput->sarAttFlatten
            # sarOutput = torch.where(torch.isnan(sarOutput), torch.tensor(1e-6), sarOutput)  # 将NaN值替换为0.0
            # sarOutput = torch.clamp(sarOutput, 1e-6, 10)
            # sarOutput=self.re_sar(sarOutput)
            # sarOutput = torch.clamp(sarOutput, 1e-6, 10)
            # print(torch.isnan(sarOutput))
            # print("saroutput2:", sarOutput)

            return sarOutput, sentOutput


# 定义一个函数testModel，输入一个模型进行模型测试
def testModel(modelPATH):
    # 创建一个测试数据集
    testData = MustardDataset(datatye='test')
    # 设置batchsize
    batchsize = 4
    # 创建一个数据加载器
    data_loader = DataLoader(
        testData,
        batch_size=batchsize,
        shuffle=True,
        pin_memory=True
    )

    # model = HyperModel(batchsize=batchsize).cuda()
    # 加载预训练模型
    model = torch.load(modelPATH)
    # model.load_state_dict(torch.load(modelPATH))
    # 设置模型为评估模式
    model.eval()
    # 使用no_grad()函数，不计算梯度
    with torch.no_grad():
        outputsar, outputsent = [], []
        # 初始化输出列表和标签列表
        tarsar, tarsent = [], []
        # 遍历数据加载器
        for batch in data_loader:
            # 获取输入数据
            textInput = batch[0][0].cuda().to(torch.float32)
            tWavPath = batch[0][2].to(torch.float32).cuda()
            cWavPath = batch[0][3].to(torch.float32).cuda()

            # 获取标签
            sarLabel = batch[1][0].to(torch.float32).cuda()
            sentLabel = batch[1][1].to(torch.float32).cuda()
            # 获取输出
            sar, sent = model(textInput, tWavPath, cWavPath)
            if sar is None and sent is None:
                continue
            else:
                # 获取标签的类别
                label_sar = np.argmax(
                    sarLabel.cpu().detach().numpy(), axis=-1)
                label_sent = np.argmax(
                    sentLabel.cpu().detach().numpy(), axis=-1)
                # 获取输出的类别
                pred_sar = np.argmax(
                    sar.cpu().detach().numpy(), axis=1)
                pred_sent = np.argmax(
                    sent.cpu().detach().numpy(), axis=1)
                # 将输出和标签添加到输出列表和标签列表中
                outputsar.append(pred_sar)
                outputsent.append(pred_sent)
                tarsar.append(label_sar)
                tarsent.append(label_sent)
        # print("outputsar",outputsar)
        # print("tarsar",tarsar)
        # print("outputsent",outputsent)
        # print("tarsent",tarsent)
        # 将输出列表和标签列表中的元素连接起来
        outputsar = np.concatenate(
            np.array(outputsar, dtype=object))
        outputsent = np.concatenate(
            np.array(outputsent, dtype=object))
        tarsar = np.concatenate(
            np.array(tarsar, dtype=object))
        tarsent = np.concatenate(
            np.array(tarsent, dtype=object))
        outputsar = np.array(outputsar.astype(int))
        tarsar = np.array(tarsar.astype(int))
        outputsent = np.array(outputsent.astype(int))
        tarsent = np.array(tarsent.astype(int))
        print("outputsar",outputsar)
        print("tarsar",tarsar)
        print("outputsent",outputsent)
        print("tarsent",tarsent)

        # print("target: ", tarsar)
        # print("output: ", outputsar)
        # 计算f1值和准确率
        sar_f1 = f1_score(
            tarsar, outputsar, average='micro')
        sent_f1 = f1_score(
            tarsent, outputsent, average='micro')
        sar_acc = accuracy_score(
            tarsar, outputsar)
        sent_acc = accuracy_score(
            tarsent, outputsent)
        # 打印结果
        # print('test tarsar:', tarsar)
        # print('test outputsar:', outputsar)
        # print('test tarsent:', tarsent)
        # print('test outputsent:', outputsent)
        # logger.info(('test-result sar-f1:%f sent-f1:%f emo-f1:%f' +
        #             'sar-acc:%f sent-acc:%f emo-acc:%f\n')
        #             % (sar_f1, sent_f1, emo_f1,
        #             sar_acc, sent_acc, emo_acc))
        logger.info(('test-result sar-f1:%f sent-f1:%f' +
                     'sar-acc:%f sent-acc:%f\n')
                    % (sar_f1, sent_f1,
                       sar_acc, sent_acc))


modelPATH = 'result\state\80_Epochs_2024-03-08-22-06-19/model.pt'


# 定义训练和评估函数
def trainEval():
    # 定义训练批次大小
    batchsize = 32
    # 定义训练轮数
    epochs = 100  # 80->40
    need_train_models_epoch = [20,40,60,80,100]  #分别保存这些epoch下的模型
    # 定义总的训练轮数
    all_epochs = 80

    # 获取当前时间
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # 定义状态保存路径
    root_stateSavePATH = f'D:\\Project_Design/result/state/'
    # 定义模型保存路径
    root_modelSavePATH = f'D:\\Project_Design/result/state/'

    # 加载训练数据集
    data = MustardDataset(datatye='train')
    # 加载验证数据集
    valData = MustardDataset(datatye='dev')

    # 定义训练数据加载器
    data_loader = DataLoader(
        data,
        batch_size=batchsize,
        shuffle=True,
        pin_memory=True
    )
    # 定义验证数据加载器
    val_loader = DataLoader(
        valData,
        batch_size=9,  # 90->32
        shuffle=True,
        pin_memory=True,
    )
    # 定义超模型
    model = HyperModel(batchsize=batchsize).cuda()

    # 遍历模型中的每一层
    for m in model.modules():
        # 如果层是线性层
        if isinstance(m, (Linear)):
            # 使用Xavier初始化线性层
            nn.init.xavier_uniform_(m.weight)
    # for m in model.modules():
    #     if isinstance(m, (BatchNorm1d)):
    #         m.eval()

    # torch.nn.init.constant_(model.sentBN.running_mean, 0)
    # torch.nn.init.constant_(model.sentBN.running_var, 1)
    # torch.nn.init.constant_(model.sarBN.running_mean, 0)
    # torch.nn.init.constant_(model.sarBN.running_var, 1)

    # 加载模型
    # model=torch.load(modelPATH)

    # 定义损失函数
    lossFun = nn.CrossEntropyLoss().cuda()
    lossFun_val = nn.CrossEntropyLoss().cuda()

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0015)  # lr 0.001->0.0001

    # 计算模型参数数量
    total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))

    # 创建文件夹
    # os.makedirs(f'.\\result/state/{all_epochs}_Epochs_{now}', exist_ok=True)

    # 循环epochs次
    # torch.autograd.set_detect_anomaly(True)
    for _ in range(epochs):
        print("epoch: ", _)
        logger.info('epoch:' + str(_))
        # 定义训练步骤
        train_step = 0
        # print(_,data_loader)
        # 遍历data_loader中的每一个batch
        for batch in data_loader:
            with autograd.detect_anomaly():
                train_step += 1
                # 将model设置为训练模式
                model.train()
                # 将batch中的textInput，imageInput，audioInput转换为cuda浮点数
                textInput = batch[0][0].to(torch.float32).cuda()  # cuda()往后提了一位
                tWavPath = batch[0][2].to(torch.float32).cuda()
                cWavPath = batch[0][3].to(torch.float32).cuda()

                # 将batch中的sarLabel，sentLabel转换为cuda浮点数
                sarLabel = batch[1][0].to(torch.float32).cuda()
                sentLabel = batch[1][1].to(torch.float32).cuda()

                # 将textInput，imageInput，audioInput转换为Variable
                textInput = Variable(textInput, requires_grad=True)
                tWavPath = Variable(tWavPath, requires_grad=True)
                cWavPath = Variable(cWavPath, requires_grad=True)

                # 将sarLabel，sentLabel转换为Variable
                sarLabel = Variable(sarLabel, requires_grad=True)
                sentLabel = Variable(sentLabel, requires_grad=True)
                sar, sent = model(textInput, tWavPath, cWavPath)
                if sar is None and sent is None:
                    continue
                else:
                    # print("output: ",sar, sent)
                    # print("target: ",sarLabel, sentLabel)
                    # print("=========================================")

                    # 将sar，sent转换为cuda浮点数
                    sar = sar.to(torch.float32)
                    sent = sent.to(torch.float32)

                    # 将sarLabel，sentLabel转换为numpy数组，并获取最大值,确定每个场景的情感和讽刺结果
                    sarArgmax = torch.argmax(sarLabel, dim=-1)
                    sentArgmax = torch.argmax(sentLabel, dim=-1)
                    # 计算损失
                    loss1 = lossFun(sar, sarArgmax)
                    loss2 = lossFun(sent, sentArgmax)
                    loss = (loss1 + loss2) / 2

                    # 将loss设置为可求导
                    loss.requires_grad_(True)
                    # 打印损失
                    logger.info('loss1:{0} loss2:{1} loss:{2}\n'.format(
                        loss1.item(), loss2.item(), loss.item()))

                    # 每训练10次，计算准确率
                    if train_step % 10 == 0:  # 除以10-》2，尽量与batchsize保持一致
                        # 将sarLabel，sentLabel转换为numpy数组，并获取最大值
                        label_sar = np.argmax(
                            sarLabel.cpu().detach().numpy(), axis=-1)
                        label_sent = np.argmax(
                            sentLabel.cpu().detach().numpy(), axis=-1)
                        # label_emo = np.argmax(
                        #     emoLabel.cpu().detach().numpy(), axis=-1)
                        # 将sar，sent转换为numpy数组，并获取最大值
                        pred_sar = np.argmax(
                            sar.cpu().detach().numpy(), axis=1)
                        pred_sent = np.argmax(
                            sent.cpu().detach().numpy(), axis=1)
                        # print("label_sar", label_sar)
                        # print("pred_sar", pred_sar)
                        # print("label_sent", label_sent)
                        # print("pred_sent", pred_sent)

                        sar_f1 = f1_score(label_sar, pred_sar, average='micro')
                        sent_f1 = f1_score(label_sent, pred_sent, average='micro')
                        sar_acc = accuracy_score(
                            label_sar, pred_sar)
                        sent_acc = accuracy_score(
                            label_sent, pred_sent)

                        logger.info(('train result sar-f1:%f sent-f1:%f ' +
                                     'sar-acc:%f sent-acc:%f \n')
                                    % (sar_f1, sent_f1,
                                       sar_acc, sent_acc))
                        model.eval()
                        with torch.no_grad():
                            outputsar, outputsent, outputemo = [], [], []
                            tarsar, tarsent, taremo = [], [], []
                            for batch in val_loader:
                                textInput = batch[0][0].cuda().to(torch.float32)
                                tWavPath = batch[0][2].to(torch.float32).cuda()
                                cWavPath = batch[0][3].to(torch.float32).cuda()

                                sarLabel = batch[1][0].to(torch.float32).cuda()
                                sentLabel = batch[1][1].to(torch.float32).cuda()
                                sar, sent = model(textInput, tWavPath, cWavPath)
                                if sar is None and sent is None:
                                    continue
                                else:
                                    sarArgmax = torch.argmax(sarLabel, dim=-1)
                                    sentArgmax = torch.argmax(sentLabel, dim=-1)
                                    loss1_val = lossFun_val(sar, sarArgmax)
                                    loss2_val = lossFun_val(sent, sentArgmax)
                                    loss_val = (loss1_val + loss2_val) / 2
                                    logger.info('val loss1:%f loss2:%f loss:%f\n'
                                                % (loss1_val.item(),
                                                   loss2_val.item(),
                                                   loss_val.item()))

                                    label_sar = np.argmax(
                                        sarLabel.cpu().detach().numpy(), axis=-1)
                                    label_sent = np.argmax(
                                        sentLabel.cpu().detach().numpy(), axis=-1)
                                    pred_sar = np.argmax(
                                        sar.cpu().detach().numpy(), axis=1)
                                    pred_sent = np.argmax(
                                        sent.cpu().detach().numpy(), axis=1)
                                    outputsar.append(pred_sar)
                                    outputsent.append(pred_sent)
                                    tarsar.append(label_sar)
                                    tarsent.append(label_sent)

                            outputsar = np.concatenate(
                                np.array(outputsar))
                            outputsent = np.concatenate(
                                np.array(outputsent))
                            tarsar = np.concatenate(
                                np.array(tarsar))
                            tarsent = np.concatenate(
                                np.array(tarsent))

                            sar_f1 = f1_score(
                                tarsar, outputsar, average='micro')
                            sent_f1 = f1_score(
                                tarsent, outputsent, average='micro')
                            sar_acc = accuracy_score(
                                tarsar, outputsar)
                            sent_acc = accuracy_score(
                                tarsent, outputsent)
                            logger.info(('val-result sar-f1:%f sent-f1:%f' +
                                         ' sar-acc:%f sent-acc:%f\n')
                                        % (sar_f1, sent_f1,
                                           sar_acc, sent_acc))

                    optimizer.zero_grad()
                    loss.backward()
                    # 裁剪前计算total_norm,默认设置norm_type=2
                    # total_norm = 0
                    #
                    # for p in model.parameters():
                    #     param_norm = p.data.norm(2)
                    #     total_norm += param_norm.item() ** 2
                    # total_norm = total_norm ** (1. / 2)
                    # print(total_norm)
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
                    optimizer.step()
        # os.makedirs(f'result/state/{all_epochs}_Epochs_{now}/', exist_ok=True)
        if _ in need_train_models_epoch:
            stateSavePATH = root_stateSavePATH + "state" + str(_) + ".pt"
            modelSavePATH = root_modelSavePATH + "model" + str(_) + ".pt"
            torch.save(model.state_dict(), stateSavePATH)
            torch.save(model, modelSavePATH)


if __name__ == '__main__':
    # 设置日志文件的模式为'w'
    logging.basicConfig(filemode='w')
    # 获取当前文件名
    logger = logging.getLogger(__name__)
    # 设置日志的级别为INFO
    logger.setLevel(level=logging.INFO)
    # 设置日志文件的位置
    handler = logging.FileHandler(
        "D:\\Project_Design\\result\\log\\train80epoch_test.txt")
    # 设置日志文件的位置
    handler.setLevel(logging.INFO)
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 设置日志文件格式
    handler.setFormatter(formatter)
    # 添加日志处理器
    logger.addHandler(handler)
    # 打印日志信息
    logger.info("\n**********Start print log**********")
    # 调用训练和评估函数
    # trainEval()
    # 调用测试模型函数
    testModel(f'D:\\Project_Design/result/state/model80.pt')

