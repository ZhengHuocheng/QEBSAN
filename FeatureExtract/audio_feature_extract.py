# 此功能完全实现

'''
description: audio feature extraction
Version: 
Author: Zheng Huocheng
Date: 2024-03-03 16:43:16
LastEditors: Zheng Huocheng
LastEditTime: 2024-03-06 00:50:48
'''
from posixpath import sep
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AlbertTokenizer, AlbertConfig, AlbertModel
import numpy as np
import csv
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms
import os
import math
import time
import sys
import resampy

def progress_bar(finish_tasks_num, tasks_num):
    percentage = round(finish_tasks_num/tasks_num * 100)
    print("\r进度{}%：".format(percentage), "▓" * (percentage // 2), end="\n")
    sys.stdout.flush()

def readcsv(fileName):
    '''
    将所有对话的目标语句、上下文语句、讽刺标签、情感标签、情绪标签都集合到uttDict中
    目标语句和上下文语句都在utterance list中，以对话的时间顺序展开，最后是目标语句
    现在返回的utterance是所有的utt，最多的语句是12句，最少的语句是2句
    '''
    with open(fileName, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        uttNameList_origin = []
        # context = np.array()
        for i in reader:
            if i[0] != '':
                uttNameList_origin.append(i[0])
        uttNameList = list(set(uttNameList_origin))
        uttNameList.sort(key=uttNameList_origin.index)   #不让集合化后索引失序
        uttNameList.remove("KEY")
        uttDict = {}
        for name in uttNameList:
            uttDict[name] = {}
            uttDict[name]['utterance'] = []
            uttDict[name]['sarcasm-label'] = ''
            uttDict[name]['sentiment-label'] = ''
            uttDict[name]['utt-number'] = ''

    with open(fileName, 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        for item in reader:
            if item[0] == 'KEY' or item[0] == '':
                continue
            uttDict[item[0]]['sarcasm-label'] = item[4]
            uttDict[item[0]]['sentiment-label'] = item[5]
            uttDict[item[0]]['utterance'].append(item[2])
            uttDict[item[0]]['utt-number'] = item[0]

    return uttDict, uttNameList


def processUttDict(uttDict):
    '''
    将生成的dict中的utterance的范围缩小到3 即两条上下文和一条目标语句
    如果对话语句不够的话在最前面补充一个空字符串
    如果对话语句多于3条的话，将多余的按照距离目标语句更远的上下文语句越先删除的顺序进行删除
    '''
    for key in uttDict.keys():
        lenUtt = len(uttDict[key]['utterance'])
        if lenUtt == 2:
            uttDict[key]['utterance'].insert(0, '')
            uttDict[key]['utterance'].insert(0, '')
        elif lenUtt > 4:
            uttDict[key]['utterance'] = uttDict[key]['utterance'][-4:]
        elif lenUtt == 3:
            uttDict[key]['utterance'].insert(0, '')
        else:
            continue
    # minLenUtt = min(len(uttDict[key]['utterance']) for key in uttDict.keys())
    # print('processUttDict-minLenUtt:', minLenUtt)
    return uttDict

def audioFeatureExtraction(datatye):
    if datatye == 'train':
        datafile = 'Dataset\Main\mustard-dataset-train.csv'
    elif datatye == 'dev':
        datafile = 'Dataset\Main\mustard-dataset-dev.csv'
    elif datatye == 'test':
        datafile = 'Dataset\Main\mustard-dataset-test.csv'

    uttDict, uttNameList = readcsv(datafile)
    uttDict = processUttDict(uttDict)
    uttList = list(uttDict.values())
    length = len(uttNameList)
    print("数据集的场景长度为：", length)
    """处理模态，提取特征过程"""
    
    frameNumbers = 4   #体现在下面的linspace
    avgPool = nn.AvgPool3d((270, 2, 2), stride=(43, 1, 1))
    avgPoolFlatten = nn.Flatten(start_dim=-2)
    # effNetModel = EfficientNet.from_pretrained('efficientnet-b0')
    vggishModel = torch.hub.load('harritaylor/torchvggish', 'vggish')
    input_dir = "D:\Project_Design\QEBSAN\QEBSAN\data_collect\\audio" + '\\' + datatye
    i = 1 #任务序号
    fail_audioList = []  #存储不合规的音频场景Name
    for this_audio in uttNameList:
        print('uttName:', this_audio)

        '''
        process the audio feature
        '''
        targetWavPath = input_dir+ '/target/' + this_audio + '.wav'            #音频文件按context和target分如不同文件夹
        contextWavPath = input_dir+ '/context/' + this_audio + '_c.wav'
        with torch.no_grad():
            try:   #2_149
                targetWav = vggishModel.forward(targetWavPath)
                tShape = targetWav.shape
                if len(tShape) == 1:
                    targetWav = torch.unsqueeze(targetWav, dim=0)
            except RuntimeError:
                fail_audioList.append(this_audio)
                continue
        with torch.no_grad():
            try:   #
                contextWav = vggishModel.forward(contextWavPath)
                tAudioFeature = torch.unsqueeze(torch.mean(targetWav, dim=0), dim=0)  # torch.Size([1, 128])
            except RuntimeError:
                fail_audioList.append(this_audio)
                continue               
            # print("是1,128吗", tAudioFeature.shape)
            '''
            把一个 上下文 音频特征去掉不能被3整除的部分，然后均等分为三份
            '''
            midShape = contextWav.shape[0]
            more = midShape % 3
            midShape -= more
            contextWav, _ = torch.split(contextWav, [midShape, more], dim=0)
            chunkList = torch.chunk(contextWav, chunks=3, dim=0)                #将输入张量（input）沿着指定维度（dim）均匀(上取整地分)的分割成特定数量的张量块（chunks），并返回元素为张量块的元组
            chunkList = [
                # 对每个chunk求平均值
                torch.unsqueeze(torch.mean(tmp, dim=0), dim=0)   #沿着某一维取平均值
                # 将tmp的维度扩展为1
                for tmp in chunkList
            ]
            cAudioFeature = torch.cat(chunkList, dim=0)                 #拼接
            cAudioFeature = torch.unsqueeze(cAudioFeature, dim=-2)
            print("上下文特征形状",cAudioFeature.shape)
            tAudioFeature = torch.unsqueeze(tAudioFeature, dim=-2)
            try:    #2_131出错
                tAudioFeature = np.squeeze(tAudioFeature.cpu().detach().numpy(), axis=1)  #squeeze删除单维条目（shape中1的条目），去除最外层维度
                cAudioFeature = np.squeeze(cAudioFeature.cpu().detach().numpy(), axis=1)
            except ValueError:
                fail_audioList.append(this_audio)          
                continue
            print("D:\Project_Design\QEBSAN\QEBSAN\\token\mustard\\audio_final/"+datatye+"/context/"+this_audio)
            print("D:\Project_Design\EBSAN\EBSAN\\token\mustard\\audio_final/"+datatye+"/target/"+this_audio)
            if os.path.exists("D:\Project_Design\EBSAN\EBSAN\\token\mustard\\audio_final/"+datatye+"/context/"+this_audio + ".txt"):
                pass
            else:
                np.savetxt("D:\Project_Design\EBSAN\EBSAN\\token\mustard\\audio_final/"+datatye+"/context/"+this_audio + ".txt", cAudioFeature, fmt="%f", delimiter=",") #txt特征文件保存到文件夹下

            if os.path.exists("D:\Project_Design\EBSAN\EBSAN\\token\mustard\\audio_final/"+datatye+"/target/"+this_audio + ".txt"):
                pass
            else:
               np.savetxt("D:\Project_Design\EBSAN\EBSAN\\token\mustard\\audio_final/"+datatye+"/target/"+this_audio + ".txt", tAudioFeature, fmt="%f", delimiter=",")
            
        '''
        process the audio feature
        '''
        progress_bar(i, length)
        i += 1
        time.sleep(0.1)
    print("语音特征提取失败的场景编号",fail_audioList)   #train: ['2_131', '2_149', '2_196', '2_202', '2_264', '2_306', '2_312', '2_323', '2_356', '2_373', '2_393']
    print("语音特征提取失败的场景数量",len(fail_audioList)) #11
                                                       #test: ['2_521', '2_522', '2_491', '2_481']
                                                       #4
                                                       #dev: ['2_536', '2_540', '2_549', '2_552', '2_58', '2_430', '2_434', '2_445']
                                                       #8
    return


if __name__ == '__main__':
    audioFeatureExtraction('dev')