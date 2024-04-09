'''
description: 
Version: 
Author: Zheng Huocheng
Date: 2024-02-26 21:16:00
LastEditors: Zheng Huocheng
LastEditTime: 2024-04-08 17:40:46
'''
# 功能完全成功--------------------------------------------------------------
import torch
# from torch import nn
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
            uttDict[name]['emotion-label'] = ''
            uttDict[name]['utt-number'] = ''

    with open(fileName, 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        for item in reader:
            if item[0] == 'KEY' or item[0] == '':
                continue
            uttDict[item[0]]['sarcasm-label'] = item[4]
            uttDict[item[0]]['sentiment-label'] = item[5]
            uttDict[item[0]]['emotion-label'] = item[7]
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

class MustardDataset(Dataset):

    def __init__(self, datatye):
        super().__init__()
        '''
        albert 文本初始化部分
        '''
        if datatye == 'train':
            datafile = 'M2Seq2Seq\Main\mustard-dataset-train.csv'
        elif datatye == 'dev':
            datafile = 'M2Seq2Seq\Main\mustard-dataset-dev.csv'
        elif datatye == 'test':
            datafile = 'M2Seq2Seq\Main\mustard-dataset-test.csv'
        self.saverootPath = "D:/Project_Design/QFNN-main/QFNN-main/token/mustard/txt_final/"+datatye+"/"    #保存文本特征的路径，按数据集作用划分
        
        pretrained = 'albert-base-v2'
        self.tokenizer = AlbertTokenizer.from_pretrained(pretrained)
        self.albertModel = AlbertModel.from_pretrained(pretrained)
        self.config = AlbertConfig.from_pretrained(pretrained)
        uttDict, self.uttNameList = readcsv(datafile)
        print("该数据集长度为：", self.uttNameList)
        # print("下面打印前五项场景名称及其值")
        # count = 0
        # for item in self.uttNameList:
        #     if count < 5:
        #         print(item)
        #     else:
        #         break
        #     count += 1
        uttDict = processUttDict(uttDict)
        self.uttList = list(uttDict.values())
        # count = 0
        # for item in self.uttList:
        #     if count < 5:
        #         print(item)
        #     else:
        #         break
        #     count += 1
        # print("数据集长度为" + str(len(self.uttList)))   #500个场景
        def maxUttLen():
            '''
            寻找数据集中最长长度的文本
            '''
            maxlen = 0
            for item in self.uttList:
                encoded_input = self.tokenizer(
                    item['utterance'],
                    return_tensors='pt',
                    padding=True)
                output = self.albertModel(**encoded_input)
                shape2 = output[0].shape[1]
                if maxlen < shape2:
                    maxlen = shape2
            print(maxlen)
        # maxUttLen()   #测试后得到下面getitem方法中的最长文本长度为64
    
    def __getitem__(self, index):
        # albert
        '''
        将所有句子的嵌入长度设置为64，长度不足的进行补充，长度过长的进行剪裁
        '''
        utts = self.uttList[index]['utterance']    #utts为当前场景下的对话列表
        print(self.uttList[index]['utt-number'],end='\n')
        print(utts,end='\n')
        encoded_input = self.tokenizer(
            utts,
            max_length=64,
            truncation=True,
            return_tensors='pt',
            padding='max_length')
        with torch.no_grad():
            contextsize = encoded_input['input_ids'].shape[0]
            # print("contextsize大小是",contextsize)
            textChunks = \
                torch.chunk(encoded_input['input_ids'], contextsize, 0)
            # print("chunk",textChunks[0].shape)
            text = [
                self.albertModel(
                    torch.squeeze(item, dim=1))[0] for item in textChunks]
            # print("text",text[0].shape)
            text = [
                torch.unsqueeze(item, dim=1) for item in text]
            text = torch.cat(text, dim=1)
            text = torch.squeeze(text)
            # print(text.shape)
        # 顺带保存文本的特征
            os.mkdir(self.saverootPath) 
            # os.mkdir(self.saverootPath + self.uttList[index]['utt-number'])
            textFeaList = np.split(text, indices_or_sections=4, axis=0)  #(1, , )
            textFeaList = [np.squeeze(item, axis=0) for item in textFeaList] #4*( , )
            # print("tensor是不是64,768?",textFeaList[0].shape)
            for num in range(4):
                savePath = self.saverootPath + self.uttList[index]['utt-number'] + '/'+str(num) + '.txt'
                print(savePath)
                np.savetxt(savePath, textFeaList[num] , fmt="%f", delimiter=",")
        # torch.Size([4, 64, 768])
        # albert 几可能是文本特征
        return text

    

if __name__ == "__main__":
    data_train = MustardDataset('dev')
    data = data_train[0]
    # i = 0
    # while i < 500:
    #     temp = data_train[i]
    #     time.sleep(1)
    #     i += 1
    print("################################文本特征提取完毕###########################")