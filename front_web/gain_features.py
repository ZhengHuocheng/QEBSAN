import torch
# from torch import nn
from torch.utils.data import Dataset
from transformers import AlbertTokenizer, AlbertConfig, AlbertModel
import numpy as np
import csv
import os
from torch import nn

def readcsv(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        uttDict = {}
        uttDict['utterance']=[]
        for row in reader:
            uttDict['utterance'].append(row[1])
        uttDict['utterance'].pop(0)
    return uttDict  #{'utterance': ['How are you?', 'fine.']}
# print(readcsv('dialog_weGet\\au.csv')) 

def process_text(uttDict):
    #低4则补，多4则裁剪
    utt_len = len(uttDict['utterance'])
    if utt_len < 4:
        for i in range(4 - utt_len):
            uttDict['utterance'].insert(0, '')
    elif utt_len > 4:
        uttDict['utterance'] = uttDict['utterance'][-4:]
    return uttDict

def gain_txtFea(fileName):
    uttDict = process_text(readcsv(fileName))
    pretrained = 'albert-base-v2'
    tokenizer = AlbertTokenizer.from_pretrained(pretrained)
    albertModel = AlbertModel.from_pretrained(pretrained)    
    config = AlbertConfig.from_pretrained(pretrained)
    encoded_input = tokenizer(
        uttDict['utterance'],
        max_length=64,
        truncation=True,
        return_tensors='pt',
        padding='max_length')
    with torch.no_grad():
        contextsize = encoded_input['input_ids'].shape[0]
        # print("contextsize大小是",contextsize)
        textChunks = torch.chunk(encoded_input['input_ids'], contextsize, 0)
        text = [
            albertModel(
                torch.squeeze(item, dim=1))[0] for item in textChunks]
        # print("text",text[0].shape)
        text = [torch.unsqueeze(item, dim=1) for item in text]
        text = torch.cat(text, dim=1)
        text = torch.squeeze(text)
        saverootPath = 'out_txtFea'
        os.mkdir(saverootPath) 
        textFeaList = np.split(text, indices_or_sections=4, axis=0)
        textFeaList = [np.squeeze(item, axis=0) for item in textFeaList]
        # print("tensor是不是64,768?",textFeaList[0].shape)
        for num in range(4):
            savePath = saverootPath + '/'+str(num) + '.txt'
            print(savePath)
            np.savetxt(savePath, textFeaList[num] , fmt="%f", delimiter=",")
    return text

def gain_audioFea(cwavFile,twavFile):
    # uttDict = process_text(readcsv(csvFile))
    avgPool = nn.AvgPool3d((270, 2, 2), stride=(43, 1, 1))
    avgPoolFlatten = nn.Flatten(start_dim=-2)
    vggishModel = torch.hub.load('harritaylor/torchvggish', 'vggish')
    targetWavPath = twavFile
    contextWavPath = cwavFile
    with torch.no_grad():
        try:   #2_149
            targetWav = vggishModel.forward(targetWavPath)
            tShape = targetWav.shape
            if len(tShape) == 1:
                targetWav = torch.unsqueeze(targetWav, dim=0)
        except RuntimeError:
            print("RuntimeError：target音频特征提取失败")
    with torch.no_grad():
        try:   #
            contextWav = vggishModel.forward(contextWavPath)
            tAudioFeature = torch.unsqueeze(torch.mean(targetWav, dim=0), dim=0)  # torch.Size([1, 128])
        except RuntimeError:
            print("RuntimeError：context音频特征提取失败")

    print("target是1,128吗", tAudioFeature.shape)
    # 把一个 上下文 音频特征去掉不能被3整除的部分，然后均等分为三份
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
    # print("上下文特征形状",cAudioFeature.shape)
    tAudioFeature = torch.unsqueeze(tAudioFeature, dim=-2)
    try:    #2_131出错
        tAudioFeature = np.squeeze(tAudioFeature.cpu().detach().numpy(), axis=1)  #squeeze删除单维条目（shape中1的条目），去除最外层维度
        cAudioFeature = np.squeeze(cAudioFeature.cpu().detach().numpy(), axis=1)
    except ValueError:
        print("VakueError：音频特征提取失败")
    
    # 保存特征
    audio_saverootPath = 'out_audioFea'
    os.mkdir(audio_saverootPath) 
    np.savetxt(audio_saverootPath + "/au_context.txt", cAudioFeature, fmt="%f", delimiter=",")
    np.savetxt(audio_saverootPath + "/au_target.txt", tAudioFeature, fmt="%f", delimiter=",") 


# gain_txtFea('dialog_weGet\\au.csv')
# gain_audioFea("out_context.wav","out_target.wav")

