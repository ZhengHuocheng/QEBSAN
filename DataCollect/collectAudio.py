# 此任务已完全完成
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
import cv2
import wave
import math
import time
import pyaudio
import sys
from moviepy.editor import VideoFileClip

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

def collect_AideoFromVideo(datatye):
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

    """
    从视频中提取帧图像
    """

    # 视频文件根路径
    video_root = "D:\Project_Design\mmsd_raw_data"

    # 创建保存音频的目录
    output_dir = "D:\Project_Design\QEBSAN\QEBSAN\data_collect\\audio" + '\\' + datatye
    os.makedirs(output_dir, exist_ok=True)
    # 音频参数
    # FORMAT = pyaudio.paInt16  #音频采样格式
    # CHANNELS = 1              #视频声道数
    # RATE = 16000              #采样率HZ
    # CHUNK = 1024              #缓冲区

    i = 1 #记录任务序号
    # 捕获帧的间隔
    # frame_interval = 10  # 每隔10帧捕获一次
    fail_audioList = []
    for this_video in uttNameList:
        # 打开视频文件
        context_video_path = video_root + '\\context_final\\' + this_video + '_c.mp4'
        target_video_path = video_root + '\\utterances_final\\' + this_video + '.mp4' 
        #打开音频输出文件   
        OUT_context_audio_filename = output_dir + "\\context" + "\\" + this_video + "_c.wav"
        OUT_target_audio_filename = output_dir + "\\target"  + "\\" + this_video + ".wav"
        #使用moviepy提取音频
        context_audio_clip = VideoFileClip(context_video_path).audio
        target_audio_clip = VideoFileClip(target_video_path).audio 
        context_audio_clip.write_audiofile(OUT_context_audio_filename, codec='pcm_s16le', fps=16000)
        target_audio_clip.write_audiofile(OUT_target_audio_filename, codec='pcm_s16le', fps=16000)
        # fail_audioList.append(this_video)


        context_audio_clip.close()
        target_audio_clip.close()

        #任务进度可视化 
        progress_bar(i, length)
        time.sleep(0.05)
        i += 1
    # print("{}数据集下无法提取的音频场景编号：".format(datatye), fail_audioList)

if __name__ == "__main__":
    collect_AideoFromVideo('dev')

            
