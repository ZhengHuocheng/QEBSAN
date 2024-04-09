import torch
from torch.utils.data import Dataset
from transformers import AlbertTokenizer, AlbertConfig
import numpy as np
import csv
import math
import sys
import torch.nn.functional as F

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


# 针对Mustard数据集，定义 MustardDataset 类，继承自 Dataset 类
class MustardDataset(Dataset):

    # 初始化 MustardDataset 类
    def __init__(self, datatye):
        super().__init__()

        # 设置数据类型
        self.datatye = datatye
        """
        去除不符合的数据场景
        train:['2_131', '2_149', '2_196', '2_202', '2_264', '2_306', '2_312', '2_323', '2_356', '2_373', '2_393']
        test:  ['2_521', '2_522', '2_491', '2_481']
        dev:  ['2_536', '2_540', '2_549', '2_552', '2_58', '2_430', '2_434', '2_445']
        """
        # 根据数据类型设置数据文件
        if self.datatye == 'train':
            datafile = 'D:\Project_Design\M2Seq2Seq\Main\mustard-dataset-train.csv'
            need_remove = ['2_131', '2_149', '2_196', '2_202', '2_264', '2_306', '2_312', '2_323', '2_356', '2_373', '2_393']
        elif self.datatye == 'dev':
            datafile = 'D:\Project_Design\M2Seq2Seq\Main\mustard-dataset-dev.csv'
            need_remove = ['2_536', '2_540', '2_549', '2_552', '2_58', '2_430', '2_434', '2_445', '2_441']
        elif self.datatye == 'test':
            datafile = 'D:\Project_Design\M2Seq2Seq\Main\mustard-dataset-test.csv'
            need_remove = ['2_521', '2_522', '2_491', '2_481']
        # 读取数据文件，返回 uttDict 和 uttNameList
        uttDict, self.uttNameList = readcsv(datafile)
        # print("数据集{}的长度为:{}".format(self.datatye, len(self.uttNameList)))
        # 对 uttDict 进行处理
        uttDict = processUttDict(uttDict)
        # 去除采集失败特征的场境
        for scene in need_remove:
            self.uttNameList.remove(scene)
            del uttDict[scene]
        # 将 uttDict 转换为 uttList
        self.uttList = list(uttDict.values())
        print(self.datatye + "场景长度为：",len(uttDict.values()), len(self.uttNameList))

        # 设置帧数
        self.frameNumbers = 4
    
    def __getitem__(self, index):

        # 获取uttList中指定索引的uttName
        uttName = self.uttList[index]['utt-number']
        # print(uttName)
        # 获取文本特征
        textPath = \
            "D:\\Project_Design\\QFNN-main\\QFNN-main\\token\\mustard\\txt_final\\" + self.datatye + "\\" + uttName+"\\"
        textFea = [np.loadtxt(
                        textPath+str(num)+'.txt',
                        dtype=float , delimiter=",") for num in range(4)]
        # 将文本特征增加一个维度
        textFea = [np.expand_dims(item, axis=0) for item in textFea]
        # 将文本特征拼接起来
        textFea = np.concatenate(textFea, axis=0)
        # 将文本特征转换为tensor
        textFea = torch.tensor(textFea)   #4 64 768

        # txt_nan_mask = torch.isnan(textFea)
        # print("文本特征nan是",txt_nan_mask,"\n===============",textFea)

        # 获取图像特征
        imagePath = "D:\\Project_Design\\QFNN-main\\QFNN-main\\token\\mustard\\img\\" + self.datatye + "\\"+uttName+"\\"
        imageFea = [np.loadtxt(
                        imagePath+str(num)+'.txt',
                        dtype=float, delimiter=",") for num in range(4)]
        # print("一个图像的大小",imageFea[0].shape, imageFea[1].shape)
        # 将图像特征转换为24*56*56的矩阵
        # processList = []
        # # 采用双线性和上采样方法将特征转换为24，3136 = 24，56*56,  写的这个步骤存疑，可能导致BN层或之前的FLatten层得出NAN，之后再考察（没查出原因）
        # for item in imageFea:
        #     target_shape = (24,3136)
        #     # 计算特征数组的高度和宽度
        #     height, width = item.shape
        #     # 计算宽度的缩放比例
        #     scale_factor = target_shape[1] / width
        #     # 使用双线性插值进行上采样
        #     upsampled_feature_array = F.interpolate(torch.from_numpy(np.expand_dims(item, axis=(0, 1))),
        #                                scale_factor=(1, scale_factor),
        #                                mode='bilinear').squeeze()
        #     processList.append(upsampled_feature_array)
        # imageFea = processList
        # reshape为24,56,56
        # imageFea = [item.reshape(24,56,56) for item in imageFea]
        # 将图像特征增加一个维度
        imageFea = [np.expand_dims(item, axis=0) for item in imageFea]
        # 将图像特征拼接起来
        imageFea = np.concatenate(imageFea, axis=0)
        # 将图像特征转换为tensor
        imageFea = torch.tensor(imageFea)     #4 24 140
        # img_nan_mask = torch.isnan(imageFea)
        # print("图像特征NAN吗？",img_nan_mask,'\n===============',imageFea)


        # 获取音频特征
        context_audioPath = "D:\\Project_Design\\QFNN-main\\QFNN-main\\token\\mustard\\audio_final\\" + self.datatye + "\\context" + "\\" + uttName + ".txt"    #3 128
        target_audioPath = "D:\\Project_Design\\QFNN-main\\QFNN-main\\token\\mustard\\audio_final\\" + self.datatye+ "\\target"  + "\\" + uttName + ".txt"      #  128
        # auFea = [np.loadtxt(
        #                 audioPath+str(num)+'.txt',
        #                 dtype=float) for num in range(4)]
        context_auFea = np.loadtxt(context_audioPath, dtype=float, delimiter=",")
        context_auFea = np.expand_dims(context_auFea, axis=1)
        target_auFea = np.loadtxt(target_audioPath, dtype=float, delimiter=",")
        # print("target音频特征是",target_auFea.shape)
        # print("context音频特征是",context_auFea.shape)
        target_auFea = np.expand_dims(target_auFea, axis=0)
        target_auFea = np.expand_dims(target_auFea, axis=1)
        # print("target音频特征是",target_auFea.shape)
        # auFea = [context_auFea, target_auFea]
        # auFea = np.concatenate(auFea, axis=0)
        # 将音频特征增加一个维度
        # auFea = [np.expand_dims(item, axis=0) for item in auFea]
        # 将音频特征拼接起来

        # print("上下文音频特征是：", auFea.shape)
        # 将音频特征转换为tensor
        # auFea = torch.tensor(auFea)
        target_auFea = torch.tensor(target_auFea)
        context_auFea = torch.tensor(context_auFea)
        # print("target音频特征是",target_auFea.shape)
        # auFea = [context_auFea, target_auFea]
        # auFea = np.concatenate(auFea, axis=0)
        # 将音频特征增加一个维度
        # auFea = [np.expand_dims(item, axis=0) for item in auFea]
        # 将音频特征拼接起来

        # print("上下文音频特征是：", auFea.shape)
        # 将音频特征转换为tensor
        # auFea = torch.tensor(auFea)
        # 将形状转换为4,24,128   #导致产生NAN的最大可能应该就是这里的重复，应该会导致运算过程中的running_mean  running_var分母为0进而产生NAN
        # auFea = auFea.unsqueeze(1)
        # auFea = auFea.repeat(1, 24, 1)
        # au_nan_mask = torch.isnan(auFea)
        # print("auNAN吗：",au_nan_mask,"\n===============",auFea)
        # print(auFea.shape)


        # 获取sarcasmLabel
        sarcasmStr = self.uttList[index]['sarcasm-label']
        # 获取sentimentLabel
        sentimentStr = self.uttList[index]['sentiment-label']
        # 获取emotionLabel
        emotionStr = self.uttList[index]['emotion-label']
        # 判断sarcasmStr是否为True
        if sarcasmStr == 'True':
            sarcasmLabel = np.array([0, 1], dtype=np.int8)
        else:
            sarcasmLabel = np.array([1, 0], dtype=np.int8)

        # 判断sentimentStr是否为-1
        sentimentLabel = np.zeros(3, dtype=np.int8)
        if -1 == int(sentimentStr):
            sentimentLabel[0] = 1
        # 判断sentimentStr是否为0
        elif 0 == int(sentimentStr):
            sentimentLabel[1] = 1
        # 否则
        else:
            sentimentLabel[2] = 1
        # 判断emotionStr是否为0
        emotionLabel = np.zeros(9, dtype=np.int8)
        emotionLabel[int(emotionStr.split(',')[0])-1] = 1

        # 返回文本特征、图像特征、音频特征、sarcasmLabel、sentimentLabel、emotionLabel
        return [textFea, imageFea,
                target_auFea,context_auFea],\
            [sarcasmLabel, sentimentLabel] #, emotionLabel

    def __len__(self):
        # 返回uttList的长度
        return len(self.uttList)

# 定义MemotionDataset类，继承自Dataset类
class MemotionDataset(Dataset):
    # 初始化函数，定义文本数据集的路径
    def __init__(self):
        super().__init__()
        self.textDataset = 'memotion-dataset.csv'
        # 创建一个字典，用于存储文本数据
        uttDict = {}
        # 打开文本数据集，并使用csv模块读取数据
        with open(self.textDataset, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 跳过第一行
            a = 0
            for line in reader:
                # 如果第一列是number，则跳过
                if line[0] == 'number':
                    continue
                # 如果第二列是图片名称，则将其加入字典
                if line[1].split('.')[0] in self.imageNameList:
                    uttDict[line[0]] = {}
                    uttDict[line[0]]['utterence'] = line[3]
                    uttDict[line[0]]['imageName'] = line[1]
                    uttDict[line[0]]['sarcasm-label'] = line[5]
                    uttDict[line[0]]['sentiment-label'] = line[8]
        # 将字典赋值给self.uttDict
        self.uttDict = uttDict

    # 重写__getitem__函数，获取指定索引的文本数据
    def __getitem__(self, index):
        
        text = self.uttDict[index]['utterence']
        # 返回文本数据
        return 1
    
# if __name__ == '__main__':
#     train = MustardDataset("train")
#     print(train[0])
    