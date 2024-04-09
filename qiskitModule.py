'''
description: 
Version: 
Author: Zheng Huocheng
Date: 2024-03-09 01:23:59
LastEditors: Zheng Huocheng
LastEditTime: 2024-04-08 18:15:46
'''
import argparse
import datetime
import logging
import os
import random
import numpy as np
from numpy import double
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit import transpile
from qiskit.circuit import Instruction
# from qiskit.circuit.library import
import torch
from torch import float32
from torch.utils.data import DataLoader, dataset
from torch.autograd import Variable
from ExpDataset import MustardDataset
from ATmodel import HyperModel
# from m2model_train import M2Model



class Q_Model(torch.nn.Module):
  def __init__(self,bszi):
    super(Q_Model, self).__init__()
    # self.encoder= QuantumCircuit(5,5)
    # for i in range(5):
    #   self.encoder.rx(np.pi/2,i)
    #
    # self.sar_L=nn.Linear(2,2)  #3   3 -》 2 2
    # self.sen_L=nn.Linear(3,3)
    #
    # self.measure = QuantumCircuit(5,5)
    # for i in range(5):
    #   self.measure.measure(i,i)
    #
    # self.re_sar_L=nn.Linear(2,2)
    # self.re_sen_L=nn.Linear(3,3)
    # 定义线路和量子注册
    self.n_qubits = 5  #量子位
    self.circuit = QuantumCircuit(self.n_qubits, name="EncDec", global_phase=0)
    self.register = ClassicalRegister(self.n_qubits)  #经典位
    self.qc = QuantumCircuit(5, 5)

    # 定义线性模块
    self.sar_L = nn.Linear(2, 2)
    self.sen_L = nn.Linear(3, 3)
    self.re_sar_L = nn.Linear(2, 2)
    self.re_sen_L = nn.Linear(3, 3)

  def forward(self,sa,sen):
    bsz = sa.shape[0]  # 获取批尺寸
    # 建立批次数量的量子电路
    circuit_list = [QuantumCircuit(self.n_qubits, name="EncDec" + str(i)) for i in range(bsz)]
    creg = ClassicalRegister(self.n_qubits, name='creg')  #经典位
    for i in range(bsz):
        circuit = circuit_list[i]  # 使用不同的线路
        circuit.add_register(creg)
        sa_batch = sa[i].cpu().detach().numpy()
        # print("sa_batch",sa_batch)
        sen_batch = sen[i].cpu().detach().numpy()
        # print("sen_batch",sen_batch)
        # 输入编码
        for j in range(3): #sent
            circuit.rx(sen_batch[j], j)

        for j in range(2): #sar
            circuit.rx(sa_batch[j], j + 3)
        for j in range(3):  # sent
            circuit.rz(sen_batch[j], j)

        for j in range(2):  # sar
            circuit.rz(sa_batch[j], j + 3)
        # 耦合
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 0)

        circuit.cx(3, 4)
        circuit.cx(4, 3)
        for j in range(3):  # sent
            circuit.ry(sen_batch[j], j)

        for j in range(2):  # sar
            circuit.ry(sa_batch[j], j + 3)
        # Add measurement
        for j in range(self.n_qubits):
            circuit.measure(circuit.qubits[j], creg[j])

    # circuit_list[0].draw()
    # 执行并测量
    result_measurement = []
    result_count_measurement = []
    for i in range(bsz):
        result = Aer.get_backend('statevector_simulator').run(circuit_list[i]).result()
        result_measurement.append(result.get_statevector())
        result_count_measurement.append(result.get_counts())
    # print(result_measurement)
    # 提取输出
    # print(result_count_measurement) #[{'01101': 1}, {'01101': 1}, {'01111': 1}, {'00001': 1}, {'00101': 1}, {'00111': 1}, {'01001': 1}, {'01101': 1}, {'01111': 1}, {'00111': 1}, {'00111': 1}, {'00101': 1}, {'00111': 1}, {'00101': 1}, {'01101': 1}, {'00101': 1}, {'01111': 1}, {'01011': 1}, {'00111': 1}, {'00101': 1}, {'01111': 1}, {'00111': 1}, {'01111': 1}, {'01111': 1}, {'00101': 1}, {'01101': 1}, {'00111': 1}, {'00101': 1}, {'00101': 1}, {'01111': 1}, {'00111': 1}, {'01011': 1}]
    # 取5位测量结果
    pre_output = []
    for i in range(bsz):
        pre_output.append(list(result_count_measurement[i].keys()))
    # print(pre_output)
    # sen 前三位
    sen_strings = [string[0][0:3] for string in pre_output]
    # print('sen',sen_strings)
    # sar 后两位
    sar_strings = [string[0][3:5] for string in pre_output]
    # print('sar',sar_strings)
    # 划分出每一位
    sen_final = []   #sen处理
    for string in sen_strings:
        sen_final.extend(list(map(double, string)))
    sen_arr = np.array(sen_final).reshape(-1, len(sen_strings[0]))
    # print(sen_arr)

    sar_final = []   #sar处理
    for string in sar_strings:
        sar_final.extend(list(map(double, string)))
    sar_arr = np.array(sar_final).reshape(-1, len(sar_strings[0]))
    # print(sar_arr)
    # 转化为tensor
    sen_tensor = torch.from_numpy(sen_arr)
    sar_tensor = torch.from_numpy(sar_arr)
    # sar_tensor.to()
    # sen_tensor.to()
    sen = sen_tensor.to('cuda:0')
    sar = sar_tensor.to('cuda:0')
    # 提取测量结果
    # print(sen_tensor,sen_tensor.shape)
    # 解码和返回
    # sar = self.re_sar_L(sar_tensor)
    # sen = self.re_sen_L(sen_tensor)

    # print(sar,sen)

    return sar,sen


def trainEval():

    batchsize = 32
    epochs = 101

    all_epochs=280
    need_train_models_epoch = [20, 40, 60, 80, 100]
    # now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    root_stateSavePATH = f'Quantum_result/state/qState'
    root_modelSavePATH = f'Quantum_result/state/qModel'

    data = MustardDataset(datatye='train')
    valData = MustardDataset(datatye='dev')

    
    data_loader = DataLoader(
        data,
        batch_size=batchsize,
        shuffle=True,
        pin_memory=True
        )
    val_loader = DataLoader(
        valData,
        batch_size=9,
        shuffle=True,
        pin_memory=True,
        )
    # 与训练队的分类模型
    class_model=torch.load(f'D:\\Project_Design/result/state/model80.pt')
    class_model.eval()

    model = Q_Model(bszi=batchsize).cuda()
   
    lossFun = nn.CrossEntropyLoss().cuda()   #交叉熵损失函数
    lossFun_val = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.015)    #优化器
    total = sum([param.nelement() for param in model.parameters()])
    print(f"Number of parameter: {total}")

    train_step = 0
    # os.mkdir(f'Quantum_result/state/{all_epochs}_Epochs_{now}')

    for _ in range(epochs):
        logger.info('epoch:'+str(_))
        print("epoch:",_)
        for batch in data_loader:
            train_step += 1
           
            model.train()
            textInput = batch[0][0].cuda().to(torch.float32)
            # imageInput = batch[0][1].to(torch.float32).cuda()
            # audioInput = batch[0][2].to(torch.float32).cuda()
            tWavPath = batch[0][2].to(torch.float32).cuda()
            cWavPath = batch[0][3].to(torch.float32).cuda()
          
            sarLabel = batch[1][0].to(torch.float32).cuda()
            sentLabel = batch[1][1].to(torch.float32).cuda()

            with torch.no_grad():
               sar, sent= class_model.Qin(textInput, tWavPath, cWavPath)

            if sar is None and sent is None:
                continue
            else:
                print(sar.shape, sent.shape)
                sar,sent=model(sar,sent)

                sarArgmax = torch.argmax(sarLabel, dim=-1)
                sentArgmax = torch.argmax(sentLabel, dim=-1)
                # emoArgmax = torch.argmax(emoLabel, dim=-1)
                # logger.info(f'sar:{sarArgmax} sent:{sentArgmax} emo:{emoArgmax}\n')
                loss1 = lossFun(sar, sarArgmax)
                loss2 = lossFun(sent, sentArgmax)
                # loss3 = lossFun(emo, emoArgmax)
                loss = (loss1 + loss2 )/2

                loss.requires_grad_(True)

                logger.info('loss1:%f loss2:%f loss:%f\n'
                            % (loss1.item(),
                                loss2.item(),
                                loss.item()))

                if train_step%(epochs-1) == 0:
                    label_sar = np.argmax(
                        sarLabel.cpu().detach().numpy(), axis=-1)
                    label_sent = np.argmax(
                        sentLabel.cpu().detach().numpy(), axis=-1)
                    # label_emo = np.argmax(
                    #     emoLabel.cpu().detach().numpy(), axis=-1)
                    pred_sar = np.argmax(
                        sar.cpu().detach().numpy(), axis=1)
                    pred_sent = np.argmax(
                        sent.cpu().detach().numpy(), axis=1)

                    sar_f1 = f1_score(label_sar, pred_sar, average='micro')
                    sent_f1 = f1_score(label_sent, pred_sent, average='micro')
                    # emo_f1 = f1_score(label_emo, pred_emo, average='micro')
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
                            # imageInput = batch[0][1].cuda().to(torch.float32)
                            # audioInput = batch[0][2].cuda().to(torch.float32)
                            tWavPath = batch[0][2].to(torch.float32).cuda()
                            cWavPath = batch[0][3].to(torch.float32).cuda()

                            sarLabel = batch[1][0].to(torch.float32).cuda()
                            sentLabel = batch[1][1].to(torch.float32).cuda()

                            with torch.no_grad():
                                sar, sent=class_model.Qin(textInput, tWavPath, cWavPath)

                            if sar is None and sent is None:
                                continue
                            else:
                                sar,sent=model(sar,sent)

                                sarArgmax = torch.argmax(sarLabel, dim=-1)
                                sentArgmax = torch.argmax(sentLabel, dim=-1)

                                loss1_val = lossFun_val(sar, sarArgmax)
                                loss2_val = lossFun_val(sent, sentArgmax)

                                loss_val = (loss1_val + loss2_val)/2

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
                            np.array(outputsar),dtype=np.int64)
                        outputsent = np.concatenate(
                            np.array(outputsent),dtype=np.int64)

                        tarsar = np.concatenate(
                            np.array(tarsar),dtype=np.int64)
                        tarsent = np.concatenate(
                            np.array(tarsent),dtype=np.int64)



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
                optimizer.step()
        if _ in need_train_models_epoch:
            stateSavePATH = root_stateSavePATH + str(_) + ".pt"
            modelSavePATH = root_modelSavePATH + str(_) + ".pt"
            torch.save(model.state_dict(), stateSavePATH)  # 减少了一个缩进，下一行也是
            torch.save(model, modelSavePATH)


ClassModelPath = f'D:\\Project_Design/result/state/model80.pt'
def testModel(modelPATH):
    testData = MustardDataset(datatye='test')
    batchsize = 4
    data_loader = DataLoader(
        testData,
        batch_size=batchsize,
        shuffle=True,
        pin_memory=True
        )
    classmodel=torch.load(ClassModelPath)
    
    model=torch.load(modelPATH)
    model.eval()
    classmodel.eval()
    with torch.no_grad():
        # outputsar, outputsent, outputemo = [], [], []
        outputsar, outputsent = [], []
        tarsar, tarsent = [], []
        for batch in data_loader:
            textInput = batch[0][0].cuda().to(torch.float32)
            # imageInput = batch[0][1].cuda().to(torch.float32)
            # wavInput = batch[0][2].cuda().to(torch.float32)
            tWavPath = batch[0][2].to(torch.float32).cuda()
            cWavPath = batch[0][3].to(torch.float32).cuda()
            
            sarLabel = batch[1][0].to(torch.float32).cuda()
            sentLabel = batch[1][1].to(torch.float32).cuda()
            # emoLabel = batch[1][2].to(torch.float32).cuda()
            # with torch.no_grad():
            #     sar, sent=classmodel(textInput, wavInput)
            sar, sent = classmodel(textInput, tWavPath, cWavPath)
            if sar is None and sent is None:
                continue
            else:

                # inq=torch.cat((sar,sent),dim=1).cuda()
                sar,sent=model(sar,sent)
                # print(sar,sent)
                # 对sar‘0 0 ’，‘1 1’和sent '0 0 0 ', ' 1  1 1'情况，按相同概率进行选择
                def check(lst):
                    if torch.sum(lst) != 1:
                        # 生成随机索引
                        # print("函数内",lst)
                        index = random.randint(0, lst.size(0) - 1)
                        # print("函数内index",index)
                        # 创建全零张量
                        encoded_tensor = torch.zeros_like(lst)

                        # 将随机索引处的元素设为1
                        encoded_tensor[index] = 1
                        # print("函数内",encoded_tensor)
                        return encoded_tensor
                    else:
                        return lst
                sar = [check(sarlst) for sarlst in sar]
                sent = [check(senlst) for senlst in sent]
                # 将列表中的多个张量堆叠为一个张量
                sar = torch.stack(sar)
                sent = torch.stack(sent)
                print(sar,sent)
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
        

        sar_f1 = f1_score(
            tarsar, outputsar, average='micro')
        sent_f1 = f1_score(
            tarsent, outputsent, average='micro')
       
        sar_acc = accuracy_score(
            tarsar, outputsar)
        sent_acc = accuracy_score(
            tarsent, outputsent)
    
        # print('test tarsar:', tarsar)
        # print('test outputsar:', outputsar)
        # print('test tarsent:', tarsent)
        # print('test outputsent:', outputsent)
        logger.info(('test-result sar-f1:%f sent-f1:%f' +
                    'sar-acc:%f sent-acc:%f \n')
                    % (sar_f1, sent_f1,
                    sar_acc, sent_acc))

  

if __name__ == "__main__":
    logging.basicConfig(filemode='w')
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(
        "result/log/QiskitATepoch100_test.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("**********Start print log**********")
    

    # trainEval()
    testModel('Quantum_result/state/qModel100.pt')