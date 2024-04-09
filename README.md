# QFNN
用于情感分析和讽刺检测的量子模糊神经网络模型。
Dataset:样本数据集合,原始视频由https://github.com/Himanshu-sudo/MUStARD-dataset提供，
        下载链接:https://drive.google.com/file/d/1i9ixalVcXskA5_BkNnbR60sqJqvGyi6E/view
DataCollect:从原始视频中提取音频、图像等信息。
FeatureExtract:从模态数据中提取多模态特征。
ExpDataset:数据加载器。
ATmodel:搭建的语音文本双模态情感分析模型。
qiskitModule:构建的包含量子神经网络的QFNN模型。
Four_Noises_And_Expression_AND_Entanglement:对QFNN电路进行鲁棒性检测，并量化评估可表达性和纠缠能力。
front_web:利用streamlit搭建的web交互页面，其中包含录音、人声分割、模型预测功能的实现。
model_data:包含预训练的ATmodel和qiskitmodel及其状态数据。共同组成QFNN模型。


