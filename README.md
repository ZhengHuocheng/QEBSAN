#QEBSAN
量子融合策略下声音和文本的双模态情感分析模型
1、Dataset:样本数据集合,原始视频由https://github.com/Himanshu-sudo/MUStARD-dataset提供，
        下载链接:https://drive.google.com/file/d/1i9ixalVcXskA5_BkNnbR60sqJqvGyi6E/view  
2、DataCollect:从原始视频中提取音频等信息。  
3、FeatureExtract:从模态数据中提取文本、音频模态特征数据。  
4、ExpDataset:数据加载。  
5、ATmodel:搭建的语音文本双模态情感分析模型。  
6、qiskitModule:构建的包含量子神经网络的模型。(ATmodel输出特征\qiskitModule包含参数化量子电路，共同组成QEBSAN)
7、Four_Noises_And_Expression_AND_Entanglement:对QEBSAN电路进行四种噪音下的鲁棒性检测，并量化评估其可表达性和纠缠能力。  
8、front_web:利用streamlit搭建的web交互页面，其中包含录音、人声分割、模型预测功能的实现。  
9、model_data:包含预训练的ATmodel和qiskitmodel及其状态数据。共同组成QEBSAN模型。  


