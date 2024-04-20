# QEBSAN(量子增强的语音、文本双模态情感分析模型)
 
1、Dataset:样本数据集合,原始视频由https://github.com/Himanshu-sudo/MUStARD-dataset 提供. <br>  下载链接:https://drive.google.com/file/d/1i9ixalVcXskA5_BkNnbR60sqJqvGyi6E/view  
2、DataCollect:从原始视频中提取音频等信息。  
3、FeatureExtract:从模态数据中提取文本、音频模态特征数据。  
4、ExpDataset:数据加载。  
5、ATmodel:搭建的基于注意力机制的语音文本双模态情感分析模型。  
6、qiskitModule:添加量子增强模块的双模态情感分析模型。(ATmodel输出特征\qiskitModule包含参数化量子电路，共同组成QEBSAN) 
7、Four_Noises_And_Expression_AND_Entanglement:对QEBSAN电路进行四种噪音下的鲁棒性检测，并量化评估可表达性和纠缠能力
8、front_web:利用streamlit搭建的web交互页面，其中：        
  Soundfile实现录音；                
  Pyannote实现人声分割；        
  调用本地模型进行预测；        
  页面设计由HTML\CSS\JS\Streamlit支持；  
9、model_data:包含预训练的ATmodel和qiskitmodel及其状态数据。共同组成QEBSAN模型。  
## 架构流程
![architecture](https://github.com/ZhengHuocheng/QEBSAN/blob/main/Picture/architecture.png)

## 可表达性  
![expressibility](https://github.com/ZhengHuocheng/QEBSAN/blob/main/Picture/expressibility.png)
## 纠缠能力
![entanglement](https://github.com/ZhengHuocheng/QEBSAN/blob/main/Picture/entanglement.png)
## 交互页面
![interface](https://github.com/ZhengHuocheng/QEBSAN/blob/main/Picture/interface2.png)



