'''
description: 
    1、interface GUI design
    2、record to output.wav
    3、CSV dialogue file to GUI dialogue text
Version: 
Author: Zheng Huocheng
Date: 2024-03-15 21:53:16
LastEditors: Zheng Huocheng
LastEditTime: 2024-04-09 00:53:25
'''
# =============================================================================模型预测
import sys
sys.path.append("D:\\Project_Design\\QFNN-main\\QFNN-main")
print(sys.path)
from model_predict import predict 
from audio_speaker import audio_to_text
from gain_features import gain_txtFea,gain_audioFea

def prd_sar_sent():
   audio_to_text()  #生成out_context.wav,out_target.wav和'dialog_WeGet\\au.csv'
   gain_txtFea('dialog_weGet\\au.csv')
   gain_audioFea("out_context.wav","out_target.wav")  #生成特征文件
   pdt_sent,pdt_sar = predict()
   return pdt_sent,pdt_sar




import streamlit as st
import queue
import sounddevice as sd
import soundfile as sf
import pandas as pd
# st.title('Audio Recorder')
#==============================================================================对话情景复现
st.markdown(
   """
    <style>
    .stApp {
        background-image: url('https://img.tukuppt.com/bg_grid/00/09/02/yamdxzGxw1.jpg!/fh/350');
        background-size: cover;
    }
    .user00-message {
        text-align: left;
        background-color: lightblue;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        display:inline-block;
    }

    .user01-message {
        text-align: right;
        background-color: lightgreen;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        display:inline-block;
        float:right;
    }
    </style>
    """
, unsafe_allow_html=True)

# 读取对话CSV文件
def read_dialogue_csv(file_path):
    df = pd.read_csv(file_path)
    return df
# 读取对话CSV文件
dialogue_df = read_dialogue_csv("dialog_weGet\\au.csv")


# 定义对话函数
def chatbot_dialogue():
    # 定义对话框布局
    chat_container = st.container(border=True)
    # 定义对话框布局
    left_expander = st.expander('duihua',expanded=True)
    with chat_container:
        for index, row in dialogue_df.iterrows():
            speaker = row['speaker']
            speech = row['speech']
            print(speech)
            # with left_expander:
                # # 定义左侧用户1的对话框
                # if speaker == "SPEAKER_00":
                #     st.text_area(speaker, value=speech, height=100)
                #     st.write_message
                # elif speaker == "SPEAKER_01":
                # # 在右侧用户2的对话框显示回复
                #     st.text_area(speaker, value=speech, height=100)
                # else:
                #     continue
            if speaker == "SPEAKER_00":
               st.markdown(f'<div class="user00-message">{speech}</div>', unsafe_allow_html=True)
            elif speaker == "SPEAKER_01":
               st.markdown(f'<div class="user01-message">{speech}</div>', unsafe_allow_html=True)
    # return 

# 运行对话页面
if __name__ == "__main__":
    chatbot_dialogue()

#===============================================================================录音按键功能
q = queue.Queue()
recording = False

def record():
  global recording
  
  if recording:
    stop_record()
  else:  
    start_record()
  
def start_record():
  global recording  
  samplerate = 16000
  
  with sf.SoundFile("output.wav", mode='w', samplerate=samplerate, channels=2) as file:  
    with sd.InputStream(samplerate=samplerate, channels=2, callback=callback):
      recording = True
      print("Recording started..")
      while recording:
          file.write(q.get())
          
def stop_record():
  global recording

  recording = False
  print("Recording stopped..")
  
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())



#===============================================================================页面底部浮动条功能
st.markdown("""
    <style>

        .slide-bar-container {
            position: fixed;
            bottom: 0; /* 初始位置在页面底部 */
            left: 0;
            width: 100%;
            height: 50px; /* 初始高度为50px */
            /*background-color: green;*/
            background-image: url(''https://e0.pxfuel.com/wallpapers/469/417/desktop-wallpaper-tessellation.jpg);
            background-size: cover;
            background-position: center;
            text-align: center;
            line-height: 50px;
            cursor: pointer;
            animation-name:expand ; /* 初始动画为展开 */
            animation-duration: 0.5s; /* 动画持续时间为0.5秒 */
            animation-fill-mode: forwards; /* 动画结束后保持最后一个关键帧的状态 */
        }
        .slide-bar-container.expanded {
            /*background-color: green;*/
            background-image: url(''https://e0.pxfuel.com/wallpapers/469/417/desktop-wallpaper-tessellation.jpg);
            background-size: cover;
            background-position: center;
            animation-name: collapse;  点击后的动画为收缩 */
             /* animation-name: collapse;点击后的动画为收缩 */
            z-index: 3;
        }
                
        .toggle-button {
            width: 125px;
            height: 125px;
            left:50%;
            border-radius: 50%;
            background-color: RED;
            position: absolute; /* 使用绝对定位 */
            transform: translate(-50%, -25%); /* 将按钮的圆心位置调整到div顶部的中心位置下方1/4的半径 */
            z-index: 3;
            float:right;
            background-image: url('https://img.zcool.cn/community/0126d55d920a20a801211d53187727.jpg@1280w_1l_2o_100sh.jpg');
            background-size: cover;
            background-position: center;
        }
        
        @keyframes expand {
            0% {
                height: 125;
            }
            100% {
                height: 50;
            }
        }
        
        @keyframes collapse {
            0% {
                height: 50;
            }
            100% {
                height: 125px;
            }
        }
    </style>
    <script>
        function toggleBar() {
            var barContainer = document.getElementById('slide-bar-container');
            barContainer.classList.toggle('expand');
        }
    </script>
""", unsafe_allow_html=True)

# 初始状态为未展开
if "expanded" not in st.session_state:
    st.session_state.expanded = False

# 点击按钮时切换Bar的展开与收缩状态
# 在侧边栏中添加带背景图片的container

with st.sidebar.container():
    if st.button("   Record   ", on_click=record):
        st.session_state.expanded = not st.session_state.expanded
    if st.button("   确    定   "):                                              #预测输出功能
        # 获取输出框的值并进行处理
        sent_Label,sar_Label = prd_sar_sent()
        st.sidebar.text_area("情感", value=sent_Label)                                  #之后在写prd_sar_sent()
        st.sidebar.text_area("讽刺", value=sar_Label)                                  #之后在写prd_sar_sent()
        # 在侧边栏中显示处理后的值
        # st.sidebar.write("处理后的值：", processed_value)

      #接下来将录音转为文本
# 更新Bar的样式和动画效果
if st.session_state.expanded:
    st.markdown(
        """
        <div id="slide-bar-container" class="slide-bar-container expanded">
            <div class="toggle-button" onclick="toggleBar()"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div id="slide-bar-container" class="slide-bar-container">
            <div class="toggle-button" onclick="toggleBar()"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
# ======================================================================================录音按键功能


#===================================================================================录音结束显示对话 
