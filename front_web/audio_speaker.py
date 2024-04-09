'''
description: 
  1、提取音频对话段，并转为csv文件保存后4次到dialog_WeGet中
Version: 
Author: Zheng Huocheng
Date: 2024-03-14 20:31:38
LastEditors: Zheng Huocheng
LastEditTime: 2024-04-08 17:09:31
'''
import pyannote.audio as pa
from pyannote.audio import Pipeline
import speech_recognition as sr
import soundfile as sf
import numpy as np
import csv

def audio_to_text():
  # 加载模型
  pipeline = Pipeline.from_pretrained(
    "speaker-diarization-3.1\config.yaml",
    use_auth_token="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")

  # 人声分割
  diarization = pipeline("au.wav")

  # 对话转rttm格式
  with open("output.rttm", "w") as rttm:
      diarization.write_rttm(rttm)

  # ========================================================获得RTTM文件
  # # 获取音频文件
  audio,samplerate = sf.read("au.wav")
  # 音频采样率
  SAMPLE_RATE = samplerate
  # 提取声纹：使用的音频长度
  EMBEDDING_DURATION = 1.0
  # 提取声纹：只使用达到一定长度的音频
  EMBEDDING_DURATION_THRESHOLD = EMBEDDING_DURATION + 0.2
  # 提取声纹：读取音频帧数量(torchaaudio)
  EMBEDDING_NUM_FRAMES = int(SAMPLE_RATE * EMBEDDING_DURATION)
  # 提取声纹：验证声纹达标阈值
  EMBENDING_THRESHOLD = 0.25

  audio_segments = []
  speaker_List = []
  # 采集speakers
  with open('au.rttm') as f:
    reader = csv.reader(f)
    for row in reader:
      vals = row[0].split(' ')
      # print(vals)
      speaker = vals[7] 
      speaker_List.append(speaker)
  print(speaker_List)

  tracks = list(diarization.itertracks())
  all_start=[]
  all_end=[]
  # print(tracks)  #[(<Segment(0.82343, 2.36842)>, 'A'), (<Segment(3.16638, 4.71138)>, 'B')]
  for idx in range(0, len(tracks)):
      track, speaker = tracks[idx]
      start = max(track.start, tracks[idx - 1]
                  [0].end) if idx != 0 else track.start   #音频段开始
      end = min(track.end, tracks[idx + 1]
                [0].start) if idx != len(tracks) - 1 else track.end   #音频段结束
      all_start.append(start)
      all_end.append(end)
      # print("satart",start,"end",end)
      # segment = diarization(start=start, end=end)

      # print(
      #     f"{speaker} {track} ==>> start={start:.3f} stop={end:.3f} duration={(end - start):.3f}")  #B [ 00:00:03.166 -->  00:00:04.711] ==>> start=3.166 stop=4.711 duration=1.545
      start_sample = int(start * samplerate)
      end_sample = int(end * samplerate) 
      speech = audio[start_sample:end_sample]
      sf.write("segment{}_{}.wav".format(idx,speaker_List[idx]), speech, samplerate)


  for i in range(len(tracks)):
    audio_segments.append("segment{}_{}.wav".format(i,speaker_List[i]))
  # print(audio_segments)

  # 语音识别
  r = sr.Recognizer()
  # #=======================================================形成对话语音文件
  # # print(speaker_List)
  # # print(audio_segments)
  # #=======================================================音频转文字，形成文本对话
  r = sr.Recognizer()
  # #optional
  # r.energy_threshold = 300
  texts = []

  def startConvertion(path, lang = 'en-US'): 
      with sr.AudioFile(path) as source:
          #print('Fetching File')
          r.adjust_for_ambient_noise(source,duration=0)
          audio_file = r.record(source,offset=0)
          text = r.recognize_google(audio_file, language=lang)
          return text
    
  for i in range(len(tracks)):
    segPath = audio_segments[i]
    speaker = speaker_List[i]
    # print(segPath,speaker)
    try:
      text = startConvertion(segPath)
      texts.append({
        'speaker':speaker,
        'speech':text
      })
    except sr.exceptions.UnknownValueError:
      continue
    
  # print(texts)  #[{'speaker': 'SPEAKER_00', 'speech': 'no no no no'}]

  # 将列表写入CSV文件
  def write_list_to_csv(data_list, file_path):
      with open(file_path, mode='w', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(['speaker', 'speech'])  # 写入列标题
          for data in data_list:
              writer.writerow([data['speaker'], data['speech']])

  # 调用函数将列表写入CSV文件
  write_list_to_csv(texts, 'dialog_WeGet\\au.csv')

  # print(all_start)
  # print(all_end)

  # 将wav拼接为context wav和target wav

  def wav_concatenate(input_files,output_file):
      all_wav = []
      for wav in input_files:
        signal_data,sr = sf.read(wav)
        all_wav.append(signal_data)
      
      concated_data = np.concatenate(all_wav)
      sf.write(output_file, concated_data, SAMPLE_RATE)

  context_list = audio_segments[0:len(tracks)-1]
  target_one = audio_segments[len(tracks)-1]

  wav_concatenate(context_list,'out_context.wav')

  import shutil

  shutil.copy(target_one, 'out_target.wav')