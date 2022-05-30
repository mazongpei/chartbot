# Coding by 马宗沛
# Datatime:2022/3/11 17:36
# Filename:eval.py
# Toolby: PyCharm
"""模型的评估"""
import numpy as np
from chatbot.seq2seq  import Seq2Seq
import config
import torch
from lib import cut

import pyttsx3


class RobotSay():

    def __init__(self):
        # 初始化语音
        self.engine = pyttsx3.init()  # 初始化语音库

        # 设置语速
        self.rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', self.rate - 50)

    def say(self, msg):
        # 输出语音
        self.engine.say(msg)  # 合成语音
        self.engine.runAndWait()





def eval(by_word=True):
    seq2seq=Seq2Seq()
    seq2seq=seq2seq.to(config.device)
    seq2seq.load_state_dict(torch.load(config.chatbot_model_save_path))


    while True:
        _input=input("请输入：")
        _input=cut(_input,by_word=by_word)
        input_length = torch.LongTensor([len(_input)]).to(config.device)# if len(_input)>config.chatbot_input_max_len else config.chatbot_input_max_len
        _input=torch.LongTensor([config.chatbot_ws_input.transform(_input,max_len=config.chatbot_input_max_len)]).to(config.device)

        indices=np.array(seq2seq.evaluate(_input,input_length)).flatten()
        output="".join(config.chatbot_ws_target.inverse_transform(indices))
        print("answer: ",output)
