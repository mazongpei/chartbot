# Coding by 马宗沛
# Datatime:2022/3/10 19:59
# Filename:attention.py
# Toolby: PyCharm

import config
import torch
import  torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,method="general"):
        super(Attention, self).__init__()
        assert method in ["dot", "general", "concat"], "method error"
        self.method=method
##实例化参数
        if self.method=="dot":
            pass
        if self.method=="general":
            self.Wa=nn.Linear(config.chatbot_encoder_hidden_size,config.chatbot_decoder_hidden_size,bias=False)#设置没有偏置
        elif self.method=="concat":
            self.Wa=nn.Linear(config.chatbot_encoder_hidden_size+config.chatbot_decoder_hidden_size,config.chatbot_decoder_hidden_size,bias=False)
            self.Va=nn.Linear(config.chatbot_decoder_hidden_size,1)


    def forward(self, hidden_state,encoder_outputs):
        """

        :param hidden_state: [num_layers=1,batch_size,decoder_hidden_size]
        :param encoder_outputs: [batch_size,seq_len,encoder_hidden_sizer]
        :return:
        """
        #1.dot:
        if self.method=="dot":
            #对数据进行变形
            hidden_state=hidden_state[-1,:,:].unsqueeze(-1)  #[batch_size,hidden_size,1]
            attention=encoder_outputs.bmm(hidden_state).squeeze(-1)#encoder_outputs * hidden_state   [batch_size,seq_len]
            #进行softmax计算
            attention_weight=F.softmax(attention,dim=-1)#[batch_size,seq_len]
        #2.general
        elif self.method=="general":
            #print("hidden_state",hidden_state.size())
            encoder_outputs=self.Wa(encoder_outputs)#[batch_size,decoder_hidden_size]
            hidden_state=hidden_state[-1,:,:].unsqueeze(-1) #[batch_size,decoder_hidden_size,1]
            attention=encoder_outputs.bmm(hidden_state).squeeze(-1)#encoder_outputs * hidden_state   [batch_size,seq_len]
            attention_weight=F.softmax(attention,dim=-1)#[batch_size,seq_len]
        #3.concat:
        elif self.method=="concat":
            hidden_state=hidden_state[-1,:,:].squeeze(0) #[batch_size,hidden_size]
            hidden_state=hidden_state.repeat(1,encoder_outputs.size(1),1)#[batch_size,seq_len,decoder_hidden_size]
            concated=torch.cat([hidden_state,encoder_outputs],dim=-1)#[batch_size,decoder_hidden_size+encoder_hidden_size]

            batch_size=encoder_outputs.size(0)
            encoder_seq_len=encoder_outputs.size(1)
            attention_weight=self.Va(F.tanh(self.Wa(concated.view((batch_size*encoder_seq_len,-1))))).sequeeze(-1)# [batch_size*seq_len]
            attention_weight = F.softmax(attention_weight.view(batch_size,encoder_seq_len),dim=-1)  # [batch_size,seq_len]

        return attention_weight
