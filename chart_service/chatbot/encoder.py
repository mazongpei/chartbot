"""
编码器

"""

from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence,pad_sequence
import torch
import torch.nn as nn
import config

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        #首先数据会进行embedding
        self.embedding=nn.Embedding(num_embeddings=len(config.chatbot_ws_input),
                                    embedding_dim=config.chatbot_embedding_dim,
                                    padding_idx=config.chatbot_ws_input.PAD
                                    )
        #参数解析：
        #   num_embeddings:词典中去重的词的总数量
        #   embedding_dim:使用多长的向量去表示一个词语
        #   padding_idx:告诉embedding层，当我们传入的是PAD这样一个数值的时候表示他是一个PAD_TAG

        #实例化GRU
        self.gru=nn.GRU(input_size=config.chatbot_embedding_dim,
                        num_layers=config.chatbot_encoder_num_layers,
                        hidden_size=config.chatbot_encoder_hidden_size,
                        batch_first=True
                        )
        #参数解析：
        #input_size:输入的形状
        #num_layers:有多少层
        #hidden_size:

    def forward(self, input,input_length):
        """

        :param input:input的形状[batch_size,max_len]
        :return:out,hidden
        """
        #print("input,input_length:",input.size(),input_length)
        embeded=self.embedding(input) #将input传入进去得到embedding之后的embeded embedded的形状[batch_size,max_len,embedding_dim]
        #print("embeded:",embeded.size())
        #embeded=pad_sequence([torch.tensor(dim) for dim in embeded], batch_first=True)#自己加的
        embeded=pack_padded_sequence(embeded,input_length,batch_first=True)#进行打包


        #交给GRU去编码得到out和hidden（隐藏状态）
        out,hidden=self.gru(embeded)

        #解包操作
        out,out_length=pad_packed_sequence(out,batch_first=True,padding_value=config.chatbot_ws_input.PAD)

        #hidden：[1*1,batch_size,hidden_size]第一个维度是1，因为只有一层，单项
        #out.shape:[batch_size,seq_len,hidden_size]
        return out,hidden












