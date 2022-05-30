"""实现解码器"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import config
import random
from chatbot.attention import  Attention
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding=nn.Embedding(num_embeddings=len(config.chatbot_ws_target),
                                    embedding_dim=config.chatbot_embedding_dim,
                                    padding_idx=config.chatbot_ws_target.PAD)
        self.gru=nn.GRU(input_size=config.chatbot_embedding_dim,
                        hidden_size=config.chatbot_decoder_hidden_size,
                        num_layers=config.chatbot_decoder_num_layer,
                        batch_first=True
                        )
        self.fc=nn.Linear(config.chatbot_decoder_hidden_size,len(config.chatbot_ws_target))#全连接层
        self.atten=Attention()#初始化attention
        self.Wa=nn.Linear(config.chatbot_decoder_hidden_size+config.chatbot_encoder_hidden_size,config.chatbot_decoder_hidden_size,bias=False)




    def forward(self, target,encoder_hidden,encoder_outputs):#target：目标值
        #print("target.size:",target.size())
        #1.获取encoder的输出，作为decoder第一次的隐藏状态(hidden_size)
        decoder_hidden=encoder_hidden
        batch_size=target.size(0)
        #2.准备decoder第一个时间步的输入 形状为[batch_size,1]全为SOS
        decoder_input=torch.LongTensor(torch.ones([batch_size,1],dtype=torch.int64)*config.chatbot_ws_target.SOS).to(config.device)
        #3.在第一个时间步上进行计算，得到第一个时间步的输出，hidden_state

        #4.把前一个时间步的输出进行计算，得到第一个时间步的最后的输出结果
        #5.把前一次的hidden_state作为当前时间步的隐藏状态(hidden_state)的输入，把前一次时间步的输出作为当前时间步的输入
        #6.循环4-5步骤

        # 保存预测结果
        decoder_outputs=torch.zeros([batch_size, config.chatbot_target_max_len + 1,len(config.chatbot_ws_target)]).to(config.device)

        if random.random() > config.chatbot_teacher_forcing_ratio:  # 使用teacher_forcing机制，加速收敛
            #decoder_input = target[t]  # [batch_size,1]
            for t in range(config.chatbot_target_max_len + 1):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input,
                                                                     decoder_hidden,encoder_outputs)  # 经过一次前向计算后，得到一次的输出，一次的隐藏状态
                # 保存decoder_output_t到decoder_outputs中
                # 切片
                decoder_outputs[:,t,:] = decoder_output_t
                decoder_input=target[:,t].unsqueeze(-1)#[batch_size,1]
                #print("user_teacher_decoder_input size:",decoder_input.size())
        else:#不使用teacher_forcing
            for t in range(config.chatbot_target_max_len + 1):
                decoder_output_t, decoder_hidden = self.forward_step(decoder_input,
                                                                     decoder_hidden,
                                                                     encoder_outputs
                                                                     )  # 经过一次前向计算后，得到一次的输出，一次的隐藏状态
                # 保存decoder_output_t到decoder_outputs中
                # 切片
                decoder_outputs[:,t,:] = decoder_output_t
                value, index = torch.topk(decoder_output_t, 1)  # 在最后一个维度上进行操作
                decoder_input = index



        return decoder_outputs,decoder_hidden
    #返回最后输出，最后的hidden





    def forward_step(self,decoder_input,decoder_hidden,encoder_outputs):#向前计算一步
        """
        计算每个时间步上的结果
        :param decoder_input: [batch_size,1]
        :param decoder_hidden: [1,batch_size,hidden_size]
        :return:
        """
        #print("decoder_input:",decoder_input.size())
        decoder_input_embeded=self.embedding(decoder_input)#对解码器的输入进行embeding操作
        #print('decoder_input_embeded:',decoder_input_embeded.size())
        #decoder_input_embeded.shape=[batch_size,1,embedding_dim]

        #out.shape:[batch_size,1,hidden_size]
        #decoder_hidden.shape=[1,batch_size,hidden_size]
        out,decoder_hidden=self.gru(decoder_input_embeded,decoder_hidden)

        ###############################添加attention########################
        #attention_weight.shape=[batch_size,seq_len] encoder_outputs.shape=[batch_size,input_seq_len,input_hidden_size]
        attention_weight=self.atten(decoder_hidden,encoder_outputs).unsqueeze(1)#attention_weight.shape=[batch_size,seq_len]-->[batch_size,1,seq_len]
        # print("attention_weight1",attention_weight.size())
        # attention_weight=attention_weight
        # print("attention_weight",attention_weight.size())
        context_vector=attention_weight.bmm(encoder_outputs)#矩阵乘法 #context_vector.shape[batch_size,1,input_hidden_size]
        concated=torch.cat([out,context_vector],dim=-1).squeeze(1) #将两个矩阵合并 在dim=-1这个维度上进行合并 [batch_size,1,decoder_hidden_size+encoder_hidden_size]-->[batch_size,decoder_hidden_size+encoder_hidden_size]
        out=torch.tanh(self.Wa(concated))

        #########################attention结束#######################
        #处理out[batch_size,1,hidden_size]--->[batch_size,vocab_size]
        #out=out.squeeze(1)#[batch_size,1,hidden_size]--->[batch_size,hidden_size]
        output=F.log_softmax(self.fc(out),dim=-1)#[batch_size,hidden_size]---->[batch_size,vocab_size]



        # print("output:",output.size())
        return output,decoder_hidden



    def evalueate(self,encoder_hidden,encoder_outputs):
        """
        评估
        :param encoder_hidden:
        :return:
        """
        decoder_hidden=encoder_hidden #shape[1,batch_size,hidden_size]
        batch_size=encoder_hidden.size(1)
        decoder_input=torch.LongTensor(torch.ones([batch_size,1],dtype=torch.int64)*config.chatbot_ws_target.SOS).to(config.device)

        indices=[]

        for i in range(config.chatbot_target_max_len+5):#预测值肯定是要比max_len+5要短的
            decoder_output_t,decoder_hidden=self.forward_step(decoder_input,decoder_hidden,encoder_outputs)#decoder_input和前面来的hidden作为forward计算的数值，输出为  本时刻的输出decoder_output_t,本时刻的隐藏状态decoder_hidden
            value,index=torch.topk(decoder_output_t,1)#获取当前时刻最大值和对应的索引#[batch_size,1]
            decoder_input=index
            # if index.item()==config_demo.num_sequence.EOS:#EOS为结束符
            #     break

            indices.append(index.squeeze(-1).cpu().detach().numpy())
        return indices #[max_len,batch_size]
        #indices形状
        #[
        #   [batch_size,1]         batch_size个输入的第一个输出
        #   [batch_size,1]          batch_size个输入的第2个输出
        #   [batch_size,1]  batch_size个输入的第3个输出
        #   [batch_size,1]  batch_size个输入的第4个输出
        #]






