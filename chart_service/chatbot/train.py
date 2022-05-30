# Coding by 马宗沛
# Datatime:2022/3/9 15:24
# Filename:train.py
# Toolby: PyCharm

from chatbot.dataset import train_data_loader
from chatbot.seq2seq  import Seq2Seq
from torch.optim import Adam
import torch.nn.functional as F
import config
from tqdm import tqdm
import torch
import torch.nn as nn

#训练流程
#1.实例化model，optimizer loss
#2.遍历dataloder
#3.调用模型得到output
#4，计算损失
#5.模型的保存和加载

seq2seq=Seq2Seq()

seq2seq=seq2seq.to(config.device)
#seq2seq=seq2seq.load_state_dict(torch.load(config.chatbot_model_save_path))
#实例化优化器
optimizer=Adam(seq2seq.parameters(),lr=0.001)
#参数解析：seq2seq.parameters()为模型参数，lr学习率
#optimizer.load_state_dict(torch.load(config.chatbot_optimizer_save_path))
def train(epoch):#定义train方法，进行epoch轮训练
    total_loss = []
    bar=tqdm(enumerate(train_data_loader),total=len(train_data_loader),ascii=True,desc="train")
    for index,(input,target,input_length,target_length) in bar:
        #input,target,input_length,target_length 为dataloder的返回值，打开dataset.py即可查看
        #将数据放到GPU上
        input=input.to(config.device)
        target=target.to(config.device)
        input_length=input_length.to(config.device)
        target_length=target_length.to(config.device)
        #梯度归零,必须要在计算损失之前
        optimizer.zero_grad()
        decoder_outputs,_=seq2seq(input,target,input_length,target_length)
        #将参数传入seq2seq得到decoder——outputs,和decoder_hidden，因为我们不需要decoder_hidden所以写_
        #计算得到损失

        #print(decoder_outputs.size(),target.size())
        #为了方便计算loss    我将decoder_outputs转化为2维（原来为3维），将target转换为1维，原来为二维
        #问题发现：decoder——从一开始是cpu类型的tensor，但是已经在seq2seq（）里todevice了，改！
        decoder_outputs=decoder_outputs.view(decoder_outputs.size(0)*decoder_outputs.size(1),-1)
        target=target.view(-1)

        loss=F.nll_loss(decoder_outputs,target,ignore_index=config.chatbot_ws_target.PAD)#计算带权损失，decoder_out_puts是模型生成的结果，target是被预测的结果

        loss.backward()#进行反向传播
        nn.utils.clip_grad_norm_(seq2seq.parameters(),config.clip)
        optimizer.step()#进行参数(也就是梯度的)更新
        bar.set_description("epoch:{}\t index:{}\t loss:{:.3f}".format(epoch,index,loss.item()))
        if index%10==0:
            torch.save(seq2seq.state_dict(),config.chatbot_model_save_path)
            torch.save(optimizer.state_dict(),config.chatbot_optimizer_save_path)

