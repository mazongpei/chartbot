# Coding by 马宗沛
# Datatime:2022/3/8 18:43
# Filename:dataset.py
# Toolby: PyCharm
"""完成数据集的准备"""


from torch.nn.utils.rnn import pad_sequence

import torch
from torch.utils.data import Dataset,DataLoader
import config






class ChatbotDataset(Dataset):

    def __init__(self):
        self.input_path=config.chatbot_input_path
        self.target_path=config.chatbot_target_path
        self.input_lines=open(self.input_path,encoding='utf-8').readlines()
        self.target_lines=open(self.target_path,encoding='utf-8').readlines()
        assert len(self.input_lines) == len(self.target_lines),"input和target长度不一致"#保障input和target长度一致

    def __getitem__(self, index):
        ##获取数据中的每一个元素
        input = self.input_lines[index].strip().split()#按照空格进行切分
        target = self.target_lines[index].strip().split()
        input_length=len(input) if len(input) <config.chatbot_input_max_len else config.chatbot_input_max_len
        target_length=len(target) if len(input) <config.chatbot_target_max_len+1 else config.chatbot_target_max_len+1
        return input,target,input_length,target_length


    def __len__(self):
        return len(self.input_lines)




def collate_fn(batch):#重写collate_fn函数
    """

    :param batch: [(一条数据),(input,target,input_length,target_length)]
    :return:
    """
    #1.排序
    batch=sorted(batch,key=lambda x:x[-2],reverse=True)
    input,target,input_length,target_length=zip(*batch)#对数据进行解包，（input,target,input_length,target_length）
    input=[config.chatbot_ws_input.transform(i,max_len=config.chatbot_input_max_len) for i in input]
    input=torch.LongTensor(input)

    target=[config.chatbot_ws_target.transform(i,max_len=config.chatbot_target_max_len,add_eos=True) for i in target]
    target = pad_sequence([torch.LongTensor(i) for i in target],batch_first=True)#自己改的，增加维度。空缺的填充0
    target=torch.LongTensor(target)


    #print('target',target)
    #print("target.shape",target.size())
    input_length=torch.LongTensor(input_length)
    target_length=torch.LongTensor(target_length)
    return input,target,input_length,target_length







#实例化DataLoader
train_data_loader=DataLoader(ChatbotDataset(),batch_size=config.chatbot_batch_size,shuffle=True,collate_fn=collate_fn)
















