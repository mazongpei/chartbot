import torch
from  torch.nn.utils.rnn import pad_sequence

class WordSequence:
    PAD_TAG = "PAD"  # 填充词
    UNK_TAG = "UNK"  # 未知词
    SOS_TAG = "SOS"  # start of sentence 句子开始符号
    EOS_TAG = "EOS"  # END of sentence句子结束符号
    PAD = 0  # 填充
    UNK = 1  # 未知
    SOS = 2  # 开始
    EOS = 3  # 结束

    def __init__(self):
        self.dict = {self.PAD_TAG: self.PAD,
                     self.UNK_TAG: self.UNK,
                     self.SOS_TAG: self.SOS,
                     self.EOS_TAG: self.EOS
                     }
        self.count={}#字典 保留词


    def fit(self,sentence):
        """
        传入一条一条的句子，进行词频统计
        :param sentence:[]
        :return:
        """
        for word in sentence:
            self.count[word]=self.count.get(word,0)+1




    def build_vocab(self,min_count=0,max_count=None,max_features=None):
        """
        构造词典
        :param min_count: 最小词频数量
        :param max_count: 最大词频数量
        :param max_features: 一共可以有多少个词
        :return:
        """
        temp = self.count.copy()
        for key in temp:
            cur_count = self.count.get(key, 0)  #
            if min_count is not None:#获取词频，删除词频小于5的词
                if cur_count<min_count:
                    del self.count[key]

            if max_count is not None:
                if cur_count>max_count:
                    del self.count[key]
        if max_features is not None:
            self.count=dict(sorted(self.count.items(),key=lambda x:x[1],reverse=True)[:max_features])#取词频数量最大的max_feature个

        #构造字典
        for key in self.count:
            self.dict[key]=len(self.dict)
        self.inverse_dict=dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len,add_eos=False):
        """
        传入一个句子sentence
        把sentence转换为数字序列
        :parameter sentence:输入的句子
        :parameter max_len:句子的最大长度
        :parameter add_eos:True在句子后加上结束符号EOS，输出句子的长度为max_len+1
                   add_eos:False在句子后不加EOS，输出的句子长度为max_len
        """



        if len(sentence)>max_len:#如果句子长度超过max_len的话删除后面多余的
            sentence=sentence[:max_len]#0->max_len

        sentence_len = len(sentence)#提前计算句子长度，实现add_eos后，句子长度统一

        if add_eos:# 如果需要加eos结束符的话
            sentence = sentence + [self.EOS_TAG]

        if len(sentence)<max_len:#句子长度<max_len
            sentence=sentence+[self.PAD_TAG]*(max_len-sentence_len)#使用PAD（填充符所代表的数据）进行填充！

        result = [self.dict.get(i, self.UNK) for i in sentence]  # 取sentence中每一个把他转换成数字，如果取不到的话就返回UNK

        return result

    def inverse_transform(self,indices):
        """把序列转为字符串"""
        #indices= [self.inverse_dict.get(i ,self.UNK_TAG) for i in indices]#如上同理，从incerse_dict中取i，如果没有取到就返回self.UNK_TAG
        result=[]
        for i in indices:
            if i==self.EOS:
                break
            result.append(self.inverse_dict.get(i,self.UNK_TAG))
        return result








    def __len__(self):
        return len(self.dict)
if __name__ == '__main__':
    num_sequence=WordSequence()
    print(num_sequence.dict)