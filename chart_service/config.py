""""配置文件"""
import pickle

import torch

device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
#############语料相关############
user_dict_path="corpus/user_dict/keyword.txt"
stopwords_path=r"corpus/user_dict/stopwords.txt"#"\corpus\user_dict\stopwords.txt"
classify_corpus_train_path="corpus/classify/classify_train.txt"
classify_corpus_test_path="corpus/classify/classify_test.txt"


classify_corpus_by_word_train_path="corpus/classify/classify_train_by_word.txt"
classify_corpus_bu_word_test_path="corpus/classify/classify_test_by_word.txt"




###################################分类相关##############################
classify_model_path="model/classify.model"#词语作为特征的模型的保存地址
classify_model_path_by_word="model/classify_by_word.model"#单个字作为特征的模型的保存地址


classify_model_final_path="model/classify.model"#一个词语作为特征的模型
classify_model_final_path_by_word="model/classify_by_word.model"#单个字作为特征的模型






##################################chatbot相关################################
#
# chatbot_input_path1=r'E:\Chat_robot_project\chart_service\corpus\chatbot\input.txt'#将小黄鸡未分词的问 放到一个路径下
# chatbot_target_path1=r'E:\Chat_robot_project\chart_service\corpus\chatbot\target.txt'#将小黄鸡未分词的回答 放到一个路径下


chatbot_by_word=True
if chatbot_by_word:

    chatbot_input_path='corpus/chatbot/input_by_word.txt'#将小黄鸡未分词的问 放到一个路径下
    chatbot_target_path='corpus/chatbot/target_by_word.txt'#将小黄鸡未分词的回答 放到一个路径下

else:
    chatbot_input_path='corpus/chatbot/input.txt'#将小黄鸡未分词的问 放到一个路径下
    chatbot_target_path='corpus/chatbot/target.txt'#将小黄鸡未分词的回答 放到一个路径下

#ws
if chatbot_by_word:
    chatbot_ws_input_path = "model/chatbot/ws_by_word_input_path.pkl"
    chatbot_ws_target_path="model/chatbot/ws_by_word_target_path.pkl"
else:
    chatbot_ws_input_path="model/chatbot/ws_input_path.pkl"
    chatbot_ws_target_path="model/chatbot/ws_target_path.pkl"



chatbot_ws_input=pickle.load(open(chatbot_ws_input_path,'rb'))
chatbot_ws_target=pickle.load(open(chatbot_ws_target_path,'rb'))

chatbot_batch_size=128
if chatbot_by_word:
    chatbot_input_max_len=30
    chatbot_target_max_len = 30
else:
    chatbot_input_max_len = 12
    chatbot_target_max_len = 12


chatbot_embedding_dim=256
chatbot_encoder_num_layers=1
chatbot_encoder_hidden_size=128



chatbot_decoder_num_layer=1
chatbot_decoder_hidden_size=128


chatbot_teacher_forcing_ratio=0.7


chatbot_model_save_path="model/chatbot/seq2seq.model" if chatbot_by_word else "model/chatbot/seq2seq_by_word.model"
chatbot_optimizer_save_path="model/chatbot/optimizer.model" if chatbot_by_word else "model/chatbot/optimizer_by_word.model"

clip=0.01