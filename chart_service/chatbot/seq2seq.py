"""完成seq2seq模型
把encoder和decoder合并得到seq2seq模型
"""
import torch
import torch.nn as nn
from chatbot.encoder import Encoder
from chatbot.decoder import Decoder
import config
from torch.nn.utils.rnn import pad_sequence
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        #初始化encoder和decoder
        self.encoder=Encoder().to(config.device)#使用gpu
        self.decoder=Decoder().to(config.device)#使用gpu

    def forward(self,input,target,input_length,target_length):
        # input=pad_sequence([i for i in input],batch_first=True)
        # target = pad_sequence([i for i in target], batch_first=True)
        # input_length = pad_sequence([i for i in input_length], batch_first=True)
        # target_length = pad_sequence([i for i in target_length], batch_first=True)

        #将input和input_length传入加码器encoder得到outputs和隐藏状态（hidden）
        encoder_outputs,encoder_hidden=self.encoder(input,input_length)

        #将target,encoder_hidden传入解码器decoder，得到outputs和hidden
        decoder_outputs,decoder_hidden=self.decoder(target,encoder_hidden,encoder_outputs)
        return decoder_outputs,decoder_hidden



    def evaluate(self,input,input_length):#encoderoutput后来加的

        encoder_outputs,encoder_hidden=self.encoder(input,input_length)
        indices=self.decoder.evalueate(encoder_hidden,encoder_outputs)#预测结果,encoder_outputs
        return indices

#
#
#
#                                                                     ///////////
#                                                                  //         //
#                                                               //          //
#                                                            //           //
#                                                         //             //
#                                                   //                 //
#                                                 //                 //
#                                               //                 //
#                                             //                 //   //
#                                           //                 //         //
#                                         //                 //               //
#                                       //                 //                     //
#                                     //                 // //                        //
#                                   //                 //       //                        //
#                                 //                 //             //                        //
#                               //                 //                   //                        //
#                                 //             //                         //                         //
#                                    //        //                               //                         //
 #                                       //   //                                     //                         //
#                                          //                                           //                          //
#                                                                                           //                          //
#                                                                                               //                          //
#                                                                                                   //                           //
#                                                                                                       //                           //
#                                                                                                           //
#                                                                                                               //
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
