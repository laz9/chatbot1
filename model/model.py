import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from matplotlib import ticker
from nltk.translate.bleu_score import sentence_bleu
import time,random,os,jieba,logging
import numpy as np
import pandas as pd
jieba.setLogLevel(logging.INFO)

# 定义开始符和结束符
sosToken = 1
eosToken = 0

# 定义Encoder

class EncoderRNN(nn.Module):
    # 初始化
    def __init__(self,featureSize,hiddenSize,embedding,numLayers=1,dropout=0.1,bidirectional=True):
        super(EncoderRNN,self).__init__()
        self.embedding = embedding
        # 核心API
        self.gru = nn.GRU(featureSize,hiddenSize,num_layers = numLayers,dropout=(0 if numLayers==1 else dropout),bidirectional=bidirectional,batch_first=True)
        # 超参
        self.featureSize = featureSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.bidirectional = bidirectional

        # 前向计算，训练和测试中必须的部分
    def forward(self,input,lengths,hidden):
        # input:batchSize*seq_len;hidden:numLayers*d*batchSize*hiddenSize
        # 给定输入
        input = self.embedding(input)    #batchSize * seq_len * featureSize
        # 加入paddle方便计算
        packed = nn.utils.rnn.pack_padded_sequence(input,lengths,batch_first=True)
        output,hn = self.gru(packed,hidden)    #output batchSize * seq_len * hiddenSise * d    hn:numLayers*d*batchSize*hiddenSize
        output,__ = nn.utils.rnn.pack_padded_sequence(output,batch_first=True)
        # 判断是否使用双向GRU
        if self.bidirectional:
            output = output[:,:,:self.hiddenSize] + output[:,:,:self.hiddenSize]
        return output,hn

# 定义Decoder
class DecoderRNN(nn.Module):
    # 初始化
    def __init__(self, featureSize, hiddenSize,outputSize, embedding, numLayers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.embedding = embedding
        # 核心API 搭建GRU
        self.gru = nn.GRU(featureSize,hiddenSize,num_layers=numLayers,batch_first=True)
        self.out = nn.Linear(featureSize,outputSize)

        # 前向传播
    def forward(self,input,hidden):
        input = self.embedding(input)
        # relu 激活 softmax计算结果
        input = F.relu(input)
        output,hn = self.gru(input,hidden)
        output = F.log_softmax(self.out(output),dim=2) #output:batchSize * seq_len * outputSize
        return output,hn,torch.zeros([input.size(0),1,input.size(1)])

# 定义B Attention的Decoder
class BahdanauAttentionDecoderRNN(nn.Module):
    # 初始化
    def __init__(self, featureSize, hiddenSize, outputSize, embedding, numLayers=1, dropout=0.1):
        super(BahdanauAttentionDecoderRNN,self).__init__()
        self.embedding = embedding

        # 定义attention的权重，如何去联合，以及dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_weight = nn.Linear(hiddenSize*2,1)
        self.attention_combine = nn.Linear(featureSize+hiddenSize,featureSize)

        # 核心API 搭建GRU层，并给定超参
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers,dropout=(0 if numLayers==1 else dropout), batch_first=True)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.numLayers = numLayers

    # 定义前向计算
    def forward(self,inputStep,hidden,encoderOutput):
        # 防止过拟合
        inputStep = self.embedding(inputStep)
        inputStep = self.dropout(inputStep)
        #计算attention的权重 本质softmax
        attentionWeight = F.softmax(self.attention_weight(torch.cat((encoderOutput,hidden[-1:].expand(encoderOutput.size(1),-1,-1).transpose(0,1)),dim=2)).transpose(1,2),dim=2)
        context = torch.bmm(attentionWeight,encoderOutput) #context:batchSize *1* hiddenSize
        attentionCombine = self.attention_combine(torch.cat((inputStep,context),dim=2)) #combine:batchSize * featureSize
        attentionInput = F.relu(attentionCombine)
        output,hidden = self.gru(attentionInput,hidden)
        output = F.softmax(self.out(output),dim=2)
        return output,hidden,attentionWeight

# 定义L attention
class LuongAttention(nn.Module):
    # 初始化
    def __init__(self, method, hiddenSize):
        super(LuongAttention, self).__init__()
        self.method = method
        # 三种模式 dot general concat
        if self.method not in ['dot','general','concat']:
            raise ValueError(self.method,"is not an attention method.")
        if self.method == 'general':
            self.Wa = nn.Linear(hiddenSize,hiddenSize)
        if self.method == 'concat':
            self.Wa = nn.Linear(hiddenSize*2,hiddenSize)
            self.v = nn.Parameter(torch.FloatTensor(1,hiddenSize))
        # 给出dot计算方法
        def dot_score(self,hidden,encoderOutput):
            return torch.sum(hidden*encoderOutput,dim=2)

        # 给出general计算方法
        def general_score(self, hidden, encoderOutput):
            energy = self.Wa(encoderOutput)
            return torch.sum(hidden*energy,dim=2)

        # 给出concat计算方法
        def concat_score(self, hidden, encoderOutput):
            energy = torch.tanh(self.Wa(torch.cat((hidden.expand(-1,encoderOutput(1),-1),encoderOutput),dim=2)))
            return torch.sum(hidden * energy, dim=2)

    # 定义前向计算
    def forward(self,hidden,encoderOutput):
        # 3选1
        if self.method == 'general':
            attentionScore = self.general_score(hidden,encoderOutput)
        elif self.method == 'concat':
            attentionScore = self.concat_score(hidden,encoderOutput)
        elif self.method == 'dot':
            attentionScore = self.dot_score(hidden,encoderOutput)
        return F.softmax(attentionScore,dim=1)

# 定义L Attention Decoder
class LuongAttentionDecoderRNN(nn.Module):
    # 初始化
    def __init__(self, featureSize, hiddenSize, outputSize, embedding, numLayers=1, dropout=0.1,attnMethod='dot'):
        super(LuongAttention, self).__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)

        # 核心API 搭建GRU层，并给定超参
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, dropout=(0 if numLayers == 1 else dropout),batch_first=True)
        # 定义权重计算和连接方式
        self.attention_weight = LuongAttention(attnMethod,hiddenSize)
        self.attention_combine = nn.Linear(hiddenSize*2,hiddenSize)
        self.out = nn.Linear(hiddenSize, hiddenSize)
        self.numLayers = numLayers

    # 定义前向计算
    def forward(self, inputStep, hidden, encoderOutput):
        # 防止过拟合
        inputStep = self.embedding(inputStep)
        inputStep = self.dropout(inputStep)
        # 对输出计算
        output, hidden = self.gru(inputStep, hidden)
        # attenion 权重计算
        attentionWeight = self.attention_weight(output,encoderOutput)
        context = torch.bmm(attentionWeight,encoderOutput)
        attentionCombine = self.attention_combine(torch.cat(output, context), dim=2)
        attentionOutput = torch.tanh(attentionCombine)
        # 最终的output
        output = F.log_softmax(self.out(attentionOutput),dim=2)
        return output, hidden, attentionWeight

#如何去选择decoder L B None
def __DecoderRNN(attnType,featureSize,hiddenSize,outputSize,embedding,numLayers,dropout,attnMethod):
    #使用哪个attention
    if attnType not in ['L','B','Mone']:
        raise ValueError(attnType,"is not an appropriate attention type")
    if attnType == 'L':
        return LuongAttentionDecoderRNN(featureSize,hiddenSize,outputSize,embedding=embedding,numLayers=numLayers,dropout=dropout,attnMethod=attnMethod)
    elif attnType == 'B':
        return BahdanauAttentionDecoderRNN(featureSize,hiddenSize,outputSize,embedding=embedding,numLayers=numLayers,dropout=dropout)
    else:
        return DecoderRNN(featureSize,hiddenSize,outputSize,embedding=embedding,numLayers=numLayers,dropout=dropout)


#seq2seq 定义核心类
class seq2seq:
    # 初始化
    def __init__(self,dataClass,featureSize,hiddenSize,encoderNumLayers=1,decoderNumLayers=1,attnType='L',attnMethod='dot',dropout=0.1,encoderBidirectional=False,outputSize=None,embedding=None,device=torch.device("cpu")):
        # 定义输出的维度
        outputSize = outputSize if outputSize else dataClass.wordNum
        embedding = embedding if embedding else nn.Embedding(outputSize+1,featureSize)
        # 数据读入
        self.dataClass = dataClass
        # 搭建模型架构
        self.featureSize = featureSize
        self.hiddenSize = hiddenSize
        # encoder调用 构建
        self.encoderRNN = EncoderRNN(featureSize,hiddenSize,embedding=embedding,numLayers=encoderNumLayers,dropout=dropout,bidirectional=encoderBidirectional).to(device)
        # decoder构建
        self.decoderRNN = __DecoderRNN(attnType,featureSize,hiddenSize,outputSize,embedding=embedding,numLayers=decoderNumLayers,dropout=dropout,attnMethod=attnMethod).to(divice)
        self.embedding = embedding.to(device)
        self.device = device

    # 定义训练方法
    def train(self,batchSize,isDataEnhance=False,dataEnhanceRatio=0.2,epoch=100,stopRound=10,lr=0.001,betas=(0.9,0.99),eps='le-08',weight_decay=0,teacherForcingRatio=0.5):
        # 使用哪个api训练
        self.encoderRNN.train(),self.decoderRNN.train()
        # 给定batchSize是否使用数据增广
        batchSize = min(batchSize,self.dataClass.trainSampleNum) if batchSize>0 else self.dataClass.trainSampleNum
        dataStream = self.dataClass.random_batch_data_stream(batchSize=batchSize,isDataEnhance=isDataEnhance,dataEnhanceRatio=dataEnhanceRatio)

        # 定义优化器Adam
        # 对于测试数据batch制作
        if self.dataClass.testsize>0:
            testStrem = self.dataClass.random_batch_data_stream(batchSize=batchSize,type="test")
        itersPerEpoch = self.dataClass.trainSampleNum//batchSize
        encoderOptimzer = torch.optim.Adam(self.encoderRNN.parameters(),lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        decoderOptimzer = torch.optim.Adam(self.decoderRNN.parameters(),lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        st = time.time()
        # 做循环
        for e in range(epoch):
            for i in range(itersPerEpoch):
                X,XLens,Y,YLens = next(dataStream)
                loss = self._train_step(X,XLens,Y,YLens,encoderOptimzer,decoderOptimzer,teacherForcingRatio)
                # blue embAve 评价指标，评判模型好坏的指标，其实是机器翻译的指标
                if (e*itersPerEpoch+i+1)%stopRound==0:
                    bleu = _bleu_score(self.encoderRNN,self.decoderRNN,X,XLens,Y,YLens,self.dataClass.maxSentLen,device=self.device)
                    embAve = _embAve_score(self.encoderRNN,self.decoderRNN,X,XLens,Y,YLens,self.dataClass.maxSentLen,device=self.device)
                    print("After iters%d:loss = %.3lf;train blue:%3lf,embAve:%.3lf"%(e*itersPerEpoch+i+1,loss,bleu,embAve),end='')
                resetNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*batchSize
                speed = ((e*itersPerEpoch+i+1)*batchSize//(time.time()-st))
                print("%.3lf qa/s;remaining time:%.3lf"%(speed,resetNum/speed))

                #保存模型
                def save(self,path):
                    torch.save({"encoder":self.encoderRNN,"decoder":self.decoderRNN,"word2id":self.dataClass.word2id,"id2word":self.dataClass.id2word},path)
                    print("Moder saved in '%s'."%path)

               # 训练中的梯度计算及loss计算
                def _train_step(self,X,XLens,Y,YLens,encoderOptimzer,decoderOptimzer,teacherForcingRatio):
                    #计算梯度，实现BP
                    encoderOptimzer.zero_grad()
                    decoderOptimzer.zero_grad()
                    # 计算loss
                    loss,nTotal = _calculate_loss(self.encoderRNN,self.decoderRNN,X,XLens,Y,YLens,teacherForcingRatio,device=self.device)

                    # 实现BP
                    (loss/nTotal).backward()
                    encoderOptimzer.step()
                    decoderOptimzer.step()
                    return loss.item()/nTotal