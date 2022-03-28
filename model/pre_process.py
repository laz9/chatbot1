import re,jieba,random,time
import numpy as np
import torch
from sklearn.model_selection import train_test_split

class Corpus:
    def __init__(self,filePath,maxSentenceWordNum=-1,id2word=None,word2id=None,wordNum=None,tfidf=False,QIDF=None,AIDF=None,testSize=0.2,isCharVec=False):
        self.id2word,self.word2id,self.wordNum=id2word,word2id,wordNum
        # 遍历所有内容
        with open(filePath,'r',encoding='utf-8') as f:
            txt = self._purity(f.readline())
            data = [i.split(',') for i in txt]
            # 判断是否有字符，使用jieba进行分词
            if isCharVec:
                data = [[[c for c in i[0]],[c for c in i[1]]] for i in data]
            else:
                data = [[jieba.lcut(i[0]),jieba.lcut(i[1])] for i in data]
            data = [i for i in data if (len(i[0])<maxSentenceWordNum and len(i[1])<maxSentenceWordNum) or maxSentenceWordNum == -1]
            self.chatDataWord = data
            self._word_id_map(data)
            try:
                chatDataId = [[[self.word2id[w] for w in qa[0]],[self.word2id[w] for w in qa[i]]] for qa in self.chatDataWord]
            except:
                chatDataId = [[[self.word2id[w] for w in qa[0] if w in self.id2word],[self.word2id[w] for w in qa[i] for w in qa[i] if w in self.id2word]] for qa in self.chatDataWord]

            self._QAlens( chatDataId)
            self.maxSentLen = max(maxSentenceWordNum,self.AMaxLen)
            self.QChatDtaId,self.AChatDataId = [qa[0] for qa in chatDataId],[qa[1] for qa in chatDataId]
            self.totalSampleNum = len(data)

            #定义toidf手动实现
            if tfidf:
                self.QIDF = QIDF if QIDF is not None else np.array([np.log(self.totalSampleNum/(sum([(i in qa[0]) for qa in chatDataId])+1)) for i in range(self.wordNum)],dtype='float32')
                self.AIDF = AIDF if AIDF is not None else np.array([np.log(self.totalSampleNum/(sum([(i in qa[1]) for qa in chatDataId])+1)) for i in range(self.wordNum)],dtype='float32')
            print("Total sample num:",self.totalSampleNum)

            # 数据集的划分 训练和测试的数据
            self.trainIdList,self.testIdList = train_test_split([i for i in range(self.totalSampleNum)],test_size=testSize)
            self.trainSampleNum,self.testSampleNum = len(self.trainIdList,self.testIdList)
            print("train size:%d;test size:%d"%(self.trainSampleNum,self.testSampleNum))
            self.testSize = testSize
            print("Finished loading Corpus")



        # 重置词的ID和映射关系
        def reset_word_id_map(self,id2word,word2id):
            self.id2word,self.word2id = id2word,word2id
            chatDataId = [[[self.word2id[w] for w in qa[0]],[self.word2id[w] for w in qa[1]]] for qa in self.chatDataWord]
            self.QchatDataId,self.AchatDataId = [qa[0] for qa in chatDataId],[qa[1] for qa in chatDataId]


        # 数据随机打乱 随机数据流
        def random_batch_data_stream(self,batchSize=128,isDataEnhance=False,dataEnhanceRatio=0.2,tyoe='train'):
            # 判断数据是不是训练数据，另外给定结束符
            idList = self.trainIdList if type == 'train' else self.testIdList
            eosToken,unkToken = self.word2idp['<EOS>'],self.word2idp['<UNK>']
            while True:
                samples = random.sample(idList,min(batchSize.len(idList))) if batchSize>0 else random.sample(idList,len(idList))
                # 数据增广
                if isDataEnhance:
                    yield self._dataEnhance(samples,dataEnhanceRatio,eosToken,unkToken)
                else:
                    QMaxLen,AMaxLen = max(self.Qlens[samples]),max(self.Alens[samples])
                    QDataId = np.array([self.QchatDataId[i]+[eosToken for j in range(QMaxLen-self.Qlens[i]+1)] for i in samples],dtype='float32')
                    ADataId = np.array([self.AchatDataId[i]+[eosToken for j in range(AMaxLen-self.Alens[i]+1)] for i in samples],dtype='float32')
                    yield QDataId,self.Qlens[samples],ADataId,self.Alens[samples]
                    






