#coding:utf-8
#Gibbs Sampling in Python;
#input:分词后的训练文档：train.dat；
        #日志配置文件：logging.conf；
        #配置文件：setting.conf（提供输入输出文件保存的路径，
        #聚类个数K，迭代次数iter_times,每个类特征词个数top_words_num,超参数α（alpha） β(beta)
#output:词与序号对应关系：wordidmap.dat
        #文章-主题分布：model_theta.dat
        #词-主题分布：model_phi.dat
        #采用的参数：model_parameter.dat
        #每个主题的top N词：model_twords.dat
        #文章中词的主题分派结果：model_tassign.dat



#########################################
#导入相应的模块
#########################################
#日志相关
import logging
import logging.config
#读写配置文件
import ConfigParser
#数组操作相关
import numpy as np
#操作系统相关
import os
#自然语言编码转换
import codecs
#生成随机数
import random
#有序字典
from collections import OrderedDict 


#########################################
#初始化，读入配置文件，定义全局变量
#########################################
#获得当前路径
path = os.getcwd()
#导入日志配置文件，格式
logging.config.fileConfig("logging.conf")
#创建日志对象
logger = logging.getLogger()
#读取配置文件
conf = ConfigParser.ConfigParser()
conf.read("setting.conf")
#从配置文件中导入各输入输出文件路径
trainfile = os.path.join(path,  os.path.normpath(conf.get("filepath",  "trainfile")))
wordidmapfile = os.path.join(path,  os.path.normpath(conf.get("filepath",  "wordidmapfile")))
thetafile = os.path.join(path,  os.path.normpath(conf.get("filepath",  "thetafile")))
phifile = os.path.join(path,  os.path.normpath(conf.get("filepath",  "phifile")))
paramfile = os.path.join(path,  os.path.normpath(conf.get("filepath",  "paramfile")))
topNfile = os.path.join(path,  os.path.normpath(conf.get("filepath",  "topNfile")))
tassginfile = os.path.join(path,  os.path.normpath(conf.get("filepath",  "tassginfile")))

#从配置文件中导入初始参数
K = int(conf.get("modelparas","K"))
alpha = float(conf.get("modelparas","alpha"))
beta = float(conf.get("modelparas","beta"))
iter_times = int(conf.get("modelparas","iter_times"))
top_words_num = int(conf.get("modelparas","top_words_num"))


#########################################
#训练文档的预处理
#########################################
#保存一个文档的信息
class document(object):
    def __init__(self):
        self.words = []
        self.length = 0
        
#保存多个文档的整体信息
class datapreprocessing(object):
    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.wordid = OrderedDict()

    def cachewordidmap(self):
        #创建文件wordidmap,将word,id写入文件中
        with codecs.open(wordidmapfile, 'w', 'utf-8') as f:
            for word, id in self.wordid.items():
                f.write(word+'\t'+str(id)+'\n')


#The preprocessing of the input documents
#input:Documets prepared to train the model
#output:information of words from the prepared documents
def Preprocessing():
    with codecs.open(trainfile, 'r', 'utf-8') as f:
        #读取文档中的全部内容，为一个数组，每一行为一个元素
        docs=f.readlines()
    data=datapreprocessing()
    index = 0
    #按照行，来读取文档
    for line in docs:
        if line !="":
            sumwords=line.strip().split()
            #将每一行的文本保存在doc中
            doc=document()
            for item in sumwords:
                if data.wordid.has_key(item):
                    doc.words.append(data.wordid[item])
            
                else :
                    data.wordid[item]=index
                    doc.words.append(index)
                    index +=1
            
            doc.length=len(sumwords)
            data.docs.append(doc)
            
            
        else:
            pass
    data.docs_count=len(data.docs)
    data.words_count=len(data.wordid)
    logger.info(u"共有%s个文档" % data.docs_count)
    data.cachewordidmap()
    logger.info(u"词与序号对应关系已保存到%s" % wordidmapfile)
 
    return data
    


#########################################
#LDA训练过程
#########################################

class LDA(object):

    def __init__(self,data):

        self.data = data
        
        #模型参数
        #聚类个数K，迭代次数iter_times,每个类特征词个数top_words_num,超参数α（alpha） β(beta)
        #把全局变量赋给对象自己的变量
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iter_times=iter_times
        self.top_words_num = top_words_num 

        #
        #文件变量
        #分好词的文件trainfile
        #词对应id文件wordidmapfile
        #文章-主题分布文件thetafile
        #词-主题分布文件phifile
        #每个主题top N词文件topNfile
        #最后分派结果文件tassginfile
        #模型训练选择的参数文件paramfile
        self.wordidmapfile = wordidmapfile
        self.trainfile = trainfile
        self.thetafile = thetafile
        self.phifile = phifile
        self.topNfile = topNfile
        self.tassginfile = tassginfile
        self.paramfile = paramfile

        #
        # p,概率向量 double类型，存储采样的临时概率
        # nw,topic中每个word的总数
        # nwsum,每各topic的词的总数
        # nd,每个doc中各个topic的总数
        # ndsum,每各doc中词的总数
        # z,哪个doc中的哪个词对应的tipic
        self.p = np.zeros(self.K) 
        self.nw = np.zeros((self.data.words_count,self.K),dtype='int')
        self.nwsum = np.zeros(self.K,dtype='int')
        self.nd = np.zeros((self.data.docs_count, self.K),dtype='int')
        self.ndsum = np.zeros (self.data.docs_count,dtype='int')
        self.z = np.array([[ 0 for y in xrange(self.data.docs[x].length)] for x in xrange(self.data.docs_count)])

        #给每个文档中的每个词，随机赋予一个主题
        for x in xrange (len(self.z)):
            self.ndsum[x]=self.data.docs[x].length
            for y in xrange (self.data.docs[x].length):
                topic = random.randint(0,self.K-1)
                self.z[x][y]=topic
                self.nw[self.data.docs[x].words[y]][topic] +=1
                self.nwsum[topic] +=1
                self.nd[x][topic] +=1
        #初始化文章-主题分布、主题-词分布文件
        self.theta = np.array([ [0.0 for y in xrange(self.K)] for x in xrange(self.data.docs_count) ])
        self.phi = np.array([ [ 0.0 for y in xrange(self.data.words_count) ] for x in xrange(self.K)]) 
        

    def TrainModel(self):
        for x in xrange(self.iter_times):
            for i in xrange(self.data.docs_count):
                for j in xrange(self.data.docs[i].length):
                    #吉布斯抽样，抽取主题
                    topic=self.Sampling(i,j)
                    self.z[i][j]=topic

        #保存结果
        logger.info(u"迭代完成。")
        logger.debug(u"计算文章-主题分布")
        self._theta()
        logger.debug(u"计算词-主题分布")
        self._phi()
        logger.debug(u"保存模型")
        self.save()

    #Gibbs sampling过程
    def Sampling(self,i,j):

        word=self.data.docs[i].words[j]
        topic=self.z[i][j]

        self.nw[word][topic] -= 1
        self.nwsum[topic] -= 1
        self.nd[i][topic] -= 1
        self.ndsum[i] -= 1
        
        self.p=((self.nd[i]+self.alpha)/(self.ndsum[i]+self.K * self.alpha))*\
                ((self.nw[word] + self.beta)/(self.nwsum+self.data.words_count * self.beta))
        for k in xrange(1, self.K):
            self.p[k] += self.p[k-1]
        u = random.uniform(0,self.p[self.K-1])
        for topic in xrange(self.K):
            if self.p[topic]>u:
                break

        self.nw[word][topic] +=1
        self.nwsum[topic] +=1
        self.nd[i][topic] +=1
        self.ndsum[i] +=1

        return topic
    
    #计算文章-主题分布theta    
    def _theta(self):
        for i in xrange(self.data.docs_count):
            self.theta[i] = (self.nd[i]+self.alpha)/(self.ndsum[i]+self.K * self.alpha)
    #计算主题-词分布phi 
    def _phi(self):
        for i in xrange(self.K):
            self.phi[i] = (self.nw.T[i] + self.beta)/(self.nwsum[i]+self.data.words_count * self.beta)
    #保存分布信息、参数、每个主题的top N词、文章中词分派的主题
    def save(self):
        #保存theta文章-主题分布
        logger.info(u"文章-主题分布已保存到%s" % self.thetafile)
        with codecs.open(self.thetafile,'w') as f:
            for x in xrange(self.data.docs_count):
                for y in xrange(self.K):
                    #'\t'空格
                    f.write(str(self.theta[x][y]) + '\t')
                f.write('\n')
        #保存phi词-主题分布
        logger.info(u"词-主题分布已保存到%s" % self.phifile)
        with codecs.open(self.phifile,'w') as f:
            for x in xrange(self.K):
                for y in xrange(self.data.words_count):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')
        #保存参数设置
        logger.info(u"参数设置已保存到%s" % self.paramfile)
        with codecs.open(self.paramfile,'w','utf-8') as f:
            f.write('K=' + str(self.K) + '\n')
            f.write('alpha=' + str(self.alpha) + '\n')
            f.write('beta=' + str(self.beta) + '\n')
            f.write(u'迭代次数  iter_times=' + str(self.iter_times) + '\n')
            f.write(u'每个类的高频词显示个数  top_words_num=' + str(self.top_words_num) + '\n')
        #保存每个主题topic的词
        logger.info(u"主题topN词已保存到%s" % self.topNfile)
        with codecs.open(self.topNfile,'w','utf-8') as f:
            self.top_words_num = min(self.top_words_num,self.data.words_count)
            for x in xrange(self.K):
                f.write(u'第' + str(x) + u'类：' + '\n')
                twords = []
                twords = [(n,self.phi[x][n]) for n in xrange(self.data.words_count)]
                twords.sort(key = lambda i:i[1], reverse= True)
                for y in xrange(self.top_words_num):
                    word = OrderedDict({value:key for key, value in self.data.wordid.items()})[twords[y][0]]
                    f.write('\t'*2+ word +'\t' + str(twords[y][1])+ '\n')
        #保存最后退出时，文章的词分派的主题
        logger.info(u"文章-词-主题分派结果已保存到%s" % self.tassginfile)
        with codecs.open(self.tassginfile,'w') as f:
            for x in xrange(self.data.docs_count):
                for y in xrange(self.data.docs[x].length):
                    f.write(str(self.data.docs[x].words[y])+':'+str(self.z[x][y])+ '\t')
                f.write('\n')
        logger.info(u"模型训练完成。")

if __name__ == "__main__":
    data=Preprocessing()
    lda=LDA(data)
    lda.TrainModel()

 



