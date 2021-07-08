# 文本挖掘比赛：（附四）BERT

## 1、NLP模型比较

NLP模型家族：

![image-20210704201841241](D:\develop\github\MLStudy\Deep Learning\textMining\img\54.jpg)

现在的预训练模型从ELMO到switch transformer变得越来越大：

![image-20210704201750008](D:\develop\github\MLStudy\Deep Learning\textMining\img\53.jpg)

![image-20210704202007964](D:\develop\github\MLStudy\Deep Learning\textMining\img\55.jpg)

![image-20210704202127727](D:\develop\github\MLStudy\Deep Learning\textMining\img\56.jpg)

## 2、Self Supervised Learning：BERT

### 2.1 BERT的两个任务

#### 2.1.1 masking input

![image-20210704205806106](D:\develop\github\MLStudy\Deep Learning\textMining\img\57.jpg)

![image-20210704210003673](D:\develop\github\MLStudy\Deep Learning\textMining\img\58.jpg)

#### 2.2.2 Next Sentence Prediction

![image-20210704210239194](D:\develop\github\MLStudy\Deep Learning\textMining\img\59.jpg)

![image-20210704210800490](D:\develop\github\MLStudy\Deep Learning\textMining\img\60.jpg)

### 2.2 BERT的使用

![image-20210704211402801](D:\develop\github\MLStudy\Deep Learning\textMining\img\61.jpg)

![image-20210704211552773](D:\develop\github\MLStudy\Deep Learning\textMining\img\62.jpg)

![image-20210704213013761](D:\develop\github\MLStudy\Deep Learning\textMining\img\63.jpg)

![image-20210704213152103](D:\develop\github\MLStudy\Deep Learning\textMining\img\64.jpg)

![image-20210704213458564](D:\develop\github\MLStudy\Deep Learning\textMining\img\65.jpg)

![image-20210704213534004](D:\develop\github\MLStudy\Deep Learning\textMining\img\66.jpg)

![image-20210704213735900](D:\develop\github\MLStudy\Deep Learning\textMining\img\67.jpg)

![image-20210704213954307](D:\develop\github\MLStudy\Deep Learning\textMining\img\68.jpg)

 ![image-20210704215655532](D:\develop\github\MLStudy\Deep Learning\textMining\img\69.jpg)

![image-20210704215741071](D:\develop\github\MLStudy\Deep Learning\textMining\img\70.jpg)

![image-20210704215815248](D:\develop\github\MLStudy\Deep Learning\textMining\img\71.jpg)

![image-20210704215931864](D:\develop\github\MLStudy\Deep Learning\textMining\img\72.jpg)

BERT可以视作deep版的CBOW，同时BERT还可以做到根据不同的上下文，其Embedding不一样。

下面看一个比较奇怪的实验，把BERT用于蛋白质、DNA、音乐分类，看似用来训练的文本杂乱无章，但是仍旧可以得到不错的结果，所以只能说，我们以为BERT在五层，其实它可能在十层。

 ![image-20210704220602856](D:\develop\github\MLStudy\Deep Learning\textMining\img\73.jpg)

![image-20210704220739566](D:\develop\github\MLStudy\Deep Learning\textMining\img\74.jpg)

![image-20210704220806889](D:\develop\github\MLStudy\Deep Learning\textMining\img\75.jpg)

多语言BERT：

![image-20210704221240783](D:\develop\github\MLStudy\Deep Learning\textMining\img\76.jpg)

![image-20210704221401356](D:\develop\github\MLStudy\Deep Learning\textMining\img\77.jpg)

BERT在未见过的语言上也能获得比较好的预测结果。

![image-20210704221433248](D:\develop\github\MLStudy\Deep Learning\textMining\img\78.jpg)

![image-20210704221551006](D:\develop\github\MLStudy\Deep Learning\textMining\img\79.jpg)

如果说不同语言对应的单词在多语言BERT上可以获得相近的向量，但是在做预测的时候又不会把英文预测成中文，这说明BERT还是学到了不同语言之间的差异。

![image-20210704223541307](D:\develop\github\MLStudy\Deep Learning\textMining\img\80.jpg)

现在我们用中文词向量的平均值减去英文词向量的平均值，其差异认为是中英文之间的向量差，然后做如下实验：英文经过Multi-BERT转换后输出的结果，加上中英文的向量差，重构之后的单词竟然真的是同语义的中文，非常神奇。

![image-20210704223613452](D:\develop\github\MLStudy\Deep Learning\textMining\img\81.jpg)

![image-20210704223649126](D:\develop\github\MLStudy\Deep Learning\textMining\img\82.jpg)