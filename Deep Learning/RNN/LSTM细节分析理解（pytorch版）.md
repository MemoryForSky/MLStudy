# LSTM细节分析理解（pytorch版）

虽然看了一些很好的blog了解了LSTM的内部机制，但对框架中的lstm输入输出和各个参数还是没有一个清晰的认识，今天打算彻底把理论和实现联系起来，再分析一下pytorch中的LSTM实现。

先说理论部分。[一个非常有名的blog](https://link.zhihu.com/?target=http%3A//colah.github.io/posts/2015-08-Understanding-LSTMs/)把原理讲得很清楚，推荐参考。总之就是这些公式：

![img](https://pic4.zhimg.com/80/v2-9d56a2693e0b45f3b4690e074f683537_720w.jpg)


简单来说就是，LSTM一共有三个门，输入门，遗忘门，输出门， ![[公式]](https://www.zhihu.com/equation?tex=i%2Cf%2Co) 分别为三个门的程度参数， ![[公式]](https://www.zhihu.com/equation?tex=g) 是对输入的常规RNN操作。公式里可以看到LSTM的输出有两个，细胞状态 ![[公式]](https://www.zhihu.com/equation?tex=c%27) 和隐状态 ![[公式]](https://www.zhihu.com/equation?tex=h%27) ，![[公式]](https://www.zhihu.com/equation?tex=c%27)是经输入、遗忘门的产物，也就是当前cell本身的内容，经过输出门得到![[公式]](https://www.zhihu.com/equation?tex=h%27)，就是想输出什么内容给下一单元。

那么实际应用时，我们并不关心细胞本身的状态，而是要拿到它呈现出的状态![[公式]](https://www.zhihu.com/equation?tex=h%27)作为最终输出。以pytorch中的LSTM为例：

**torch.nn.LSTM(\*args,** kwargs)**

官方API：
[https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%3Fhighlight%3Dlstm%23torch.nn.LSTM)

- 参数
  – **input_size**
  – **hidden_size**
  – **num_layers**
  – **bias**
  – **batch_first**
  – **dropout**
  – **bidirectional**
- 输入
  – **input** (seq_len, batch, input_size)
  – **h_0** (num_layers * num_directions, batch, hidden_size)
  – **c_0** (num_layers * num_directions, batch, hidden_size)
- 输出
  – **output** (seq_len, batch, num_directions * hidden_size)
  – **h_n** (num_layers * num_directions, batch, hidden_size)
  – **c_n** (num_layers * num_directions, batch, hidden_size)

用起来很简单，当作黑箱时只要设置参数让它输出我们想要的shape就行了，但这些参数好像很难和前面公式里的那些联系起来，不便于理解和灵活使用。

先看一张很好的图（[LSTM神经网络输入输出究竟是怎样的？ - Scofield的回答 - 知乎](https://www.zhihu.com/question/41949741/answer/318771336)）：

![img](https://pic2.zhimg.com/80/v2-9e99c9b75bc4a23f207d460934937c95_720w.jpg)


这张图是以MLP的形式展示LSTM的传播方式（不用管左边的符号，输出和隐状态其实是一样的），方便理解hidden_size这个参数。其实hidden_size在各个函数里含义都差不多，就是参数W的第一维（或最后一维）。那么对应前面的公式，hidden_size实际就是以这个size设置所有W的对应维。

再看另一张很好的图（[https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714](https://link.zhihu.com/?target=https%3A//medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714%29)）：

![img](https://pic4.zhimg.com/80/v2-ebf8cd2faa564d9d80a958dcf25e6b3b_720w.jpg)


这张图非常便于理解参数num_layers。实际上就是个depth堆叠，每个蓝色块都是LSTM单元，只不过第一层输入是 ![[公式]](https://www.zhihu.com/equation?tex=x_t%2Ch_%7Bt-1%7D%5E%7B%280%29%7D%2Cc_%7Bt-1%7D%5E%7B%280%29%7D) ，中间层输入是 ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bt%7D%5E%7B%28k-1%29%7D%2Ch_%7Bt-1%7D%5E%7B%28k%29%7D%2Cc_%7Bt-1%7D%5E%7B%28k%29%7D) 。

剩下的参数就比较好理解了，input_size即输入的隐层维度，比embedding_dim。batch_first，第一维是否是batch，为什么要设置这个参数后面再说。bidirectional，是否为双向LSTM。

接下来看一下输入输出。关于input，API中提到也可以是一个packed变量序列，这个后面再讲。输入输出中h和c的shape (num_layers * num_directions, batch, hidden_size) 也是个容易困惑的点，但有了上面那张图就好说多了。绿色块 ![[公式]](https://www.zhihu.com/equation?tex=h_n%2Cc_n) 即长度为n的序列的最终输出，可以看出是所有depth输出的拼接，维度是num_layers。双向LSTM情况，相当于有两个图中的网络，只不过输入颠倒过来了，再将这两个最终隐状态拼接起来，维度num_layers*2。

> PS：感谢评论区纠正，下面这两段有修改，重新梳理了一下思路。（细节啊细节

最后看一下输出output。初学时看别人的代码，总是搞不清到底是取output还是用 ![[公式]](https://www.zhihu.com/equation?tex=h_n) ，怎么用的都有。其实从图中可以看到，**output就是最后一个layer上，序列中每个时刻（横向）状态h的集合（若为双向则按位置拼接，输出维度2\*hidden_size），而![[公式]](https://www.zhihu.com/equation?tex=h_n)实际上是每个layer最后一个状态（纵向）输出的拼接。**

也就是说，对于单向LSTM来说，![[公式]](https://www.zhihu.com/equation?tex=h_n%5B-1%2C%3A%2C%3A%5D)就是 ![[公式]](https://www.zhihu.com/equation?tex=output%5B-1%2C%3A%2C%3A%5D) ，相当于序列最后一个时间步的输出。如果使用LSTM的目的是得到整个序列的embedding，与序列长度无关，由于LSTM具有序列信息传递性，因此一般可以取![[公式]](https://www.zhihu.com/equation?tex=h_n%5B-1%2C%3A%2C%3A%5D)当作序列embedding。但双向LSTM推广后，每个时间步的隐层输出都可以作为当前词的一个融合了上下文的embedding，因此BiLSTM可以视为一种词级别的encoder方法，得到的output既可以用于词级别的输出拼接，也可以进行融合（比如attention加权求和、pooling）得到序列级的输出。

理论终于和实践联系起来了，下面来具体分析一下pytorch的LSTM实现。

## pytorch的LSTM

**1、torch.nn.LSTMCell(input_size, hidden_size, bias=True)**

官方API：[https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTMCell](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%3Fhighlight%3Dlstm%23torch.nn.LSTMCell)**

一个LSTM单元。相当于一个time step的处理，应该是对应TensorFlow里类似的实现。基本不用。

**2、torch.nn.LSTM(\*args,** kwargs)**

官方API：[https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%3Fhighlight%3Dlstm%23torch.nn.LSTM)

前面基本讲得差不多了，只剩下两处：参数batch_first和input的packed variable length sequence。

为什么要有batch_first这个参数呢？常规的输入不就是(batch, seq_len, hidden_size)吗？而且参数默认为False，也就是它鼓励你第一维不是batch，更奇怪了。

取pytorch官方的一个tutorial（[chatbot tutorial](https://link.zhihu.com/?target=https%3A//pytorch.org/tutorials/beginner/chatbot_tutorial.html)）中的一个图：

![img](https://pic4.zhimg.com/80/v2-0e5ed63bd017844570649d4175f3b527_720w.jpg)


左边是我们的常规输入（先不考虑hidden dim，每个数字代表序列中的一个词），右边是转置后，第一维成了max_length。我们知道在操作时第一维一般可视为“循环”维度，因此左边一个循环项是一个序列，无法同时经LSTM处理，而右边跨batch的循环项相当于当前time step下所有序列的当前词，可以并行过LSTM。（当然不管你是否batch_first它都是这么处理的，这个参数应该只是提醒一下这个trick）

### pack&pad

（感觉说起来没那么简单，所以加了个小标题。）

前面说过nn.LSTM的输入也可以是“packed”形式，那么这是个什么形式？

先不问为什么，看一下pack和pad的操作是怎样的。

**torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=True)**

官方API：[https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.utils.rnn.pack_sequence](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%3Fhighlight%3Dlstm%23torch.nn.utils.rnn.pack_sequence)

这是pack操作，输入的sequences是tensor组成的list，要求按长度从大到小排序。官网的例子：

![img](https://pic4.zhimg.com/80/v2-5a287f507f12eac67db3dff6340f678b_720w.jpg)



**torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0)**

官方API：[https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.utils.rnn.pad_sequence](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%3Fhighlight%3Dlstm%23torch.nn.utils.rnn.pad_sequence)

这是pad操作，sequences也是list。这个比较好理解，就是给list里的tensor都用padding_value来pad成最长的长度，并组合成一个tensor：



![img](https://pic4.zhimg.com/80/v2-77d3332ded97e0850ad5239dcf7dc7bb_720w.jpg)


看了这两个操作，隐隐约约和前面的LSTM联系起来了。我们知道一个batch里的序列不一定等长，需要pad操作用0把它们都填充成max_length长度。但前面说了LSTM的一次forward对应一个time step，接收的是across batches的输入，这就导致短序列可能在当前time step上已经结束，而你还是在给它输入东西（pad），这就会对结果产生影响（可以对照公式看看，即便输入全0还是会有影响）。我们想要的效果是，LSTM知道batch中每个序列的长度，等到某个序列输入结束后下面的time step就不带它了。

传统的pad不能用，LSTM需要一种其它的方法来处理变长输入。这时我们观察刚看到的pack操作，感觉终于明白了它的道理。官方的例子有点混淆，我写了一个更直观的：

![img](https://pic3.zhimg.com/80/v2-ee9a5600a993d50a8ce3eaa4c6a47a26_720w.jpg)


把这个例子看成是LSTM处理一个batch的过程，注意看成转置的形式，即batch_first=False，也就是[4,1,9]是第一个序列，[5,2]是第二个序列…max_length=3，batch_size=5。从输出可以看出其实是一个很简单的过程，有点像稀疏矩阵的存储方法，先都塞到一起再记录位置（这里是长度）。

这两个函数都是基本操作，一般不会直接使用。常用的是下面这两个：

**torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)**

官方API：[https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.utils.rnn.pack_padded_sequence](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%3Fhighlight%3Dlstm%23torch.nn.utils.rnn.pack_padded_sequence)

顾名思义，pack一个经过pad的sequence，因为我们一般在处理数据时就已经将序列pad成等长了。lengths即为序列的长度。



**torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None)**

官方API：[https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.utils.rnn.pad_packed_sequence](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%3Fhighlight%3Dlstm%23torch.nn.utils.rnn.pad_packed_sequence)

这是上面函数的逆操作，再pad回去供后续使用。这里的total_length是个很实用的参数，在下面的例子中可以看到。



一个完整的例子：

![img](https://pic3.zhimg.com/80/v2-2094341be3d93c998405651471562f8a_720w.jpg)



![img](https://pic2.zhimg.com/80/v2-beaf0bee6170e93fa1bda77cb27e9b85_720w.jpg)



![img](https://pic1.zhimg.com/80/v2-d340837a5fc4d8282fdfe5f6373735a4_720w.jpg)



```python
import torch as t
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

a = t.tensor([[1,2,3],[6,0,0],[4,5,0]]) #(batch_size, max_length)
lengths = t.tensor([3,1,2])

# 排序
a_lengths, idx = lengths.sort(0, descending=True)
_, un_idx = t.sort(idx, dim=0)
a = a[idx]

# 定义层 
emb = t.nn.Embedding(20,2,padding_idx=0) 
lstm = t.nn.LSTM(input_size=2, hidden_size=4, batch_first=True) 

a_input = emb(a)
a_packed_input = t.nn.utils.rnn.pack_padded_sequence(input=a_input, lengths=a_lengths, batch_first=True)
packed_out, _ = lstm(a_packed_input)
out, _ = pad_packed_sequence(packed_out, batch_first=True)
# 根据un_idx将输出转回原输入顺序
out = t.index_select(out, 0, un_idx)
```

上面便是常用的使用方法（个人认为完全可以封装到LSTM函数里，不知道为什么要这么设计）。但此时假设另一个batch，b：

```python
# b是另一个batch
b = t.tensor([[7,8,0],[9,0,0],[10,0,0]])
```

batch中的最大长度为2，而对于整个数据流来说max_length=3，这就导致b经LSTM后pad的结果与整体的长度不匹配，此时设置pad_packed_sequence的total_length=3即可。