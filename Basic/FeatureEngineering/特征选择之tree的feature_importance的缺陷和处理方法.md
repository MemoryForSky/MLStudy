# 特征选择之tree的feature_importance的缺陷和处理方法

特征重要性的缺陷，rf和gbdt都存在类似的问题，以下均为针对rf或者gbdt讨论，lr的情况不一样不可一概而论。总结一下：

**1、受到噪声的影响**，很多时候我们往原始的特征矩阵加入许多纯粹的random里的高斯随机数，或者是泊松分布随机数，得到的特征重要性经常会高于部分原始特征，这从先验上来说是很不合理的，这里实际上我们需要去定义什么是噪声，假设某个加入的随机特征在训练集和测试集上的分布非常一致并且加入之后提高了模型的训练误差和测试误差的表现，那么实际上这个特征就不能算是噪声，首先，它在未知数据上是稳定的，其次，它对总体的预测存在很好的贡献，换一句话说，假设某一个有明确业务意义的特征加入原始的特征矩阵之后，带来了好的影响，但是它的分布差异性非常大，那么这样的特征是不能直接入模使用的，这涉及到了特征稳定性的问题，如果某个特征的区分能力很强，但是在分布上非常的混乱，常见的情形就是训练集和测试集的评价函数上很大的差异性，并且通过调整正则化参数对于测试集的影响较小甚至毫无影响,举个例子：

这是kaggle ieee的真实数据，一开始使用了大佬参数的情况下的训练过程：

[200] training's auc: 0.95087 valid_1's auc: 0.907384
[400] training's auc: 0.980323 valid_1's auc: 0.926144
[600] training's auc: 0.992067 valid_1's auc: 0.935025
[800] training's auc: 0.996941 valid_1's auc: 0.93877
[1000] training's auc: 0.998826 valid_1's auc: 0.940702
[1200] training's auc: 0.999535 valid_1's auc: 0.941806
[1400] training's auc: 0.999802 valid_1's auc: 0.94262
[1600] training's auc: 0.99992 valid_1's auc: 0.942825
[1800] training's auc: 0.999969 valid_1's auc: 0.943064

然后把max_depth等正则化参数增大：

[200] training's auc: 0.952427 valid_1's auc: 0.905396
[400] training's auc: 0.979786 valid_1's auc: 0.924007
[600] training's auc: 0.991654 valid_1's auc: 0.932671
[800] training's auc: 0.996667 valid_1's auc: 0.936853
[1000] training's auc: 0.998688 valid_1's auc: 0.938653
[1200] training's auc: 0.99948 valid_1's auc: 0.939555
[1400] training's auc: 0.999783 valid_1's auc: 0.939761
[1600] training's auc: 0.999912 valid_1's auc: 0.94006
[1800] training's auc: 0.999964 valid_1's auc: 0.94031

其实我尝试了应该有四五次，每一次都是按照固定的比例增大一些起到约束作用的参数，但是得到的结果就是测试集上的分数从不上升，甚至因为模型的约束而得分下降，而训练集的分数一直都是冲着1 去的，也就是说特征矩阵中存在许多特征属于噪声，单纯只能提高在训练集上的精度而对于测试集的精度毫无作用，如果噪声在训练集上的作用还非常大，那么泛化误差会变得更大，比如这个数据集中原来有一个特征是时间序列数据，表示用户的申请时间节点之类的含义，它的分布就是一条有序的斜线：

![img](https://pic4.zhimg.com/80/v2-407826da61155b6f58484dca98983847_720w.jpg)



很明显这种特征在使用上非常困难，虽然经过重要性分析这个特征的重要性非常高，但是显然这个特征的未来数据的取值范围完全就不在这个特征的数据范围里，那么如果是用xgb或者lgb在预测未来数据的时候，**在分裂这个特征的时候所有的未来数据都会被分到infinite的那个取值范围内**。这已经不是分布偏移的问题了，这是特征完全改变的问题，所以这样的特征虽然对于训练集的拟合贡献很大，但是它完全就属于噪声的范畴。



2、**受到特征共线性（相关性很高）的影响**。举一个最简单直观的例子吧：

```text
##注意设置随机种子，避免抽样误差的问题
import os
import random
import numpy as np
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
SEED = 42
seed_everything(SEED) 



import pandas as pd
from sklearn.datasets import load_iris
import lightgbm as lgb
X=load_iris().data
X=pd.DataFrame(X)
y=pd.Series(load_iris().target)
clf=lgb.LGBMClassifier(colsample_bytree=0.3,bagging_seed=123,random_state=1234)
clf.fit(X,y)
feature_importance=pd.DataFrame()
feature_importance['features']=X.columns
feature_importance['feature_importances']=clf.feature_importances_

print(np.mean(cross_val_score(estimator=clf,X=X,y=y,scoring='accuracy',cv=StratifiedKFold(5,random_state=123))))
```

![img](https://pic3.zhimg.com/80/v2-0d1a37a5fa2d319c07b091814fa8c7e2_720w.jpg)



![img](https://pic2.zhimg.com/80/v2-ebaaff90be1d9100457eab9e921c5e25_720w.jpg)

可以看到特征重要性整体比较平均，都很重要，其中特征2最重要。accuracy如上图



接着：

```text
X['the_same_as_2_1']=X[2].values
X['the_same_as_2_2']=X[2].values
X['the_same_as_2_3']=X[2].values
X['the_same_as_2_4']=X[2].values
X['the_same_as_2_5']=X[2].values
X['the_same_as_2_6']=X[2].values
X['the_same_as_2_7']=X[2].values
X['the_same_as_2_8']=X[2].values
X['the_same_as_2_9']=X[2].values
X['the_same_as_2_10']=X[2].values
X['the_same_as_2_11']=X[2].values
X['the_same_as_2_12']=X[2].values
X['the_same_as_2_13']=X[2].values
X['the_same_as_2_14']=X[2].values
X['the_same_as_2_15']=X[2].values
X['the_same_as_2_16']=X[2].values
X['the_same_as_2_17']=X[2].values
X['the_same_as_2_18']=X[2].values
X['the_same_as_2_19']=X[2].values
X['the_same_as_2_20']=X[2].values
X['the_same_as_2_21']=X[2].values
X['the_same_as_2_22']=X[2].values
X['the_same_as_2_23']=X[2].values



clf=lgb.LGBMClassifier(colsample_bytree=0.3)
clf.fit(X,y)
feature_importance=pd.DataFrame()
feature_importance['features']=X.columns
feature_importance['feature_importances']=clf.feature_importances_

print(np.mean(cross_val_score(estimator=clf,X=X,y=y,scoring='accuracy',cv=StratifiedKFold(5))))
```

![img](https://pic2.zhimg.com/80/v2-8384e0b1a2672f11e0dee5a5031c2d5d_720w.jpg)

![img](https://pic3.zhimg.com/80/v2-845a5b54385833d15890030422d8cb86_720w.png)



**我们固定了随机种子保证结果的可复现并且不受随机性的影响**，我们考虑了最极端的情况，新增的特征和特征2完全相同，我们可以看到，经过了列采样之后，相关性很高的新增特征也出现了特征重要性，而其它特征的重要性都得到了削弱。新增的完全相关的特征的重要性甚至出现了超过其它特征的情况。很明显这是非常不合理的，而从泛化性能上来看，整体logloss增大，模型的效果变差了。

这是因为相关性高的特征常常携带了类似的相同的信息，在这类相似的特征上训练虽然可以在训练集上快速减少损失函数的值，但是对于未来的数据预测帮助不大，并且会使得模型非常依赖于这类特征，从而在其它携带额外重要信息的特征在树的分裂过程中被使用的次数变少。

换个好理解的说法，本来我们使用列采样希望每一棵树都能在不同的特征子集上训练出不同的子树，但是由于大量的相似特征的存在使得这一类特征被采样到的几率变大，从而使得子树中有很多子树都是在这类特征上训练，这样实际上是变相削弱了列采样的约束能力。

**3、feature importance 并不能给出特征重要性的阈值**，多大阈值的特征应该删除，多大阈值的特征应该保留是没有明确结论的，这一块基本是主观判断为主；

4、无法表现特征与标签之间的相互关系，可解释性问题。

------

针对于第一个噪声的问题，有permutation importance。原理见：

[kaggle | Machine Learning for Insights Challengeyyqing.me![图标](https://pic3.zhimg.com/v2-b75b2755edeb2b0587532e62705ddbf6_ipico.jpg)](https://link.zhihu.com/?target=https%3A//yyqing.me/post/2018/2018-09-25-kaggle-model-insights)

```text
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```



![img](https://pic2.zhimg.com/80/v2-77e29c5f78e8998dad706d276f717625_720w.jpg)

eli5中已经实现过了，不过发现了一些坑要注意一下。

![img](https://pic3.zhimg.com/80/v2-94fe10a012c56f03c93b296f99f991aa_720w.jpg)







下面我们在原来的代码中做测试：

```text
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,StratifiedKFold

import lightgbm as lgb
X=load_iris().data
X=pd.DataFrame(X)
y=pd.Series(load_iris().target)
clf=lgb.LGBMClassifier(colsample_bytree=0.3)
clf.fit(X,y)
feature_importance=pd.DataFrame()
feature_importance['features']=X.columns
feature_importance['feature_importances']=clf.feature_importances_

print(np.mean(cross_val_score(estimator=clf,X=X,y=y,scoring='neg_log_loss',cv=StratifiedKFold(5))))


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(clf, random_state=123,cv='prefit',n_iter=5,scoring='accuracy').fit(X, y)
print(perm.feature_importances_)
```

![img](https://pic2.zhimg.com/80/v2-4fbc7280e296be277271c864a2d04bb9_720w.png)

我们每次选择一个特征，进行5次shuffle，分别进行5次预测，然后将得到的accuracy与原来的accuracy做比较，最后取变化的平均值，可以看到，初始的4个特征在shuffle之后的accuracy都要比原来的accuracy来的差，注意，最后结果为正表示accuracy之后的效果变差，为负则表示为shuffle之后效果反而变好，这个效果根据不同的评价指标来看的。

然后我们来看看相关性的特征的影响：

```text
X['the_same_as_2_1']=X[2].values
X['the_same_as_2_2']=X[2].values
X['the_same_as_2_3']=X[2].values
X['the_same_as_2_4']=X[2].values
X['the_same_as_2_5']=X[2].values
X['the_same_as_2_6']=X[2].values
X['the_same_as_2_7']=X[2].values
X['the_same_as_2_8']=X[2].values
X['the_same_as_2_9']=X[2].values
X['the_same_as_2_10']=X[2].values
X['the_same_as_2_11']=X[2].values
X['the_same_as_2_12']=X[2].values
X['the_same_as_2_13']=X[2].values
X['the_same_as_2_14']=X[2].values
X['the_same_as_2_15']=X[2].values
X['the_same_as_2_16']=X[2].values
X['the_same_as_2_17']=X[2].values
X['the_same_as_2_18']=X[2].values
X['the_same_as_2_19']=X[2].values
X['the_same_as_2_20']=X[2].values
X['the_same_as_2_21']=X[2].values
X['the_same_as_2_22']=X[2].values
X['the_same_as_2_23']=X[2].values



clf=lgb.LGBMClassifier(colsample_bytree=0.3,max_depth=3)
clf.fit(X,y)

perm = PermutationImportance(clf, random_state=123,cv='prefit',n_iter=5,scoring='accuracy').fit(X, y)
print(perm.feature_importances_)
```

![img](https://pic1.zhimg.com/80/v2-328c89b4708d0b40cff7b7a8659a1074_720w.jpg)

可以看到，permutation对于相关性高的问题会比较头大，而且相关性多的话反过来会影响到permutation的结果，因为特征之间的替代性非常强，所以其中一个shuffle之后，对于整体来说影响不大，因为其它相似的特征提供了级别相同的信息，所以没有办法真实反应permutation的效果，**所以实际上在建模之前，无论是什么算法，都建议最好走一边遍相关性分析，免得后续各种麻烦的问题。**







接着：我们把数据还原，加入噪声之后看原始的lgb输出的特征重要性

```python
X['noise1']=np.random.rand(150) ##标准正太分布
X['noise2']=np.random.uniform(size=150) ##均匀分布
clf=lgb.LGBMClassifier(colsample_bytree=0.3)
clf.fit(X,y)
feature_importance=pd.DataFrame()
feature_importance['features']=X.columns
feature_importance['feature_importances']=clf.feature_importances_
```

![img](https://pic4.zhimg.com/80/v2-7c2131638478587a2cce51b653a7785b_720w.jpg)



可以看到,noise的特征重要性甚至要高于普通的特征。。。。这也是为什么维度越高越容易过拟合的最接地气最直接的解释，那就是维度越高遭遇到噪声的可能性越大，但是如果每一个维度都非常有意义，那么维度高低和过拟合并不存在什么必然联系，简单说就是维度越高越可能过拟合，但并不是绝对。

然后呢，我们来看下eli5

```text
perm = PermutationImportance(clf, random_state=123,cv='prefit',n_iter=5,scoring='accuracy').fit(X, y)
print(perm.feature_importances_)
```

![img](https://pic3.zhimg.com/80/v2-c9153ea78832a91d009a752ad3206d1e_720w.png)

很尴尬，我们并不一定能检测出噪声。。。特别是对于数量量小的问题，会有各种麻烦的问题，尤其是模型的拟合能力还很强大的情况下。



我们改变一下参数使用更加严格的检测,顺便熟悉下api：

```text
perm = PermutationImportance(clf, random_state=123,cv=None,n_iter=5,scoring='accuracy').fit(X, y)
print(perm.feature_importances_)
```

结果都是一样的，这里实际上就是重新fit了一下模型然后进行cv='prefit'的操作,也就是shuffle5次predict然后计算偏差，而已所以在随机性固定的情况下结果完全一样，

```text
from sklearn.model_selection import StratifiedKFold
perm = PermutationImportance(clf, random_state=123,cv=StratifiedKFold(5),n_iter=5,scoring='accuracy').fit(X, y)
print(perm.feature_importances_)
```

![img](https://pic1.zhimg.com/80/v2-d4b99bca2ebcf589b73cc064ad2caf28_720w.png)

这里实际上是做交叉验证，然后在每一个验证里进行n_iter=5次的shuffle，计算出shuffle前后的差异的均值，最后计算所有结果的均值，就是训练5个模型，每个模型shuffle预测5次，一共有25次的预测，可以看到相对来说，原来的不利的情况缓解了一点点，但是这个方法在实际上很难应用，运算量太高了，一般还是用cv='prefit'的的方式会比较方便迅速一些，即不重新训练模型，仅仅在训练阶段shuffle 特征然后比较结果。

## **但是面对维度特别高的情况下，permutation整体的运行效率都是非常的低下的。如果你有100个特征，每次shuffle5次，则一共要预测500次，时间开销上非常大。所以在实际业务上除非特征比较少，否则用起来真的费时间。**



------

介于这样的性能问题，即使打比赛的时候用这样的方法也是非常的费时间的，不过kaggle上的大佬很快想到了一些别的处理方法，比如比较出名的null importance还有boruta。写多了，休息。。。