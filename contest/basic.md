
### bfs和dfs应用场景
https://zhuanlan.zhihu.com/p/112594792



### 激活函数的种类与应用场景

激活函数为模型提供了非线性特性。激活函数主要有两大类，sigmoid和relu。
激活函数选择大致原则是： 非线性，可导，计算方便， 有效梯度区域广(在参数通常取值的区间，梯度不会过大过小)，激活后的取值范围合适。


sigmoid，能将 [-s, s]映射成(0,1)，有概率/置信度的物理意义。用在神经网络最后一层来做预测，或者浅层dnn的激活函数。或者用作门控函数。

relu的梯度计算要么是1要么是0，能减缓dnn反向传播时的梯度爆炸梯度消失问题，因此广泛应用在depth比较深的dnn中，在CNN中应用广泛。

tanh的值域[-1,1]，双曲正切函数tanh
在RNN和 Attention中有用到。


关于激活函数的选取，在LSTM中，遗忘门、输入门和输出门使用 Sigmoid函数作为激活函数;在生成候选记忆时，使用双曲正切函数tanh作为激活函数。值得注意的是，这两个激活函数都是饱和的也就是说在输入达到一定值的情况下，输出就不会发生明显变化了。如果是用非饱和的激活图数，例如ReLU，那么将难以实现门控的效果。



LSTM的三个门用的是sigmoid函数，生成候选记忆时用的才是tanh，门的激活函数如果用relu 的话，由于relu没有饱和区域，没法输出[0,1]区间。

候选记忆用tanh 是因为tanh 的输出是 -1~1，是0中心的，并且在0附近的梯度大，模型收敛快。

1）在生成候选记亿时，使用tanh函数，是因为其输出在-1-1之间，这与大多数场景下特征分布是0中心的吻合。
2）此外，tanh函数在输入为0近相比 Sigmoid函数有更大的梯度，通常使模型收敛更快。



LSTM中为何即存在sigmoid又存在tanh两种激活函数？
1) 为何用两种函数，即输入门输出门遗忘门所使用的激活函数，与求值(状态、输出)的激活函数不一样
2) 为何分别使用sigmoid和tanh函数


### attention与self-attention区别
attention机制来源于人类的视觉注意力机制。看东西时只会根据需求观察特定的一部分。

attention是一种权重参数分配机制。给定一组key，value，以及一个查询向量query，attention机制是通过计算query与每一组key的相似性，得到每个key的权重稀疏，再通过对value 的加权求和。

优点：
1）一步到位获取全局与局部的联系(rnn对于长度过长对丢失信息)
2）可以并行
3）相比CNN与RNN，参数少，模型复杂度低。

局限性：
1）没法捕捉位置信息，没有顺序信息。这点可以通过加入位置信息，如bert模型


应用场景：
NLP
1）bert
2）rnn机器翻译
3）普通transformers

CV

推荐系统
CTR预测


self-attention的 QKV都来自同一个输入。

attention应用场景：

self-attention应用场景：


### siam网络

contrastive loss = 1/2*D^2 + 1/2* {max( margin - D, 0)}^2



### attention in CNN

1) SENet (2017)
2) Non-local (2018)
3) CBAM 2018(Convolutional Block Attention Module) 

CNN中比较典型的两个attention应用。

SENet
SE block流程：
1) squeeze, 用global pooling压缩成C通道的
2) excitation，用两个全连接层实现，C通道变成 C/r, 再变会 C。r一般为16
3) reweight过程，用scale层把excitation层在c 通道上乘到原来的feature map上。如果在caffe需要写Axpby层。



non-local
主要思想也很简单，CNN中的 convolution单元每次只关注邻域 kernel size 的区域，就算后期感受野越来越大，终究还是局部区域的运算，这样就忽略了全局其他片区（比如很远的像素）对当前区域的贡献。

所以 non-local blocks 要做的是，捕获这种 long-range 关系：对于2D图像，就是图像中任何像素对当前像素的关系权值；对于3D视频，就是所有帧中的所有像素，对当前帧的像素的关系权值。


应用场景
Pros：non-local blocks很通用的，容易嵌入在任何现有的 2D 和 3D 卷积网络里，来改善或者可视化理解相关的CV任务。比如前不久已有文章把 non-local 用在 Video ReID [2] 的任务里。

Cons：文中的结果建议把non-local 尽量放在靠前的层里，但是实际上做 3D 任务，靠前的层由于 temporal T 相对较大，构造 [公式] 及点乘操作那步，超多的参数，需要耗费很大的GPU Memory~ 可后续改善


CBAM 2018
是在SENet上面的拓展。
有Chanel Attention module
有Spatial Attention module



Non-local block
1) 单一的non-local block加在浅层效果显著
   因为高层丢失的信息太多了。
2) 多个non-local block加入，有一定效果提升但不明显
3) 时空同时non-local比单一时间/ 单一空间维度效果要好(废话)
4) non-local 比三维cnn要好。(废话)

为啥不用non-local把卷积层全部替换掉？
不行的。要依赖小卷积去捕捉主体信息，同时用它的block捕捉全局信息，两者相辅相成才有好的效果。

在视频变长后，non-local的trick提升变小了。
这是由于，在时间维度上，这些短视频帧数太短， 时间维度上的小卷积得到的信息不足，劣势明显。 时间变长了，non-local 不能handle这么大的信息了，损失一些信息的小卷积反而不那么差劲了。





### 带孔卷积的作用
1) 不增加参数的情况下扩大感受野(这个功能也能通过maxpooling实现，但输出图大小变了)
2) 


带孔卷积与pooling最大的区别在于，带孔卷积保持了特征图相对的空间位置，而pooling引入了平移不变性。这是抽象和具体的矛盾。

分类模型需要 平移不变性来保证抽象不因空间位置的改变而改变，但分割则要求像素精确保持空间相对位置不变，属于具体精细的任务。
conv(dilated=2, s=2) 能在下采样时照顾到分割的上下文信息，所以带孔卷积在分割领域比较有效。




感受野计算公式
RF_i = ( RF_(i+1) -1 ) * s + k

卷积核k=3，s=1， p=1.

小卷积核的好处
1）连续的小卷积核，参数比大卷积核要少，效果上接近等效。当然前几层用大的卷积核是有一定效果的。



### 残差的公式以及为何能生效

因为CNN能够提取low/mid/high-level的特征，网络的层数越多，意味着能够提取到不同level的特征越丰富。并且，越深的网络提取的特征越抽象，越具有语义信息。

深度网络的后面那些层接近恒等映射。
直接让一些层去拟合一个潜在的恒等映射函数 H(x) = x 比较困难，这是深层网络难以训练原因之一。
如果把网络结构设计层 H(x) = F(x) + x, 转换为学习一个残差函数 F(x) = H(x) - x, 只要F(x)=0, 就构成一个恒等映射。拟合残差更加容易。

公式推导的时候可以写成
x_(l+1) = x_l + F(x_l, W_l)

x_L = x_l + \sum_{i=l}{L-1} F(x_i, W_i)

identity mapping指的是 弯弯的曲线
residual mapping，指的是直的部分。最后输出 F(x) + x



残差网络的定义，从输入直接引入一个短连接到非线性层的输出上。


x 
weightlayer+ relu
weightlayer+ relu
+x

非残差网络 G
残差网络 H = F(x,w) + x
假定输入x=1

在t时刻
G_t(1) = 1.1
H_t(1) = 1.1 = F_t(1) + 1, 故 F(t) = 0.1

在t+1时刻
G_t(1) = 1.2
H_(t+1)(1) = 1.2 = F_(t+1)(1) + 1, 故 F_(t+1)(1) = 0.2

非残差网络的梯度 G的梯度= (G_(t+1)(1) - G_t(1) )/ G_t(1) = (1.2 - 1.1)/1.1
非残差网络的梯度 F的梯度= (F_(t+1)(1) - F_t(1) )/ F_t(1) = (0.2 - 0.1)/0.1

因为两者各自是对G的参数和F的参数进行更新，可以看出变化对于F的影响远远大于G。说明引入残差后的映射对输出的变化更敏感，这样是有利于网络进行传播的。


残差块
1）残差网络学习的是信号的差值，
如果网络使用了残差结构，导数包含了恒等项，仍然能有效地反向传播。



### sgd，adam的区别


### alexnet, vgg16区别
alexnet
1)最早在imagenet上使用的cnn网络，5 conv + 3 full_connect, 开始采用relu激活函数。在每个全连接层后加上dropout 来减少模型的过拟合问题。
2）两块gpu，分组计算。提高计算效率。

一个卷积块包括一个conv + relu + maxpooling + normalization(LRN)。前面用的11*11，5*5卷积核。
3full_connect   (4096 - 4096 -1000)

vgg16，
这里的16指的是有参数的层数量，包括conv和 full_connect
1）用多个连续的3*3卷积核来代替alexnet中较大的卷积核。用标准卷积conv(k=3,s=1), maxpooling(s=2步长)

有五次max pooling。加三个full_connect, 这部分与上面一致。


GoogleNet
用Inception Module不同尺度的卷积核。
用global pooling大大减小了参数。

resnet引入残差模块，解决了depth太大无法训练的问题(梯度消失梯度爆炸)


resnet
1）网络较瘦，控制了参数量
2）层级明显，特征图个数逐层递进，保证输出特征表达能力
3）使用较少的池化层，大量使用下采样，提高传播效率
4）没有用dropout，利用 bn 和 global pooling来进行正则化，加快训练速度
5）层数较高时减少了 3*3 卷积个数，并用 1*1 卷积控制了 3*3 卷积的输入输出特征图数量，即bottleneck瓶颈结构。



### softmaxloss，focalloss的应用场景

ce = -logp              if y==1
     -(1-y)log(1-p)     if y==0


在交叉熵损失前面加个系数alpha来解决 正负样本不平衡
-alpha*logp   
-(1-alpha) * log(1-p)

对于难样本，正难/负难，正易/负易样本。alpha能平衡正负样本，但对难易样本的不平衡没有帮助。
难样本是 分类的p偏低的样本

所以focalloss把高置信度样本的损失降低一些
fl = -(1-p)^gamma * logp
     -p^gamma * log(1-p)

当gamma=2， p=0.968时， (1-0.968)^2 =0.001 损失衰减了1000倍。

再把alpha系数加上。

论文提到 alpha=0.25， gamma=2 实验结果最佳。


所以focalweight= alpha *(1-p)^gamma
                (1-alpha)*p^gamma




GHM是FocalLoss的改进。
focalloss是从置信度p的角度入手衰减loss，而GHM 是一定范围置信度p的样本数量的角度衰减loss。

GHM，我们不应该过多关注易分样本，但对于特别难分样本(outliers, 离群点)也不该关注。
引入梯度密度。


GHM-R 回归损失，是用的 smooth l1 loss，再 处于梯度密度。





### 参数量计算
跟输出通道数密切相关。max pooling/relu/flatten等都没有参数。

C2是输出通道数，C1是输入通道数。

caffe卷积层，Conv的w的shape 是 (C2, C1, K, K), b是 ()
innerproduct层 w是(C1, C2), b是(C2)


1 、卷积层参数个数计算方法：（卷积核高 * 卷积核宽 * C1通道数 + 1） * 卷积核个数C2
2 、当前全连接层参数个数计算方法： （上一层神经元个数 + 1） * 当前层神经元个数
以上的1代表偏置，因为每个神经元都有一个偏置

卷积层1： 320 = （3 * 3 * 1 +1） * 32
卷积层2： 18496 = （3 * 3 * 32 +1） * 64
卷积层3： 73856 = （3 * 3 * 64 +1） * 128

全连接层1： 8256 = （128 + 1） * 64
全连接层2： 650 = （64 + 1） * 10



### 平移不变性


个人理解有两点：
1、为classification的中，CNN主要贡献的特征提取能力，而不是不变性的能力，不变性可以靠降采样，全连接，甚至全局池化（大尺度的降采样）实现。


2、而且图像分类任务当中，物体在图像中的位移一般不大，而且在物体在图片中占比较大，所以在rcnn中，pre-trained CNN可以用于对proposal regions 进行特征提取（毕竟分割出来的proposal中，本身物体的占比就比较大）




### xgboost与gdbt区别

GBDT那些部分可以并行
1、计算每个样本的负梯度；
2、分裂挑选最佳特征及其分割点时，对特征计算相应的误差及均值时；
3、更新每个样本的负梯度时；
4、最后预测过程中，每个样本将之前的所有树的结果累加的时候。


GBDT与RF的区别
2、RF中树是独立的，相互之间不影响，可以并行；而GBDT树之间有依赖，是串行。
3、RF最终的结果是有多棵树表决决定，而GBDT是有多棵树叠加组合最终的结果。
4、RF对异常值不敏感，原因是多棵树表决，而GBDT对异常值比较敏感，原因是当前的错误会延续给下一棵树。
5、RF是通过减少模型的方差来提高性能，而GBDT是减少模型的偏差来提高性能的。（原理之前分析过）



gboost类似于gbdt的优化版，不论是精度还是效率上都有了提升。与gbdt相比，具体的优点有：
1.损失函数是用泰勒展式二项逼近，而不是像gbdt里的就是一阶导数
2.对树的结构进行了正则化约束，防止模型过度复杂，降低了过拟合的可能性
3.节点分裂的方式不同，gbdt是用的gini系数，xgboost是经过优化推导后的

### LR
logit回归

待学习的分类面建模的实际上是Logit[3]，Logit本身是是由LR预测的浮点数结合建模目标满足Bernoulli分布来表征的，数学形式如下：

y = wx = log(p/(1-p))
so p = 1/(1+ exp(-y))
我们实际上将模型的浮点预测值与离散分类问题建立起了联系。


分类问题的典型Loss建模方式是基于极大似然估计，具体到每个样本上，实际上就是典型的二项分布概率建模式[1]：
最大似然 = p^(y_i) * (1-p)^(1-y_i)

取对数, 再取反
loss = -yi* logp - (1-yi) * log(1-p)

     = yi* log(1+exp(-yi)) + (1-yi)*log(1+exp(yi))



### 为何LR适合处理高维稀疏特征，而gbdt不适合
主要原因有：
1、高维特征会导致gbdt运行过于耗时

2、从高维稀疏特征中难以进行有效的特征空间划分，且对噪音会很敏感。
想想一个例子，有个年龄特征0~100，如果对这样特征进行one-hot编码后变为稀疏特征，第i维表示是否为i岁。
如果将这种特征直接输入gbdt然后输出是否是青年人。很显然gbdt将变成枚举各个年龄是否为青年人。这类特征是非常容易过拟合的，如果当训练样本中存在一些噪声样本如80岁的青年人，如果在80岁没有足够的样本，这个错误将被gbdt学到。

而如果直接采用连续特征进行分类，gbdt会有更好的泛化性能。

3、高维稀疏特征大部分特征为0，假设训练集各个样本70%的特征为0，30%的特征非0。则某个维度特征在所有样本上也期望具有近似的取0的比例。当作分裂时，特征选择非常低效，特征只会在少部分特征取值非0的样本上得到有效信息。

而稠密向量可以得到样本集的整体特征信息。


至于LR为什么在高维稀疏特征上表现较好。我的理解是：
1、LR的目标就是找到一个超平面对样本是的正负样本位于两侧，由于这个模型够简单，不会出现gbdt上过拟合的问题。
2、高维稀疏特征是不是可以理解为低维的稠密特征映射到了高维空间。这里联想到了SVM的核技巧，不也是为了将特征由低维空间映射到高维空间中实现特征的线性可分吗？在SVM中已经证实了其有效性。这里面应该存在某种规律，LR在高维空间比低维空间中具有更高的期望实现更好分类效果的。
GBDT可以理解为将空间划分为离散块，每块染上深度不同的颜色。




高维稀疏特征的时候，使用 gbdt 很容易过拟合。
假设有1w 个样本， y类别0和1，100维特征，其中10个样本都是类别1，而特征 f1的值为0，1，且刚好这10个样本的 f1特征值都为1，其余9990样本都为0(在高维稀疏的情况下这种情况很常见)，我们都知道这种情况在树模型的时候，很容易优化出含一个使用 f1为分裂节点的树直接将数据划分的很好，但是当测试的时候，却会发现效果很差，因为这个特征只是刚好偶然间跟 y拟合到了这个规律，这也是我们常说的过拟合。但是当时我还是不太懂为什么线性模型就能对这种 case 处理的好？照理说 线性模型在优化之后不也会产生这样一个式子：y = W1*f1 + Wi*fi+….，其中 W1特别大以拟合这十个样本吗，因为反正 f1的值只有0和1，W1过大对其他9990样本不会有任何影响。
后来思考后发现原因是因为现在的模型普遍都会带着正则项，而 lr 等线性模型的正则项是对权重的惩罚，也就是 W1一旦过大，惩罚就会很大，进一步压缩 W1的值，使他不至于过大，而树模型则不一样，树模型的惩罚项通常为叶子节点数和深度等，而我们都知道，对于上面这种 case，树只需要一个节点就可以完美分割9990和10个样本，惩罚项极其之小.
这也就是为什么在高维稀疏特征的时候，线性模型会比非线性模型好的原因了：带正则化的线性模型比较不容易对稀疏特征过拟合。






### xgboost 迭代过程中，同一棵树上同一个特征会不会重复出现
会。挺常见。
为了达到最大增益， 每一次分裂都是对所有特征平等对待。

比对对于age这个属性来说，有的节点是 age>20 years old, 有的是age>40 years old


### xgboost，random forest在kaggle竞赛中为何胜出
1）理论模型（vc-dimension）
2）实际数据
3）系统实现（主要基于xgboost）
通常决定一个机器学习模型能不能取得好效果，以上三因素缺一不可。


1）理论模型（vc-dimension）
bias-variance tradeoff理论

统计机器学习里经典的 vc-dimension 理论告诉我们：一个机器学习模型想要取得好的效果，这个模型需要满足以下两个条件：
1. 模型在我们的训练数据上的表现要不错，也就是 trainning error 要足够小。
2. 模型的 vc-dimension 要低。换句话说，就是模型的自由度不能太大，以防overfit.




2）实际数据
除了理论模型之外, 实际的数据也对我们的算法最终能取得好的效果息息相关。kaggle 比赛选择的都是真实世界中的问题。所以数据多多少少都是有噪音的。

而基于树的算法通常抗噪能力更强。比如在树模型中，我们很容易对缺失值进行处理。

除此之外，基于树的模型对于 categorical feature 也更加友好。

除了数据噪音之外，feature 的多样性也是 tree-ensemble 模型能够取得更好效果的原因之一。

通常在一个kaggle任务中，我们可能有年龄特征，收入特征，性别特征等等从不同 channel 获得的特征。而特征的多样性也正是为什么工业界很少去使用 svm 的一个重要原因之一，因为 svm 本质上是属于一个几何模型，这个模型需要去定义 instance 之间的 kernel 或者 similarity （对于linear svm 来说，这个similarity 就是内积）。这其实和我们在之前说过的问题是相似的，我们无法预先设定一个很好的similarity。这样的数学模型使得 svm 更适合去处理 “同性质”的特征，例如图像特征提取中的 lbp 。

而从不同 channel 中来的 feature 则更适合 tree-based model, 这些模型对数据的 distributation 通常并不敏感。




3）系统实现（主要基于xgboost）
1. 正确高效的实现某种模型。我真的见过有些机器学习的库实现某种算法是错误的。而高效的实现意味着可以快速验证不同的模型和参数。
2. 系统具有灵活、深度的定制功能。
3. 系统简单易用。
4. 系统具有可扩展性, 可以从容处理更大的数据。到目前为止，xgboost 是我发现的唯一一个能够很好的满足上述所有要求的 machine learning package. 在此感谢青年才俊 陈天奇。

在效率方面，xgboost 高效的 c++ 实现能够通常能够比其它机器学习库更快的完成训练任务。
在灵活性方面，xgboost 可以深度定制每一个子分类器，并且可以灵活的选择 loss function（logistic，linear，softmax 等等）。

除此之外，xgboost还提供了一系列在机器学习比赛中十分有用的功能，例如 early-stop， cv 等等在易用性方面，

xgboost 提供了各种语言的封装，使得不同语言的用户都可以使用这个优秀的系统。

最后，在可扩展性方面，xgboost 提供了分布式训练（底层采用 rabit 接口），并且其分布式版本可以跑在各种平台之上，例如 mpi, yarn, spark 等等。

有了这么多优秀的特性，自然这个系统会吸引更多的人去使用它来参加 kaggle 比赛。





对于过拟合，boosting，random forest等集成学习方法并不比SVM、LR更容易过拟合。

kaggle比赛很多数据，对于数据特征与语义层次差别不太大(差别大的包括以像素表示的图像、以波形表示的语音等)，集成学习ensemble learning在这一类数据上常常有极佳表现的原因如下：

1）自适应非线性：随着决策树的生长，能够产生高度非线性的模型， 而SVM等线性模型的非线性化需要基于核函数等方法，要求在学习之前就定义好核函数。而确定合适的核函数是不容易的。

2）多分类器隐含正则：高度非线性的模型容易过拟合。因此几乎没人会使用单科决策树。boosting和random forest等集成学习方法都要训练多个有差异化的学习器，能够起到正则化的作用。有论文论证这一点。从而使得总体复杂度降低，泛化能力提高。

random forest这样并行集成方法，即使每一颗树都过拟合，直观来讲，总体投票或平均后并不会过拟合。



### svm的 hinge-loss

SVM是一种二分类模型，他的基本模型是定义在特征空间上的间隔最大的线性分类器.SVM的学习策略是间隔最大化.

间隔最大化（硬间隔）
分为硬间隔最大和软间隔最大
SVM的基本思想就是求解可以正确划分数据集并且几何间隔最大的分离超平面，其原因是线性可分超平面有无数个，但是间隔最大超平面是唯一的。
间隔最大化的意思就是以充分大的确信度对训练数据进行分类，也就是说，不仅将正负实例分开，同时对最难分的实例点（距离超平面最近的点）也有足够大的确信度将其分离。

与超平面最近的点被称为支持向量，也就是使得原始问题约束项成立的点。
实际上离超平面很远的点已经被正确分类，我们让它离超平面更远并没有意义。反而我们最关心是那些离超平面很近的点，这些点很容易被误分类。如果我们可以让离超平面比较近的点尽可能的远离超平面，那么我们的分类效果会好有一些

核函数
核函数本质不是将特征映射到高维空间，而是找到一种直接在低位空间对高维空间中向量做点积运算的简便方法。

为何将原始问题转为对偶问题
总是说对偶问题更容易求解，道理在哪呢？
之所以说换为对偶问题更容易求解，其原因在于降低了算法的计算复杂度。在原问题下，算法的复杂度与样本维度相关，即等于权重w的维度，而在对偶问题下，算法复杂度与样本数量有关，即为拉格朗日算子的个数。
因此，如果你是做线性分类，且样本维度低于样本数量的话，在原问题下求解就好了，Liblinear之类的线性SVM默认都是这样做的；但如果你是做非线性分类，那就会涉及到升维（比如使用高斯核做核函数，其实是将样本升到无穷维），升维后的样本维度往往会远大于样本数量，此时显然在对偶问题下求解会更好。

为什么SVM对缺失值敏感



为什么SVM的分割超平面方程为 wx + b = 0？
1）这个超平面的公式是假设。
2）其中w和x均为向量，b是一个实数。
3）在三维空间中一个法向量w，一个位移b能够唯一确定一个平面，因此作出如上公式假设。


为什么要设其=0？
为了方便，假设两类样本点的边界到超平面的距离是相等的，因此就设为0，这样的话，wx + b > 0就表示样本点在分割平面上方，wx + b < 0的话就代表在其下方

SVM与LR的联系
1）损失函数 SVM是hinge损失 LR是log损失 
2）输出 LR给出了后验概率 SVM只给出0或1，也就是属于哪一个类别 
3）异常值 LR对异常值敏感；SVM相对不敏感，泛化能力好 
4）训练集大小 较小的训练集更适合SVM
5）LR用到所有的样本点来寻找分割面；SVM只用到少数靠近支持面的几个点。 
6）非线性处理方式 LR靠特征组合高次项；SVM也可以组合，但更多使用核函数




Hinge_loss
1）实现了软间隔分类（这个Loss函数都可以做到）
2）保持了支持向量机解的稀疏性。
换用其他的Loss函数的话，SVM就不再是SVM了。

正是因为HingeLoss的零区域对应的正是非支持向量的普通样本，
从而所有的普通样本都不参与最终超平面的决定，这才是支持向量机最大的优势所在，
对训练样本数目的依赖大大减少，而且提高了训练效率。

hinge_loss =  [1- y(wx + b)]_+  + lambda*||w||^2

只有(xi, yi)被正确分类，且函数间隔 y(wx+b)大于1时，损失是0. 否则损失是 1-y(wx+b)



