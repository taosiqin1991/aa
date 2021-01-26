
### backbone选择

分类的典型网络
mobilev1/v3
res50/101
efficientnet
darknet53

https://github.com/PaddlePaddle/PaddleDetection

目标检测loss有 smooth-l1，giou, diou, ciou, iouaware



### OCR成熟的方案是什么
文本检测- 文字识别- 词典匹配

文本检测有FasterRCNN，EAST, CTPN. 
其中EAST对倾斜文本效果不错，CPTN加入了BLSTM可以得到上下文信息，在噪声干扰上更好。整体CTPN更好一些。

文字识别FasterRCNN，CRNN，seq2seq。胜者 CRNN
CRNN和seq2seq两种算法支持单词级标注，标注难度要小很多，而FasterRCNN只能是字符级的标注。

整体方案
FasterRCNN + 模板匹配
EAST/CTPN + CRNN


paddlepaddle中提到的方案
识别单行英文
CRNN-CTC
CNN + seq2seq + attention
OCR-attention 比上文更好


Attention OCR使用了Cascade Mask RCNN作为检测框架
Attention OCR使用的识别算法是Attention LSTM




### 目标检测方案是什么
目标检测任务的目标是给定一张图像或是一个视频帧，让计算机找出其中所有目标的位置，并给出每个目标的具体类别。


faster-rcnn的框架
fpn检测多尺度目标或者小目标
yolov2 追求实时
rpn 能检出目标，早期版本后被舍弃。

backbone选择 res50，res101


### 人间检测方案
数据集 widerface
Faceboxes, 经典模型
BlazeFace, 移动端高速模型

PyramidBox, 百度方案


### 推断加速方案是什么
英伟达显卡1080/2080/3080，cuda库，加tensorrt库，tensorrt5，fp16和int8 推断
寒武纪显卡，思元100/思元270

cpu推断
英伟达家 tx2 推断


### 分割的方案是什么
数据集 cityscapes


deeplabv3+ 高精度方案
icnet  兼顾实时和准确率的方案
backbone + 上采样 + focal_loss, 普通方案

backbone选择： mobilenet 和 resnet，侧重轻量级和高精度。


### 比对检测方案是什么


### 多分类方案是什么
二分类
多分类的每类回归分类，mask-rcnn的设计(优点便于灵活调整每类阈值参数)


### 图像描述生成的方案是什么
图像描述（Image Caption），另一个叫视觉问答（Visual question answering，VQA


encoder-decoder结构, cnn + lstm


https://zhuanlan.zhihu.com/p/52499758
图像描述评价指标



### NLP核心，语义表示
BERT
GPT-2

BERT, 一个迁移能力很强的通用语义表示模型， 以 Transformer 为网络基本组件，以双向 Masked Language Model和 Next Sentence Prediction 为训练目标，通过预训练得到通用语义表示，再结合简单的输出层，应用到下游的 NLP 任务，在多个任务上取得了 SOTA 的结果。


文本生成
seq2seq

文本相似度计算
simnet





### 视频分类\动作定位\摘要生成\目标跟踪
视频数据包含语音、图像等多种信息，因此理解视频任务不仅需要处理语音和图像，还需要提取视频帧时间序列中的上下文信息。 

视频分类模型提供了提取全局时序特征的方法，主要方式有卷积神经网络 (C3D, I3D, C2D等)，神经网络和传统图像算法结合 (VLAD 等)，循环神经网络等建模方法。


视频动作定位模型需要同时识别视频动作的类别和起止时间点，通常采用类似于图像目标检测中的算法在时间维度上进行建模。 

视频摘要生成模型是对视频画面信息进行提取，并产生一段文字描述。

视频查找模型则是基于一段文字描述，查找到视频中对应场景片段的起止时间点。

这两类模型需要同时对视频图像和文本信息进行建模。 

目标跟踪任务是在给定某视频序列中找到目标物体，并将不同帧中的物体一一对应，然后给出不同物体的运动轨迹，目标跟踪的主要应用在视频监控、人机交互等系统中。

跟踪又分为单目标跟踪和多目标跟踪，当前在飞桨模型库中增加了单目标跟踪的算法。主要包括Siam系列算法和ATOM算法。




non-local, 视频分类, 视频非局部 关联建模模型,kinectics-400, top1=74%
(视频时空建模)

attention-lstm, 视频分类, 常用模型, 速度快精度高, youtube-8m, gap=86%

BMN, 2019年视频动作定位夺冠方案,activitynet1.3 dataset, auc=67%

ETS, 2015,视频描述, 视频摘要生成基准模型,activitynet captions,meteror 10.0

tall, 2017视频查找,多模态时许回归定位方案.

siamfc, 视频跟踪,,vot2018, eao=0.21
atom, 视频跟踪,voc2018, eao=0.40






### 图像生成
没有提供指标

WGAN，是GAN训练稳定的一种办法
DCGAN，将GAN与CNN结合起来，以解决GAN训练不稳定问题
CycleGAN，可以训练非成对的图片，风格迁移

### 图像相似度衡量
ssim方案, 鲁棒性高.


### 视频跟踪
SiamMask




### 3d视觉
pointNet++
pointRCNN



### 人体关键点检测
人体骨骼关键点检测对于描述人体姿态，预测人体行为至关重要。
是诸多计算机视觉任务的基础，例如动作分类，异常行为检测，以及自动驾驶等等。

simpleBaselines


### 推荐
内容理解, text-cls2014, tagspace, 
匹配, dssm2013, multiview-simnet 


词向量， word2vec，node2vec，
召回，word2vec, ssr, fasttext, yotubednn, ncf,gnn, ralm, tdm,
排序，wide&deep, deepfm, xdeepfm(cin), din, dien, lr, fm, ffm, fnn, dcn,
多任务, esmm, mmoe sharebottom, 

重排, listwise



这些没有公开的指标了。



