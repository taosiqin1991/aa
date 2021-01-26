

precision, recall, f1 score指标的局限性：
二分类时，只能评估正类的分类性能
多分类时，只能评估某一类的分类性能


```python
def binary_confusion_mat(labels, preds):
    TP, FP, TN, FN = 0,0,0,0

    for i in range(len(labels)):
        if labels[i]==1 and preds[i]==1:
            TP += 1
        if labels[i]==1 and preds[i]==0:
            FN +=1
        if labels[i]==0 and preds[i]==1:
            FP += 1
        if labels[i]==0 and preds[i]==0:
            TN += 1
    return TP, FP, TN, FN





def test_auc():
    labels = [1,1,1,1,1,0,0,1,1,1]
    pres = [1,0,1,1,1,1,0,1,1,0]
    print("confusion mat", binary_confusion_mat(labels, preds))

    

```


AUC指标局限性
1）AUC作为排序的衡量指标有一定局限性。它衡量的是整体样本间的排序能力， 对计算广告来说，它衡量的是不同用户对不同广告之间的排序能力。而线上环境往往需要关注同一个用户的不同广告之间的排序能力。
针对此问题，阿里 DIN 模型提出改进版AUC指标，用户加权平均（用曝光量去加权AUC）gAUC，希望能更好反映线上真实环境的排序能力。



```python
def calcul_auc(prob, labels):
    f = list(zip( prob, labels))
    sorted_f = sorted(f, key=lambda x: x[0])
    rank = [val2 for val1,val2 in sorted_f]
    rank_list = [i+1 for i in range(len(rank) if rank[i])==1]

    pos_num = 0
    neg_num = 0
    for i in range(len( labels)):
        if( labels[i]==1):
            pos_num +=1
        else:
            neg_num +=1

    auc = 0
    auc = float( sum(rank_list)- pos_num*(pos_num+1)/2 )/(pos_num+neg_num)
    return auc 
     
    
def script_cal_auc():
    y = []
    pred = []
    with open("tmp", "r") as ff:
        for line in ff.readlines():
            pred.append( float(line.strip()))

    with open("data/test1", "r") as ff:
        for line in ff.readlines():
            y.append( int(line.strip().split("")[0] ) )

    auc = calcul_auc(pred, y)
    print("y len {0}, pred lan {1}, auc {2}".format(y, pred, auc))

```


node2vec和deepwalk的底层是一样的，只是随机游走不一样。
node2vec中的p和q为1时，node2vec和deepwalk完全一样，但实验结果中显示node2vec效果要好。原因在于node2vec菜肴那个alias sampling，deepwalk采用python的choice函数，虽然都是等概率，但实际结果来看node2vec更好。



graph embedding时用到的alias sampling
```python
import numpy as np

def gen_prob_dist(N):
    p = np.random.randint(0, 100, N)
    return p/np.sum(p)

def create_alias_table(area_ratio):
    l = len(area_ratio)
    
    accept = [0] * l
    alias = [0] * l
    small = []
    large = []

    for i, prob in enumerate(area_ratio):
        if prob < 1.0:
            small.append( i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio[small_idx]
        alias[small_idx] = large_idx

        area_ratio[large_idx] = area_ratio[large_idx] - (1-area_ratio[small_idx])
        if area_ratio[large_idx]<1.0:
            small.append( large_idx)
        else:
            large.append( large_idx)
    
    while large:
        large_idx = large.pop()
        accept[ large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1
    
    return accept, alias


def alias_sample(accept, alias):
    N = len(accept)
    i = int( np.random.random() * N)
    r = np.random.random()
    
    if r < accept[i]:
        return i
    else:
        return alias[i]


def simulate(N=100, k=10000,):
    truth = gen_prob_dist(N)
    area_ratio = truth * N
    
    accept, alias = create_alias_table(area_ratio)
    
    ans = np.zeros(N)
    for _ in rank(k):
        i = alias_sample( accept, alias)
        ans[i] += 1
    return ans/np.sum(ans), truth

```



简单粗糙的deepwalk代码
```python

```


潜力股node2vec代码
```python


```

```python

```