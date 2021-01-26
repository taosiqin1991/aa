
https://zhuanlan.zhihu.com/p/82116922 ps
 

https://github.com/wzhe06/CTRmodel  spark code




### 凸优化

https://www.zhihu.com/question/24641575


凸优化性质好，并且即使是日常生活中的许多非凸优化问题，目前最有效的办法也只能是利用凸优化的思路去近似求解。一些例子有：带整数变量的优化问题，松弛之后变成凸优化问题（所以原问题其实是凸优化问题+整数变量）；任意带约束的非凸连续优化问题，其对偶问题作为原问题解的一个lower bound，一定是凸的！一个更具体的例子，大家都知道针对带有hidden variable的近似求解maximum likelihood estimate的EM算法，或者贝叶斯版本里头所谓的variational Bayes(VB) inference。而原本的MLE其实是非凸优化问题，所以EM和VB算法都是找到了一个比较好优化的concave lower bound对这个lower bound进行优化。



### 运筹学



### 贝叶斯推断


### 因果推断


### 马尔可夫过程

https://www.zhihu.com/question/23444414/answer/24668679




### svm 和 lr

内积？后来有负类的数学家研究了一下RBF核是否对应一个向量映射，结果是如果想把RBF核表达成一个向量内积，我们需要一个映射将向量映射到一个无穷维的线性空间去。发现了这一点的数学家又发展了Mercer定理和重建核希尔伯特空间（Reproducing Kernel Hilbert Space）理论，但这就是另外一个故事了。






