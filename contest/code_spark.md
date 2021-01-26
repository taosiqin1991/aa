
https://zhuanlan.zhihu.com/p/24332808


https://zhuanlan.zhihu.com/p/68524715
flink实现pagerank

pagerank的算法会维护两个数据集：一个由（pageID，linkList）的元素组成，包含每个页面的相邻页面的列表；另一个由（pageID，rank）元素组成，包含每个页面的当前排序值。它按如下步骤进行计算。

1) 将每个页面的排序值初始化为1.0。
2) 在每次迭代中，对页面p，向其每个相邻页面（有直接链接的页面）发送一个值为rank(p)/numNeighbors(p)的贡献值。
3) 将每个页面的排序值设为0.15 + 0.85 * contributionsReceived。
4) 最后两个步骤会重复几个循环，在此过程中，算法会逐渐收敛于每个页面的实际PageRank值。在实际操作中，收敛通常需要大约10轮迭代。


