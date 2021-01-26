
### spark性能实践
此程序跑5分钟小数据量日志不到5分钟，但相同的程序跑一天大数据量日志各种失败。经优化，使用160 vcores + 480G memory，一天的日志可在2.5小时内跑完。



### 优化思路
1）保证大数据量下任务运行成功
2）降低资源消耗
3）提高计算性能
三个目标优先级依次递减，首要解决的是程序能够跑通大数据量，资源性能尽量进行优化。


### 优化思路

程序方面，从stage、cache、partition方面考虑。
硬件资源、内存/GC优化，网络，IO



stage
在进行shuffle操作时，如reduceByKey、groupByKey，会划分新的stage。同一个stage内部使用pipe line进行执行，效率较高；stage之间进行shuffle，效率较低。故大数据量下，应进行代码结构优化，尽量减少shuffle操作。


cache
本例中，首先计算出一个baseRDD，然后对其进行cache，后续启动三个子任务基于cache进行后续计算。

对于5分钟小数据量，采用StorageLevel.MEMORY_ONLY，而对于大数据下我们直接采用了StorageLevel.DISK_ONLY。DISK_ONLY_2相较DISK_ONLY具有2备份，cache的稳定性更高，但同时开销更大，cache除了在executor本地进行存储外，还需走网络传输至其他节点。

后续我们的优化，会保证executor的稳定性，故没有必要采用DISK_ONLY_2。

实时上，如果优化的不好，我们发现executor也会大面积挂掉，这时候即便DISK_ONLY_2，也是然并卵，所以保证executor的稳定性才是保证cache稳定性的关键。

cache是lazy执行的，这点很容易犯错




partition
一个stage由若干partition并行执行，partition数是一个很重要的优化点。
本例中，一天的日志由6000个小文件组成，加上后续复杂的统计操作，某个stage的parition数达到了100w。parition过多会有很多问题，比如所有task返回给driver的MapStatus都已经很大了，超过spark.driver.maxResultSize（默认1G），导致driver挂掉。虽然spark启动task的速度很快，但是每个task执行的计算量太少，有一半多的时间都在进行task序列化，造成了浪费，另外shuffle过程的网络消耗也会增加。
对于reduceByKey()，如果不加参数，生成的rdd与父rdd的parition数相同，否则与参数相同。还可以使用coalesce()和repartition()降低parition数。例如，本例中由于有6000个小文件，导致baseRDD有6000个parition，可以使用coalesce()降低parition数，这样parition数会减少，每个task会读取多个小文件。
这里使用repartition()不使用coalesce()，是为了不降低resultRDD计算的并发量，通过再做一次shuffle将结果进行汇总.



资源参数
一些常用的参数设置如下：

--queue：集群队列
--num-executors：executor数量，默认2
--executor-memory：executor内存，默认512M
--executor-cores：每个executor的并发数，默认1
executor的数量可以根据任务的并发量进行估算，例如我有1000个任务，每个任务耗时1分钟，若10个并发则耗时100分钟，100个并发耗时10分钟，根据自己对并发需求进行调整即可。默认每个executor内有一个并发执行任务，一般够用，也可适当增加，当然内存的使用也会有所增加。





集群环境
我们的yarn集群节点上上跑着mapreduce、hive、pig、tez、spark等各类任务，除了内存有所限制外，CPU、带宽、磁盘IO等都没有限制（当然，这么做也是为了提高集群的硬件利用率），加上集群整体业务较多负载较高，使得spark的执行环境十分恶劣。常见的一些由于集群环境，导致spark程序失败或者性能下降的情况有：

节点挂掉，导致此节点上的spark executor挂掉
节点OOM，把节点上的spark executor kill掉
CPU使用过高，导致spark程序执行过慢
磁盘目录满，导致spark写本地磁盘失败
磁盘IO过高，导致spark写本地磁盘失败
HDFS挂掉，hdfs相关操作失败



查看日志
executor的stdout、stderr日志在集群本地，当出问题时，可以到相应的节点查询，当然从web ui上也可以直接看到。
executor除了stdout、stderr日志，我们可以把gc日志打印出来，便于我们对jvm的内存和gc进行调试。
除了executor的日志，nodemanager的日志也会给我们一些帮助，比如因为超出内存上限被kill、资源抢占被kill等原因都能看到。
除此之外，spark am的日志也会给我们一些帮助，从yarn的application页面可以直接看到am所在节点和log链接。



内存/gc


网络




IO









### spark与flink 的性能差异

在stream流处理中，flink的性能是优于spark的。



### spark的action有哪些

### spark的transformer和action的区别

### groupByKey，reduceByKey，reduceByKeyLocally
前两者都是Transformation。





### map，mapPartition


### persist和cache异同




### spark作业提交流程


### spark的ML 和MLLib区别


### spark支持的3种集群管理


### 检查点意义


### spark master理解


### spark streaming小文件问题


### spark streaming与 flink区别



### spark粗粒度和细粒度对比


### mesos粗粒度和细粒度对比


### spark local和spark alone区别


### sparkContext 和sparkSession区别


### 描述一下worker异常的情况


### 描述一下master异常的情况


### 介绍一下spark架构


### spark streaming是如何收集和处理数据的




### worker和excutor异同


### spark提供的两种共享变量




### 为何要用yarn来部署spark


### 


### spark的stage如何划分


### spark数据倾斜如何处理



### 用过hive吗，用过join吗，join的实时技术是啥


### 给你几张表，想得到一些数据，spark怎么处理


### 怎么接解决内存不足情况下使用python处理大数据



### 如何做分布式计算




### kafka分布式下，如何保证消息的顺序



### spark的shuffle过程


### spark的聚合类算子，应该避免什么类型的算子



### spark on yarn作业执行流程，yarn-client和yarn cluster有何区别



### spark为何快，spark sql 一定比hive快吗


### spark容错，RDD，DAG，Stage怎么理解


### RDD如何通过记录更新的方式容错


### 宽依赖、窄依赖怎么理解


### job和task 如何理解


### spark任务


### spark容错方法



### spark粗粒度和细粒度


### 


