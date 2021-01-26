

## 美团spark优化思路

基础思路：
开发调优、资源调优、数据倾斜调优、shuffle调优。

针对业务，困难/更棘手/影响力更大的是 数据倾斜调优和shuffle调优。

https://zhuanlan.zhihu.com/p/21922826

https://zhuanlan.zhihu.com/p/49169166


### 开发调优


### 资源调优

以下是一份spark-submit命令的示例，大家可以参考一下，并根据自己的实际情况进行调节：

./bin/spark-submit \
  --master yarn-cluster \
  --num-executors 100 \
  --executor-memory 6G \
  --executor-cores 4 \
  --driver-memory 1G \
  --conf spark.default.parallelism=1000 \
  --conf spark.storage.memoryFraction=0.5 \
  --conf spark.shuffle.memoryFraction=0.3 \



### 数据倾斜调优
绝大多数task执行得都非常快，但个别task执行极慢。比如，总共有1000个task，997个task都在1分钟之内执行完了，但是剩余两三个task却要一两个小时。这种情况很常见。

原本能够正常执行的Spark作业，某天突然报出OOM（内存溢出）异常，观察异常栈，是我们写的业务代码造成的。这种情况比较少见。





### shuffle调优












