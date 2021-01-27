

1279 红绿灯路口

c++互斥锁
记录上一次通行的道路

class TrafficLight{
public:
TrafficLight();
void carArrived(int carId, int roadId, int direction, 
    function<void()> turnGreen,
    function<void()> crossCar){

}
};

```cpp
class TrafficLight{
private:
mutex m;
char road;

public:
TrafficLight(){
    road='A';
}
void carArrived(int carId, int roadId, int direction, 
    function<void()> turnGreen,
    function<void()> crossCar){
    unique_lock<mutex> l(m);  // lock
    if(direction<=2 & road!='A'){
        turnGreen();
        road='A';
    }else if(direction>2 && road!='B'){
        turnGreen();
        road='B';
    }
    crossCar();
 
}
};

```

1188 设计有限阻塞队列

void enqueue(int element);
int dequeue();

利用c++11 条件变量

```cpp
// one condition_var can work
class BoundedBlockingQueue{
private:
    condition_variable cvEn;
    // condition_variable cvDe;
    mutex m;
    queue<int> que;
    int cap;

public:
    BoundedBlockingQueue(int capacity){
        cap = capacity;
    }

    void enqueue(int element){
        unique_lock<mutex> l(m); //
        cvEn.wait(l, [&]{
            return que.size()< cap;
        });
        que.push( element);
        cvEn.notify_one(); //
    }

    int dequeue(){
        unique_lock<mutex> l(m);
        cvEn.wait(l, [&]{
            return !que.empty();
        });
        int ret = que.front();
        que.pop();
        cvEn.notify_one();
        return ret;
    }

    int size(){
        unique_lock<mutex> l(m);
        int size = que.size();
        return size;
    }
};

```

```cpp

```


```cpp

```


```cpp

```


```cpp

```


```cpp

```


```cpp

```


```cpp

```


```cpp

```


```cpp

```


```cpp

```


```cpp

```


```cpp

```

```cpp

```

```cpp

```
