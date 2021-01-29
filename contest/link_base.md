
队列实现栈
1）用一个队列实现即可。
入栈O(n), 出栈O(1), top O(1), empty O(1)

入栈操作需要将队列中的 nn 个元素出队，并入队 n+1n+1 个元素到队列，共有 2n+1 次操作，每次出队和入队操作的时间复杂度都是 O(1)O(1)，因此入栈操作的时间复杂度是 O(n)。

```cpp
class MyStack{
public:
    MyStack(){}
    void push(int x){
        int n = q.size();
        q.push( x);

        for(int i=0; i<n; i++){
            q.push( q.front());
            q.pop();
        }
    }
    int pop(){
        int r = q.front();
        q.pop();
        return r;

    }
    int top(){
        return q.front();
    }
    bool empty(){
        return q.empty();
    }
private:
    queue<int> q;

};

```


2个栈实现队列
1）stk1处理push，stk2处理弹出。直接从stk2弹出，或者从stk1全部倒入stk2再从stk2弹出。
push time O(1), pop time O(n)

```cpp
class MyQueue {
public:
    /** Initialize your data structure here. */
    MyQueue() {

    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
        stk1.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        if( !stk2.empty()){
            int a = stk2.top();
            stk2.pop();
            return a;
        }
        else if( !stk1.empty()){
            while( !stk1.empty()){
                stk2.push( stk1.top());
                stk1.pop();

            }
            int a = stk2.top();
            stk2.pop();
            return a;
        }
        else{
            return -1;
        }
    }
    
    /** Get the front element. */
    int peek() {

        if( !stk2.empty()){
            return stk2.top();

        }
        else if( !stk1.empty()){

            while( !stk1.empty()){
                stk2.push( stk1.top());
                stk1.pop();

            }
            return stk2.top();

        }
        else{
            return -1;
        }

    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
        return stk1.empty() && stk2.empty();
    }
private:
    stack<int> stk1;
    stack<int> stk2;
};
```




```cpp

```







链表插入
```cpp

```


```cpp

```

```cpp

```