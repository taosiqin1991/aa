
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



合并两个有序链表
1) time O(m+n), space O(1) 迭代方式
2) time O(m+n), space O(m+n) 递归方式

递归调用 mergeTwoLists 函数时需要消耗栈空间，栈空间的大小取决于递归调用的深度。结束递归调用时 mergeTwoLists 函数最多调用 n+m 次，因此空间复杂度为 O(n+m)


```cpp
ListNode* mergetwoLists(ListNode* l1, ListNode* l2){
    ListNode prehead(-1);
    ListNode* pre = &prehead;

    while(l1!=nullptr && l2!=nullptr){
        if(l1->val < l2->val){
            pre->next = l1;
            l1 = l1->next;
            pre = pre->next;
        }
        else{
            pre->next = l2;
            l2 = l2->next;
            pre = pre->next;
        }

    }
    pre->next = l1==nullptr? l2: l1;
    return prehead.next;
}



ListNode* mergeTwoLists(ListNode* l1, ListNode* l2){
    if( l1==nullptr) return l2;
    else if(l2==nullptr) return l1;

    if(l1->val < l2->val){
        l1->next = mergeTwoLists(l1->next, l2);
        return l1;
    }
    else{
        l2->next = mergeTwoLists(l1, l2->next);
        return l2;
    }
}
```


合并k个有序的链表
1) 优先队列实现，time O(knlogk), space O(k)
   优先队列元素不超过k个，insert和delete时间为logk，最多有kn个点。对于每个点都插入删除各一次，故渐进时间复杂度为 knlogk

2）分治合并，time O(knlogk), space O(logk) 栈空间
3）顺序合并，time O(k^2 *n), space O(1)

故最好的办法是分治合并。

分治合并，第一轮合并k/2组链表，每一组时间代价是O(2n)
第二轮合并k/4组链表，每一组时间代价是O(4n)
总时间是 sum_(i=1)^(inf) k/(2^i) * 2^i * n = knlogk

```cpp
ListNode* mergeKLists(vector<ListNode*>& lists){
    return merge(lists, d0, lists.size()-1);
}

ListNode* merge(vector<ListNode*>& lists, int l, int r){
    if(l==r) return lists[l];
    if(l>r) return nullptr;

    int mid = l + (r-l)/2;
    ListNode* p1 = merge(lists, l, mid);
    ListNode* p2 = merge(lists, mid+1, r);
    return mergeTwoLists( p1, p2);
}

// bad case
ListNode* mergeKLists_order(vector<ListNode*>& lists){
    ListNode* ans = nullptr;
    for(int i=0; i<lists.size(); i++){
        ans = mergeTwoLists(ans, lists[i]);
    }
    return ans;
}

ListNode* mergeTwoLists(ListNode* 11, ListNode* l2){
    if(l1==nullptr) return l2;
    else if(l2==nullptr) return l1;

    ListNode* a = l1;
    ListNode* b = l2;
    ListNode tmp(INT_MIN);

    ListNode* p = &tmp;
    while(a && b){
        if(a->val < b->val){
            p->next = a;
            a = a->next;
            p = p->next;
        }
        else{
            p->next = b;
            b = b->next;
            p = p->next;
        }
    }
    p->next = a? a: b;
    return tmp.next;
}

```

优先队列合并，需要维护每个链表没有被合并的元素的最前面一个。
k个链表最多有k个这样的元素。

```cpp

priority_queue<int, vector<int>, less<int> > max_heap;
priority_queue<int, vector<int>, greator<int> > min_heap;

class Solution{
public:
    ListNode* mergeKLists(vector<ListNode*>& lists){
        for(auto node: lists){
            if(node)  q.push( {node->val, node});
        }
    

        ListNode tmp;
        ListNode* p = &tmp;

        while( !q.empty()){
            auto f = q.top();
            q.pop();

            p->next = f.ptr;
            p = p->next;

            if(f.ptr->next) q.push( {f.ptr->next->val, f.ptr->next});
        }
        return tmp.next;
    }


private:
    struct Status{
        int val;
        ListNode* ptr;
        bool operator<(const Status& a) const{
            return val > a.val;
        }
    };

    priority_queue<Status> q;
};



struct ListNode{
    int val;
    ListNode* next;
    ListNode(int v){val=v; next=nullptr;}
};

struct cmp{
    bool operator()(ListNode* a, ListNode* b){
        if(a.x == b.x) return a.y > b.y;
        return a.x > b.x;
    }
}

priority_queue<ListNode, vector<ListNode*>, cmp> p;

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