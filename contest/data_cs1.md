

423 全O(1)的数据结构

hash + 双向链表

考虑到可能会出现多个value是相同的值，那么一个元素自增或自减可能要“越过”这些相同的元素(因为它们在有序链表是相邻的)，则修改的复杂度会退化为O(N)，因此这里需要用一个节点存储所有相同的value。


```cpp
class AllOne{
private:
    struct Node{
        unordered_set<string> ns; // why
        int val=0;
        Node(int v):val(v){}
    };

    unordered_map<string, list<Node>::iterator> kv;
    list<Node> ll;

public:
    AllOne(){

    }

    void inc(string key){
        if(kv.count(key)){
            auto old_n = kv[key];
            auto new_n = next(old_n, 1);
            if(new_n==ll.end() || new_n->val > old_n->val+1){
                new_n = ll.insert(new_n, Node(old_n->val+1));
            }

            new_n->ns.insert(key);
            old_n->ns.erase(key);

            if(old_n->ns.empty()){
                ll.erase(old_n);
            }
            kv[key] = new_n;
        }
        else{
            auto new_n = ll.begin();
            if(ll.empty() || ll.begin()->val >1){
                new_n = ll.insert(ll.begin(), Node(1));
            }
            new_n->ns.insert(key); //
            kv[key] = new_n; // 
        }
    }

    void dec(string key){
        if(kv.count(key)){
            auto old_n = kv[key];
            if(old_n->val==1){
                kv.erase(key);  // erase map
            }
            else{
                auto new_n = next(old_n, -1);
                if(old_n==ll.begin() || new_n->val < old_n->val-1){
                    new_n = ll.insert(old_n, Node(old_n->val-1));
                }
                new_n->ns.insert(key);
                kv[key] = new_n;
            }

            old_n->ns.erase(key);
            if(old_n->ns.empty()){
                ll.erase(old_n); // erase ll
            }
        }
    }

    string getMaxKey(){
        if(ll.empty()) return "";
        return *ll.rbegin()->ns.begin();
    }

    string getMinKey(){
        if(ll.empty()) return "";
        return *ll.begin()->ns.begin();
    }
};
```


295 数据流中的中位数

简单排序 nlogn + 1 = nlogn, space n
插入排序  n + logn = n
两个堆, 5*logn + 1 = logn
multiset和双指针, logn + 1 = logn (best)
space all n

单指针最快。
双指针比单指针容易写，容易调试。
```cpp
class medianFinder{
private:
    multiset<int> data;
    multiset<int>::iterator lo_mid, hi_mid;
    
public:
    medianFinder():lo_mid(data.end()), hi_mid(data.end()){

    }

    void addNum(int num){
        const int n = data.size();
        data.insert(num);
        
        if(!n){ // first element insert
            lo_mid = data.begin();
            hi_mid = data.begin();
        }
        else if( n&1 ){ // odd
            if(num<*lo_mid) lo_mid--;
            else hi_mid++;

        }
        else{ // even
            if(num>*lo_mid && num<*hi_mid){
                lo_mid++;
                hi_mid--;
            }
            else if(num>= *hi_mid){
                lo_mid++;
            }
            else{ // num<= lo< hi
                lo_mid = --hi_mid;
            }
        }
    }

    double findmedian(){
        return (*lo_mid + *hi_mid)*0.5;
    }
};

```

```cpp
class medianFinder{
private:
    priority_queue<int> lo; // max heap
    priority_queue<int, vector<int>, greater<int>> hi; // min heap
    
public:
    void addNum(int num){
        lo.push(num);
        
        hi.push(lo.top());
        lo.pop();
        
        if(lo.size()< hi.size()){
            lo.push( hi.top());
            hi.pop();
        }

    }
    double findmedian(){
        return lo.size()> hi.size()? (double) lo.top(): (lo.top()+hi.top())*0.5;
    }
};
```


// multiset insert logn, find O(1)
```cpp
class medianFinder{
private:
    multiset<int> data;
    multiset<int>::iterator mid;
    
public:
    medianFinder():mid(data.end()){

    }

    void addNum(int num){
        const int n = data.size();
        data.insert(num);
        
        if(!n){ // first element insert
            mid = data.begin();
        }
        else if(num< *mid){
            mid = (n&1? mid: prev(mid));
        }
        else{
            mid = (n&1? next(mid): mid);
        }
    }

    double findmedian(){
        const int n = data.size();
        return (*mid + *next(mid, n%2-1))*0.5;
    }
};

```

```cpp
class medianFinder{
private:
    vector<int> store;

public:
    void addNum(int num){
        if(store.empty()) store.push_back(num);
        else store.insert(lower_bound(store.begin(), store.end(), num), num);  // logn + n
    }

    double findmedian(){
        sort((store.begin(), store.end()));
        int n = store.size();
        return (n&1? store[n/2]: (store[n/2]+store[n/2-1])*0.5);
    }
};
```


```cpp
// bad
vector<double> stone;

void addNum(int num){
    store.push_back(num);
}

double findmedian(){
    sort((store.begin(), store.end()));
    int n = store.size();
    return (n&1? store[n/2]: (store[n/2]+store[n/2-1])*0.5);
}

```



1656 设计有序流

```cpp
class OrderedStream{
int ptr;
vector<string> data; // len n
int size;

public:
OrderedStream(int n){
    size = n;
    ptr=0;
    for(int i=0; i<n; i++){ 
        data.push_back("");
    }
}

vector<string> insert(int id, string val){
    data[id-1] = val;
    
    vector<string> res;
    while(data[ptr]!="" &&  ptr<size ){
        res.push_back(data[ptr]);
        ++ptr;
    }
    return res;
}
};
```





