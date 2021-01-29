

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





