

LRU 最久未使用被淘汰，用哈希表 + 双向链表
LFU 最小频率未使用被淘汰，可以使用和LRU一样的数据结构，只是在 removeLastNode()中要根据node的cnt 来remove。



LRU设计
1) 设计数据结构，接收capacity参数作为缓存的最大容量
2）实现put和get方法，时间复杂度要求 O(1)

LRU缓存算法的数据结构是哈希链表，即 双向链表 + 哈希表。
插入数据时，数据插入list的头部。


```cpp
class LRUCache{
public:

    LRUCache(int capacity): cap_(capacity){}
    
    int get(int key){
        if( map.find(key) == map.end()){
            return -1;
        }
        
        int val = map[key]->second;
        // put data front
        put(key, val);
        return val;
    }

    void put(int key, int value){
        pair<int, int> x = {key, value};
        
        if(map_.find(key) != map.end()){
            // delete old
            cache_.erase( map[key]);

            // new insert to front
            cache_.emplace_front(x);
            // update map.
            map[key] = cache_.begin();
        }
        else{
            if( cap_==cache_.size()){
                // delete the last
                pair<int, int> last = cache_.back();
                cache_.pop_back();
                map.erase( last.first);
            }
            
            // directly add
            cache_.emplace_front(x);
            map_[key] = cache_.bagin();
        }
    }



private:
    // key ->   iterator to pair(key, val) in the list
    unordered_map<int, list<pair<int, int>>::iterator > map_;
    list<pair<int, int>> cache_;
    int cap_; 
}

```




LRUCache
map + dlinknode更好
map + list

LFUCache  460
map + balanced_tree


```cpp
class LFUCache{
public:
    LFUCache(int capacity){
        
    }
    int get(int key){

    }

    void put(int key, int value){
        
    }
}

```

```cpp

// unordered_map + list  188ms
// unordered_map + DLinkNode  164ms
struct DLinkNode{
    int key;
    int val;
    DLinkNode* pre;
    DLinkNode* next;
    DLinkNode(): key(0), val(0), pre(nullptr), next(nullptr){}
    DLinkNode(int k, int v): key(k), val(v), pre(nullptr), next(nullptr){}
};

class LRUCache{
private:
    DLinkNode* head;
    DLinkNode* tail;
    int size;
    int cap;
    unordered_map<int, DLinkNode*> mp;

public:
    LRUCache(int capacity):cap(capacity), size(0){
        head = new DLinkNode();
        tail = new DLinkNode();
        head->next = tail;
        tail->pre = head;

    }

    ~LRUCache(){
        DLinkNode* p = nullptr;
        while( head!=nullptr){
            p = head->next;
            delete head;

            head = p;
        }
        delete head;
        
    }

    int get(int key){
        if( !mp.count(key)) return -1;
         
        DLinkNode* node = mp[key];
        moveToHead(node);
        
        return node->val;
    }

    void put(int key, int val){
        if( !mp.count(key)){
            DLinkNode* node = new DLinkNode(key, val);
            mp[key] = node;
            
            addToHead(node);
            size++;

            if(size>cap){
                DLinkNode* removed = removeTail();
                mp.erase( removed->key);

                delete removed;  // key
                size--;
            }
        }
        else{
            DLinkNode* node = mp[key];
            node->val = val;
            moveToHead(node);
        }
    }

    // head, node, head->next
    void addToHead(DLinkNode* node){
        node->pre = head;
        node->next = head->next;

        head->next->pre = node;
        head->next = node;
    }

    void removeNode(DLinkNode* node){
        node->pre->next = node->next;
        node->next->pre = node->pre;
    }

    void moveToHead(DLinkNode* node){
        removeNode(node);
        addToHead(node);
    }

    // node, tail
    DLinkNode* removeTail(){
        DLinkNode* node = tail->pre;

        removeNode(node);
        return node;
    }

};



```




LRU缓存机制
哈希链表


因此有哈希链表 = 双向链表 + 哈希表。
python中是 OrderedDict
Java中是  LinkedHashMap


LFU缓存
哈希表 + 平衡二叉树
双哈希表

在 C++ 语言中，我们可以直接使用 std::set 类作为平衡二叉树

有bug

```cpp
struct DLinkNode{
    int key;
    int val;

    DLinkNode* next;
    DLinkNode* pre;
    DLinkNode(int k, int v):key(k), val(v), pre(nullptr), next(nullptr){}
    DLinkNode(int k, int v, DLinkNode* pp, DLinkNode* np):key(k), val(v), pre(pp), next(np){}

};

class LRUCache{
private:
    DLinkNode* head;
    DLinkNode* tail;
    int cap;
    int cnt;
    unordered_map<int, DLinkNode*> mp;

public:
    LRUCache(int c){
        cap = c;
        cnt = 0;
        head = new DLinkNode(100, 100);
        tail = new DLinkNode(-1, -1);
        head->next = tail;
        tail->pre = head;
    }

    ~LRUCache(){
        mp.clear();
        DLinkNode* tmp = nullptr;
        while( head!=nullptr){
            tmp = head->next;
            delete head;
            head = head->next;
        }
        
    }

    void put(int k, int v){
        if(mp.count(k)){
            mp[k]->val = v;
            move_from_mid(mp[k]);
            move_to_head(mp[k]);
        }
        else{
            if(cnt==cap) delete_tail();
            else cnt++;

            add_head(k, v);
        }
    }

    int get(int k){
        if(mp.count(k)==0) return -1;
        
        move_from_mid(mp[k]);
        move_to_head(mp[k]);
        return mp[k]->val;
    }
    
    // head, x, head->next.
    void add_head(int& k, int& v){
        DLinkNode* l = head;
        DLinkNode* r = head->next;
        DLinkNode* mid = new DLinkNode(k, v, l, r);
        l->next = mid;
        r->pre = mid;
        mp[k] = mid;
    }

    // tail->pre, x, tail.
    void delete_tail(){
        DLinkNode* mid = tail->pre;
        DLinkNode* l = tail->pre->pre;
        DLinkNode* r = tail;

        l->next = r;
        r->pre = l;
        
        mp.erase(mid->key);
        delete mid;
    }

    void move_from_mid(DLinkNode* mid){
        DLinkNode* l = mid->pre;
        DLinkNode* r = mid->next;
        
        mid->pre = nullptr;
        mid->next = nullptr;
        l->next = r;
        r->next = l;
    }

    void move_to_head(DLinkNode* mid){
        DLinkNode* l = head->pre;
        DLinkNode* r = head;
        mid->pre = l;
        mid->next = r;
        l->next = mid;
        r->pre = mid;
    }

};

```

```cpp

class LRUCache {
public:
    struct node {
        int val;
        int key;
        node* pre;
        node* next;
        node(){}
        node(int key, int val):key(key), val(val), pre(NULL), next(NULL){}
    };

    LRUCache(int capacity) {
        this->capacity = capacity;
        head = new node();
        tail = new node();
        head->next = tail;
        tail->pre = head;
    }
    
    void inserttohead(node* cur)
    {
        node* next = head->next;
        head->next = cur;
        cur->pre = head;
        next->pre = cur;
        cur->next = next;
    }

    node* nodedelete(node* cur)
    {
        cur->pre->next = cur->next;
        cur->next->pre = cur->pre;
        return cur;
    }
    void movetohead(node* cur)
    {
        node* temp = nodedelete(cur);
        inserttohead(temp);
    }
    int get(int key) 
    {
        int ret = -1;
        if ( ump.count(key))
        {
            node* temp = ump[key];
            movetohead(temp);
            ret = temp->val;
        }
        return ret;
    }

    void put(int key, int value) {
        if ( ump.count(key))
        {
            node* temp = ump[key];
            temp->val = value;
            movetohead(temp);
        }
        else
        {
            node* cur = new node(key, value);
            if( ump.size()== capacity )
            {
                node *temp = nodedelete(tail->pre);
                ump.erase(temp->key);
            }
            inserttohead(cur);
            ump[key] = cur;

        }
    }
    unordered_map<int, node*> ump;
    int capacity;
    node* head, *tail;
};


```



get logn, put logn, 性能瓶颈。space cap.

```cpp
// hashtable + binary_tree
struct Node{
    int cnt;  // freq
    int time; 
    int key;
    int val;
    
    Node(int c, int t, int k, int v):cnt(c), time(t), key(k), val(v){}

    bool operator<(const Node& b) const{
        return cnt==b.cnt? time< b.time: cnt<b.cnt;
    }
};

class LFUCache{
private:
    int cap;
    int time;
    unordered_map<int, Node> key_table;
    set<Node> s;

public:
    LFUCache(int c){
        cap = c;
        time = 0;
        key_table.clear();
        s.clear();
    }

    int get(int key){
        if(cap==0) return -1;
        
        auto it = key_table.find(key);
        if(it==key_table.end()) return -1;
        // old
        Node cache= it->second;
        s.erase(cache);
        cache.cnt ++;
        cache.time = ++time;

        // new
        s.insert(cache);
        it->second = cache;
        return cache.val;
    }

    void put(int key, int val){
        if(cap==0) return ;
        
        auto it = key_table.find(key);
        if( it==key_table.end()){
            // create
            if(key_table.size()==cap){
                // delete first in tree
                key_table.erase( s.begin()->key);
                s.erase(s.begin());
            }

            Node cache = Node(1, ++time, key, val);
            
            key_table.insert( {key, cache});
            s.insert( cache);
        }
        else{ // modify old
            Node cache = it->second;
            s.erase( cache);
            cache.cnt +=1;
            cache.time = ++time;
            cache.val = val;
            
            s.insert(cache);
            it->second = cache;
        }
    }    
};

```

双哈希表实现LFU

```cpp
struct Node{
    int key;
    int val;
    int freq;
    Node(int k, int v, int f): key(k), val(v), freq(f){}
};

class LFUCache{
private:
    int min_freq;
    int cap;
    unordered_map<int, list<Node>::iterator> key_table;
    unordered_map<int, list<Node>> freq_table;

public:
    LFUCache(int c){
        min_freq = 0;
        cap = c;
        key_table.clear();
        freq_table.clear();
    }

    int get(int key){
        if(cap==0) return -1;
        
        auto it = key_table.find(key);
        if( it==key_table.end()) return -1;

        list<Node>::iterator node = it->second;
        int val = node->val;
        int freq = node->freq;

        freq_table[freq].erase(node);
        // delete if link is empty
        if(freq_table[freq].size()==0){
            freq_table.erase(freq);
            if(min_freq==freq) min_freq +=1;
        }

        // insert freq+1
        freq_table[freq+1].push_front( Node(key, val, freq+1));
        key_table[key] = freq_table[freq+1].begin();
        return val;

    }

    void put(int key, int val){
        if(cap==0) return ;
        
        auto it = key_table.find(key);
        if(it==key_table.end()){  // insert new
            if(key_table.size()==cap){
                // delete
                auto it2 = freq_table[min_freq].back();
                key_table.erase( it2.key);
                freq_table[min_freq].pop_back();

                if(freq_table[min_freq].size()==0){
                    freq_table.erase( min_freq);
                }
            }

            freq_table[1].push_back( Node(key, val, 1));
            key_table[key] = freq_table[1].begin();
            min_freq = 1;
        }
        else{
            list<Node>::iterator node = it->second;
            int freq = node->freq;
            freq_table[freq].erase(node);

            if(freq_table[freq].size()==0){
                freq_table.erase(freq);
                if(min_freq==freq) min_freq +=1;
            }
            freq_table[freq+1].push_front( Node(key, val, freq+1));
            key_table[key] = freq_table[freq+1].begin();
        }
    }
};

```




