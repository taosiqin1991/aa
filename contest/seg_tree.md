


1649 通过指令创建有序数组

线段树，890 ms
树状数组，590 ms, 又改代码了 280 ms
平衡树, 1184 ms


```cpp
// 280 ms
class Solution{
public:
static constexpr int mod = 1e9+7;

int c[100005]; // 1e6+5;
int n;
void add(int x, int val){
    while(x<=100000){
        c[x] += val;  //
        x += (x&(-x));
    }
}

int query(int x) const{
    int res=0;
    while(x){
        res += c[x];
        x -= (x&(-x));
    }
    return res;
}

int createSortedArray(vector<int>& insts) {
    int idx=1;
    long res=0;
    memset(c, 0, sizeof(c));
    int cnt[100005]={0};
    int mod = pow(10,9)+7;

    for(int i=0; i<insts.size(); i++){
        int low_n = query( insts[i]-1);
        res += min(low_n, i -low_n -cnt[insts[i]]);  // 
        res %= mod;
        cnt[ insts[i]]++;
        add( insts[i], 1);
    }
    return res;
}
};

/*
int createSortedArray(vector<int>& insts) {
    int ub = *max_element(insts.begin(), insts.end());
    BIT bit(ub);
    long long res=0;
    for(int i=0; i<insts.size(); i++){
        int x = insts[i];

        int small = bit.query(x-1);
        int big = i - bit.query(x);
        res += min(small, big);
        bit.add( x, 1);
    }
    return res % mod;
}
*/

```

线段树
```cpp
// 1080 ms
#define MAX_V 100005
struct Node{
    int l;
    int r;
    int cnt;
    Node():l(0),r(0),cnt(0){}
};

class Solution{
public:
    Node* bit;
    void build(int p, int l, int r){
        bit[p].l =l;
        bit[p].r =r;
        if(l==r) return;
        
        int mid = (l+r)/2;
        build(p*2, l, mid);
        build(p*2+1, mid+1, r);
    }

    int query(int p, int l, int r){
        if(bit[p].l> r || bit[p].r<l || bit[p].cnt==0) return 0;

        if(bit[p].l >=l && bit[p].r <= r) return bit[p].cnt; // 
        return query(p*2,l,r) + query(p*2+1, l, r);
    }

    void insert(int cur, int p){
        bit[cur].cnt++;
        if(bit[cur].l== bit[cur].r) return ;
        
        int mid=(bit[cur].l + bit[cur].r)/2;
        if(mid>=p) insert(cur*2, p);
        else insert(cur*2+1, p);
    }

    int createSortedArray(vector<int>& insts){
        bit =new Node[MAX_V *4];
        build( 1,1, 100000);
        long res=0;
        int mod= 1e9+7;
        unordered_map<int,int> umap;
        
        for(int i=0; i<insts.size(); i++){
            int low_n = query(1, 1, insts[i]-1);
            res += min(low_n, i -low_n -umap[insts[i]]);
            res %=mod;
            umap[insts[i]]++;
            insert(1, insts[i]); //
        }
        return res;
    }

};
```



```cpp
// 890 ms
// not good
class SegTree{
private:
    int n;
    vector<int> seg_node;

public:
    SegTree(int n_):n(n_), seg_node(n_*4, 0){}
    void update(int x){
        update_dfs(0, 1, n, x);
    }

    int query(int l, int r){
        return query_dfs(0, 1, n, l, r);
    }

    void update_dfs(int id, int l, int r, int x){
        if(x<l || x>r) return ;

        ++seg_node[id];
        if(l==r) return ;

        int m = l + (r-l)/2;
        update_dfs(id*2+1, l, m, x);
        update_dfs(id*2+2, m+1, r, x);
    }

    int query_dfs(int id, int l, int r, int ql, int qr){
        if(qr<l || ql>r) return 0;
        
        if(ql<=l && r<=qr) return seg_node[id]; //
        
        int m = l+(r-l)/2;
        int tl = query_dfs(id*2+1, l, m, ql, qr);
        int tr = query_dfs(id*2+2, m+1, r, ql, qr);
        return tl + tr;
    }

};

class Solution{
public:
static constexpr int mod = 1e9+7;
int createSortedArray(vector<int>& insts) {
    int ub = *max_element(insts.begin(), insts.end());
    SegTree st(ub);
    long long res=0;
    for(int i=0; i<insts.size(); i++){
        int x = insts[i];
        int small = st.query(1, x-1);
        int big = st.query(x+1, ub);
        res += min(small, big); // why
        st.update( x);
    }
    return res% mod;
}
};
```




```cpp
// 1200 ms
struct BNode{
    int val;
    int seed;
    int cnt;
    int size;
    BNode* l;
    BNode* r;
    
    BNode(int v, int s):val(v),seed(s),cnt(1), size(1), l(nullptr),r(nullptr){}

    BNode* left_rotate(){
        int prev_n = size;
        int cur_n = (l? l->size:0) + (r->l? r->l->size:0) + cnt;
        BNode* root = r;
        r = root->l;

        root->l = this;
        root->size = prev_n;
        size = cur_n;
        return root;
    }

    BNode* right_rotate(){
        int prev_n = size;
        int cur_n = (r? r->size: 0) + (l->r? l->r->size: 0) + cnt;
        BNode* root = l;
        l = root->r;
        
        root->r = this;
        root->size = prev_n;
        size = cur_n;
        return root;
    }
};

class BalancedTree{
private:
    BNode* root;
    int size;
    mt19937 gen;
    uniform_int_distribution<int> dis;

public:
    BalancedTree(): root(nullptr), size(0), gen(random_device{}()), dis(INT_MIN,INT_MAX){}

    int get_size() const {
        return size;
    }

    void insert(int x){
        ++size;
        root = insert_dfs(root, x);
        // ++size;
    }

    int lower_bound(int x) const{
        BNode* node = root;
        int res = INT_MAX;
        while(node){
            if(x== node->val) return x;

            if(x< node->val){
                res = node->val;
                node = node->l;
            }
            else{
                node = node->r;
            }
        }
        return res;
    }

    int upper_bound(int x) const{
        BNode* node = root;
        int res = INT_MAX;
        while(node){
            if(x < node->val){
                res = node->val;
                node = node->l;
            }
            else{
                node = node->r;
            }
        }
        return res;
    }

    pair<int, int> rank(int x) const{
        BNode* node = root;
        int res=0;
        while(node){
            if(x < node->val){
                node = node->l;
            }
            else{
                res +=(node->l? node->l->size: 0) + node->cnt;
                if(x== node->val){
                    return {res-node->cnt+1, res};
                }
                node = node->r;
            }
        }
        return {INT_MIN, INT_MAX};
    }

    BNode* insert_dfs(BNode* node, int x){
        if(!node) return new BNode(x, dis(gen)); //
        
        ++node->size; // size++
        if(x< node->val){
            node->l = insert_dfs(node->l, x);
            if(node->l->seed > node->seed){
                node = node->right_rotate();
            }
        }
        else if(x> node->val){
            node->r = insert_dfs(node->r, x);
            if(node->r->seed > node->seed){
                node = node->left_rotate();
            }
        }
        else{
            ++node->cnt;
        }
        return node;
    } 
};

class Solution{
public:
static constexpr int mod = 1e9+7;
int createSortedArray(vector<int>& insts) {
    BalancedTree bt;

    long long res=0;
    for(int i=0; i<insts.size(); i++){
        int x = insts[i];
        int lb = bt.lower_bound(x);
        int rb = bt.upper_bound(x);
        int small =(lb==INT_MAX? i: bt.rank(lb).first-1);
        int big = (rb==INT_MAX?0: i- bt.rank(rb).first+1);

        res += min(small, big);
        bt.insert( x);
    }
    return res % mod;
}
};
```


