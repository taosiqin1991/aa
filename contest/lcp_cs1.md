


LCP 05 发LeetCoin

dfs序 + 线段树
zkw树()


进行线段树的操作(单点修改，区间修改，和区间查询)


后续遍历确定每个根节点所表示的树的结点范围，双树状数组log(n)动态更新区间，log(n)区间查询。

6
[[1, 2], [1, 6], [2, 3], [2, 5], [1, 4]]
[[1, 1, 500], [2, 2, 50], [3, 1], [2, 6, 15], [3, 1]]

```cpp
// 364 ms rank 100%
#define mAX 50001
#define ll long long
#define mod(v) ((v)%(1000000007)) // 1e9+7
#define lowbit(i) ((i)&(-i))

class Solution{
public:
// typedef long long ll;
// static const ll mod=1e9+7;

vector<vector<int>> g;
int s[mAX];
int f[mAX];
int vit=1;
ll tc[mAX];
ll ts[mAX];

vector<int> bonus(int n, vector<vector<int>>& ls, vector<vector<int>>& opt) {
    g.resize(n+1);
    for(auto& i: ls) g[i[0]].push_back( i[1]);
    dfs(1);
    
    vector<int> res;
    for(auto& e: opt){
        if(e[0]==1) update(f[e[1]], f[e[1]], e[2], n);
        else if(e[0]==2) update(s[e[1]], f[e[1]], e[2], n);
        else res.push_back( query( s[e[1]], f[e[1]], n));  //
    }
    return res;
}

void update(int i, int v, int n){
    for(int t=i; t<=n; t+= lowbit(t)){
        tc[t] += v;
        ts[t] += v*(i-1);
    }
}

void update(int l, int r, int v, int n){
    update(l, v, n);
    update(r+1, -v, n);
}

ll query(int i, int n){
    ll res=0;
    int t=i;
    while(i>0){
        res += t*tc[i]-ts[i];
        i-= lowbit(i);
    }
    return res;
}

ll query(int l, int r, int n){
    ll res = query(r, n) - query(l-1, n);  // l-1
    return mod(res);
}

void dfs(int id){
    s[id] = vit;
    for(auto& i: g[id]) dfs(i);
    f[id] = vit++;
}
};

```

后续遍历确定每个根节点所表示的树的结点范围，带lazy标记线段树log(n)动态更新区间，log(n)区间查询
```cpp
// 424 ms
#define mAX 50001
#define ll long long
#define mod(v) ((v)%(1000000007)) // 1e9+7
#define lowbit(i) ((i)&(-i))

class Solution{
public:
struct node{ // why
    ll v;
    ll lazy;
};
vector<vector<int>> g;
int s[mAX];
int f[mAX];
int vit=1;
node res[4*mAX];  // 

vector<int> bonus(int n, vector<vector<int>>& ls, vector<vector<int>>& opt) {
    g.resize(n+1);
    for(auto& i: ls) g[i[0]].push_back( i[1]);
    dfs(1);
    
    vector<int> ans;
    for(auto& e: opt){
        if(e[0]==1) update(1, 1, n, f[e[1]], f[e[1]], e[2]);
        else if(e[0]==2) update(1, 1, n, s[e[1]], f[e[1]], e[2]);
        else ans.push_back( query(1,1,n, s[e[1]], f[e[1]]));
    }
    return ans;
}

void up(int rt){
    res[rt].v = mod( res[rt<<1].v + res[rt<<1 |1].v);
}

void setv(int rt, int bg, int ed, int v){
    res[rt].v += (ed-bg+1)*v;
    if(bg!= ed)  res[rt].lazy +=v;
}

void down(int rt, int bg, int ed){
    if(res[rt].lazy==0) return ;
    int mid = (bg+ed)>>1;
    setv( rt<<1, bg, mid, res[rt].lazy);
    setv( rt<<1 | 1, mid+1, ed, res[rt].lazy);
    res[rt].lazy=0;
}

void update(int rt, int bg, int ed, int l, int r, int v){
    if(bg>r || ed<l) return ;
    if(l<=bg && ed <=r){
        setv(rt, bg, ed, v);
        return ;
    }

    down(rt, bg, ed);
    int mid=(bg+ed)>>1;
    update(rt<<1, bg, mid, l, r, v);
    update(rt<<1 |1, mid+1, ed, l, r, v);
    up(rt);
}

ll query(int rt, int bg, int ed, int l, int r){
    if(bg>r || ed<l) return 0;
    if(l<= bg && ed<=r) return res[rt].v;
    
    down(rt, bg, ed);
    int mid= (bg+ed)>>1;
    ll tmp = query(rt<<1, bg, mid, l, r) + query(rt<<1|1, mid+1, ed, l, r);
    return mod(tmp);
}

void dfs(int id){
    s[id] = vit;
    for(auto& i: g[id]){
        dfs(i);
    }
    f[id] = vit++;
}
};
```

// zkw tree
源代码bug  看看数组访问越界
```cpp
// bug  bug
#define mAX 50001
#define ll long long
#define mod(v) ((v)%(1000000007)) // 1e9+7
#define lowbit(i) ((i)&(-i))

class zkw_tree{
private:
vector<ll> sum;
vector<ll> tg;
int m;

public:
zkw_tree(const int& n);
zkw_tree(const vector<int>& arr);
void modify(int l, int r, ll v);
ll query(int l, int r);
};

zkw_tree::zkw_tree(const int& n){
    for(m=1; m<n+2; m<<=1); // m 
    sum.resize(m<<1);
    tg.resize(m<<1);
    
    for(int i=1; i<=n; i++) sum[m+i]=0;
    for(int i=m-1; i; i--){
        sum[i] = sum[i<<1] + sum[i<<1 |1];
    }
}

zkw_tree::zkw_tree(const vector<int>& arr){
    int n=arr.size()-1;
    for(m=1; m<n+2; m<<=1); //
    sum.resize(m<<1);
    tg.resize(m<<1);
    
    for(int i=1; i<=n; i++) sum[m+i]=arr[i];
    for(int i=m-1; i; i--)  sum[i]=sum[i<<1] + sum[i<<1 |1];
}
void zkw_tree::modify(int l, int r, ll v){
    ll len=1;
    ll lc=0;
    ll rc=0;
    
    l=l+m-1;
    r=r+m+1;  // +m+1
    while( l^r^1){
        if( ~l &1) {tg[l+1]+=v; lc+= len;}
        if(r &1) {tg[r-1]+=v; rc+=len;}
        sum[l>>1] += v*lc;
        sum[r>>1] += v*rc;

        l>>=1;
        r>>=1;
        len<<=1;
    }

    for(lc+=rc, l>>=1; l; l>>=1){
        sum[l] += v*lc;
    }
}


ll zkw_tree::query(int l, int r){
    ll res=0;
    ll len=1;
    ll lc=0;
    ll rc=0;
    
    l=l+m-1;
    r=r+m+1;  // +m+1
    while(l^r^1){
        if(~l&1) {res +=sum[l+1] +tg[l+1]*len; lc+=len;}
        if(r&1) {res += sum[r-1]+tg[r-1]*len; rc+=len;}
        res += tg[l>>1]*lc;
        res += tg[r>>1]*rc;

        l>>=1;
        r>>=1;
        len<<=1;
    }

    for(lc+=rc, l>>=1; l; l>>=1){
        res += tg[l]*lc;  // the left mulitply
    }
    return res;
}

class Solution{
public:
vector<pair<int, int>> pos;
int cur=0;
void dfs(vector<vector<int>>& sd, int i){
    cur++;
    pos[i].first=cur;
    for(auto it: sd[i]){
        dfs(sd, it);
    }
    pos[i].second = cur;
}

vector<int> bonus(int n, vector<vector<int>>& ls, vector<vector<int>>& opt) {
    vector<vector<int>> sd(n+1);
    cur=0;
    pos.resize(n+1);
    for(auto& e: ls){
        sd[e[0]].push_back( e[1]);
    }
    dfs(sd, 1);
    
    zkw_tree zk(n+2);
    vector<int> ans;
    for(auto& e: opt){
        if(e[0]==1) zk.modify(pos[e[1]].first, pos[e[1]].first, e[2]);
        else if(e[0]==2) zk.modify(pos[e[1]].first, pos[e[1]].second, e[2]);
        else if(e[0]==3) ans.push_back( zk.query(pos[e[1]].first, pos[e[1]].second));
    }
    return ans;
}
};


// raw code 
#define ll long long
class zkw_tree
{
private:
    vector<ll> sum;
    vector<ll> tg;
    int m;
public:
    zkw_tree(const vector<int> &array);
    zkw_tree(const int& n);
    void modify(int l, int r, ll del);
    ll query(int l, int r);
};
zkw_tree::zkw_tree(const int& n)
{
    for (m = 1; m < n + 2; m <<= 1);
    sum.resize(m << 1);
    tg.resize(m << 1);
    for (int i = 1; i <= n; i++)
    {
        sum[m+i] = 0;
    }
    for (int i = m - 1; i; i--)
    {
        sum[i] = sum[i << 1] + sum[i << 1 | 1];
    }
}
zkw_tree::zkw_tree(const vector<int> &array)
{
    int n = array.size() - 1;
    for (m = 1; m < n + 2; m <<= 1);
    sum.resize(m << 1);
    tg.resize(m << 1);
    for (int i = 1; i <= n; i++)
    {
        sum[m+i] = array[i];
    }
    for (int i = m - 1; i; i--)
    {
        sum[i] = sum[i << 1] + sum[i << 1 | 1];
    }
}
void zkw_tree::modify(int l, int r, ll del)
{
    ll len = 1, lc = 0, rc = 0;
    for (l = l + m - 1, r = r + m + 1; l ^ r ^ 1; l >>= 1, r >>= 1, len <<= 1)
    {
        if (~l & 1)
            tg[l + 1] += del, lc += len;
        if (r & 1)
            tg[r - 1] += del, rc += len;
        sum[l >> 1] += del * lc;
        sum[r >> 1] += del * rc;
    }
    for (lc += rc, l >>= 1; l; l >>= 1)
        sum[l] += del * lc;
}


ll zkw_tree::query(int l, int r)
{
    ll res = 0, len = 1, lc = 0, rc = 0;
    for (l = l + m - 1, r = r + m + 1; l ^ r ^ 1; l >>= 1, r >>= 1, len <<= 1)
    {
        if (~l & 1)
            res += sum[l + 1] + tg[l + 1] * len, lc += len;
        if (r & 1)
            res += sum[r - 1] + tg[r - 1] * len, rc += len;
        res += tg[l >> 1] * lc;
        res += tg[r >> 1] * rc;
    }
    for (lc += rc, l >>= 1; l; l >>= 1)
        res += tg[l] * lc;
    return res;
}

class Solution {
public:
    vector<pair<int, int>> pos;
    int cur = 0;
    void dfs(vector<vector<int>>& sd, int i)
    {
        ++cur;
        pos[i].first = cur;
        for(auto it: sd[i])
        {
            dfs(sd, it);
        }
        pos[i].second = cur;
    }
    vector<int> bonus(int n, vector<vector<int>>& leadership, vector<vector<int>>& operations) {
        vector<vector<int>> sd(n+1);
        cur = 0;
        pos.resize(n+1);
        for(auto& it : leadership)
        {
            sd[it[0]].push_back(it[1]);
        }
        dfs(sd, 1);
        zkw_tree bonus(n+2);
        vector<int> ans;
        for(auto &it:operations)
        {
            if(it[0] == 1)
            {
                bonus.modify(pos[it[1]].first,pos[it[1]].first, it[2]);
            }
            if(it[0] == 2)
            {
                bonus.modify(pos[it[1]].first, pos[it[1]].second, it[2]);
            }
            if(it[0] == 3)
            {
                ans.push_back(bonus.query(pos[it[1]].first,pos[it[1]].second) % 1000000007);
            }
        }
        return ans;
    }
};


```


LCP 10 二叉树任务调度

time n, space n
```cpp
double minimalExecTime(TreeNode* root) {
    auto p = dfs(root);
    return p.first - p.second;
}

pair<int, double> dfs(TreeNode* root){
    if(!root) return {0, 0.0};
    auto l = dfs(root->left);
    auto r = dfs(root->right);
    
    int a = l.first;
    int c = r.first;
    double b = l.second;
    double d = r.second;
    int tot = a + c + root->val;
    
    if((c-2*d<=a && a<=c) || (a-2*b<=c && c<=a)){ // why
        return {tot, (a+c)/2.0};
    }
    if(a-2*b>c){
        return {tot, b+c};
    }
    return {tot, a+d};

}
```



LCP 13 寻宝


bfs + 状压dp求解hamilton路径

bfs time O(mn)
compute tsDist arr O(pqmn)
compute ttDist arr O(pqq)
dp O(q*q * 2^q)



time O(ms + mmo + mm*2^m)
space O(s+bs + m*2^m)

```cpp
int minimalSteps(vector<string>& maze){
    
}

```

https://leetcode-cn.com/problems/xun-bao/solution/bfs-zhuang-tai-ya-suo-dp-by-haozheyan97/

```cpp
struct Point{
    int x;
    int y;
    Point(int x_, int y_):x(x_), y(y_){}
    void print(){
        printf("%d %d\n", x, y);
    }
};

class Solution{
public:
    static const int maxn = 102;
    int dist[maxn][maxn], vis[maxn][maxn];
    Point q[maxn*maxn], S, T, puzzle[20],stone[50];

    int distS[50], distT[50], distm[20][50], distBm[20][20];
    int dp[1<<16][16];
    
    int dx[4] = {1, 0, -1, 0};
    int dy[4] = {0, 1, 0, -1};

    void bfs(vector<string>& maze, int& n, int& m, Point start){
        
    }

    int minimalSteps(vector<string>& maze){
        int n = maze.size();
        int m = maze[0].size();
        int curp;
        int curs;
        curp = curs = 0;
        
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                if(maze[i][j]=='S') S = Point(i,j);
                if(maze[i][j]=='T') T = Point(i,j);
                if(maze[i][j]=='m') puzzle[curp++] = Point(i,j);
                if(maze[i][j]=='O') stone[curs++] = Point(i,j);
            }
        }
    }

};

```


LCP 14 切分数组

每个数，需要遍历他所有质因子，并在此过程中进行 DP 状态转移，所以需将每个数不断除以它的最大质因子，此处的时间复杂度为 O(N\log m)O(Nlogm)。

预处理1~ 10^6 以内每个数最大质因子的时间复杂度为 O(m)




质因数分解和dp

test case: [2,3,3,2,3,3]
test case: [326614, 489921]

假如当前nums[i]有p这个质因子，上一次以p结尾的切割我们可以应用到这一次来
例如nums[]={2,3,3,2}

dp[2]=1,dp[3]=2,dp[3]=2,dp[2]=1;

以上即为dp数组的变换。
上一次以p结尾和这一次以p结尾能用同一个答案是因为他们的公因子%p==0

ans=min(前i-1个的最优解+1，dp[p])

```cpp
// 
class Solution 
{
private:
    const static int N=1e6+10;
    const static int m=1e6;
    int prim[N];
    int cnt=0;
    int min_fa[N];
    int f[N];
    bool vis[N];

public:
    void init()
    {
        for(int i=2; i<=m; i++)
        {
            if(!vis[i]){
                prim[++cnt]=i;
                min_fa[i]=i;
            }
            for(int j=1; j<=cnt && i*prim[j]<=m; j++)
            {
                vis[ i*prim[j] ]=1;
                min_fa[i*prim[j]] = prim[j];
                if( i%prim[j]==0) break;
            }
        }
    }
    int splitArray(vector<int>& nums) 
    {
        init();
        int ans=0x3fffffff, // why
        int now=0;
        memset(f,0x3f,sizeof(f));

        for(auto i: nums)
        {
            int x=i;
            ans=0x3fffffff;

            while(x>1)
            {
                f[min_fa[x]]=min( f[min_fa[x]], now+1);
                ans=min(ans, f[min_fa[x]]);  //
                x/=min_fa[x];
            }
            now=ans;
        }
        return ans;
    }
};

```


```cpp
// easy
for(int i=2; i<=1e6; i++){
    if(!not_prime[i]){
        prime[++tot] = i;
        for(int j=i+1; j<1e6; j+=i){
            not_prime[j]=1;
        }
    }
}
// pre数组保存的就是每个数的所有质因子中最大的那个

// modified
while( s[i]!=-1){
    int p = pre[s[i]];
    s[i] = s[i]/p;
}
// time O(logn/(loglogn))
```

https://leetcode-cn.com/problems/qie-fen-shu-zu/solution/
// bug
```cpp
// time O(nlogm + m), space m
// 有bug 并且超时
class Solution{
public:
    vector<int> pre;
    Solution(){
        pre = vector<int>(1e6+1, 0); // 
        for(int i=2; i<=1e6; i++){
            if(!pre[0]){
                for(int j=i; j<=1e6; j+=i){
                    pre[j]=i;
                }
            }
        }
    }

    int splitArray(vector<int>& s){
        int n = s.size();
        vector<int> dp(n,n); //
        vector<int> dpp(1e6, n);
        for(int i=0; i<n; i++){
            while( s[i]!=1){
                int p = pre[s[i]];
                s[i] = s[i]/p;
                while(pre[s[i]]!= p){
                    s[i] = s[i]/p;
                }

                dpp[p] = min(dpp[p], i? dp[i-1]: 0);
                dp[i] = min(dp[i], dpp[p]+1);
            }
        }
        for(int i=0; i<n; i++) printf("%d %d\n", dp[i], dpp[i]);
        return dp[n-1];
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

LCP 26 导航装置

树形dp 或者 dfs
```cpp
int res = 1;
bool tip = true;

int dfs(TreeNode* root, int k){
    if(root->left && root->right){
        int a = dfs(root->left, k+1);
        int b = dfs(root->right, k+1);
        if(a==0 || b==0){
            if(tip && k>0){
                res++;
                tip = false;
            }
            return 1;
        }
        res++;
        return 2;
    }

    if(root->left) return dfs(root->left, k+1);
    if(root->right) return dfs(root->right, k+1);
    return 0;
}

int navigation(TreeNode* root){
    if(dfs(root, 0)==2)  res--;
    return res;
}

```

LCP 20 快速公交

记忆化dfs

从终点返回到起点有多种方法
直接全部步行
直接坐公交车
向前走几步再坐公交
向后走几步再坐公交

```cpp
typedef long long ll;

class Solution{
    ll mod = 1e9+7;
    unordered_map<int, ll> mem;
    ll inc;
    ll dec;
    vector<int> jump;
    vector<int> cost;

public:
    int busRapidTransit(int target, int inc_, int dec_, vector<int>& jump_, vector<int>& cost_){
        mem.clear();
        inc = inc_;
        dec = dec_;
        jump = jump_;
        cost = cost_;
        return dfs(target) % mod;
    }

    ll dfs(int cur){
        if(cur==0) return 0;
        
        if(mem.find(cur)!=mem.end()) return mem[cur];

        ll ans = inc * cur;
        if(cur==1) return ans;
        
        int n = jump.size();
        for(int i=0; i<n; i++){
            ll j = jump[i];
            ll c = cost[i];

            int step1 = cur % j;
            int a = cur/ j;
            if(cur- step1>0 ){
                ans = min(ans, step1* inc + c + dfs(a));
            }
            // back and get to bus
            if(step1>0){
                int step2 = j - cur%j;
                a = cur/j + 1;
                ans = min(ans, step2*dec + c + dfs(a));
            }
        }
        mem[cur] = ans;
        return ans;
    }
};

```

LCP 16 游乐园的游览计划
在图中找到两个三角形（边可以重复），两个三角形至少需要一个点相连，使得最终所有点的权值和最大。

官方 time m^(3/2), space m


根据度数分段+权重最大的前三条边至少取一条

priority_queue 大顶堆。top元素最大。
edges 处都加上引用，能将time 从 1010 ms 减少到 390 ms


```cpp
// 390 ms
int maxWeight(vector<vector<int>>& edges, vector<int>& val) {
    int n = val.size();
    int m = edges.size();
    vector<unordered_map<int, int>> adj(n);  // second, id
    // weight desc, time mlogm
    sort(edges.begin(), edges.end(), [&](vector<int>& a, vector<int>& b){
        return val[a[0]] + val[a[1]] > val[b[0]]+val[b[1]];
    });
    int res=0;

    int id=0;
    for(auto& e: edges){  // &
        adj[e[0]][e[1]] =id;
        adj[e[1]][e[0]] =id;
        id++;
    }
    for(int i=0; i<n; i++){
        unordered_map<int, int>& s= adj[i];
        vector<int> pos;
        int ns= s.size();

        if( ns * ns <=m){  // degree < sqrt(m)
            for(auto [j, ji]: s){
                for(auto [k, ki]: s){
                    if(j!=k && adj[j].count(k)>0 ){
                        pos.emplace_back( adj[j][k]); // edge j-k
                    }
                }
            }
        }
        else{ // degree > sqrt(m)
            int j=0;
            for(auto& e: edges){
                int u =e[0];
                int v =e[1];
                if( s.count(u)>0 && s.count(v)>0 ){
                    pos.emplace_back(j);
                }
                j++;
            }
        }
        // id smaller, weight bigger.
        priority_queue<int> pq;
        for(int j: pos){
            if(pq.size()< 3 || j< pq.top()) pq.push(j); // 
            if(pq.size()>3 ) pq.pop();
        }
        while( !pq.empty()){
            auto& e = edges[ pq.top()];
            int x = e[0];
            int y =e[1];
            int sum= val[i] + val[x] + val[y]; //
            pq.pop();
            // other edge
            for(int j: pos){
                auto& b= edges[j];
                int u=b[0];
                int v=b[1];
                int cur=sum;
                if(u!=x && u!=y) cur+= val[u];
                if(v!=x && v!=y) cur+= val[v];
                res = max(res, cur);
            }
        }
    }
    return res;
}
```


LCP 24 数字游戏

维护中位数，time nlogn, space n
两个优先队列来实时维护中位数

```cpp
typedef long long ll;
static constexpr int mod=1e9+7;
vector<int> numsGame(vector<int>& arr){
    int n=arr.size();
    if(n==1) return {0};
    
}
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