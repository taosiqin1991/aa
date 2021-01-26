
acm系列


### 求解反素数
https://zhuanlan.zhihu.com/p/41759808


给定因子数，求满足因子数恰好等于这个数的最小数。
给定一个n， 求n 以内因子数量最多的数。

```cpp
#include <stdio.h>
#define ULL unsigned long long
#define INF ~0ULL

int p[16]={2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

ULL ans;
int n;

void dfs(int depth, int tmp, int num, int up){
    if( num>n || depth>=16) return;
    
    if( num==n && ans > tmp){
        ans = tmp;
        return ;
    }
    for(int i=1; i<=up; i++){
        if( tmp/p[depth] > ans) break;

        tmp = tmp * p[depth];
        dfs(depth+1, tmp, num*(i+1), i);
    }

}

void test(){
    n = 50;
    dfs(0, 1, 1, 60);

    printf("%lld\n", ans);
}

```



```cpp
int p[16]={2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};

ULL ans;
int n;
ULL ans_num; // ans_sum是ans的因子数

void dfs(int depth, ULL tmp, ULL num, int up(){
    if( depth >= 16 || tmp>n) return ;
    
    if( num > ans_num){
        ans = tmp;
        ans_num = num;
    }
    if( num==ans_num && ans > tmp){
        ans = tmp;
    }

    for(int i=1; i<=up; i++){
        if( tmp * p[depth]>n) break;
        
        tmp = tmp * p[depth];
        dfs( depth+1, tmp, num*(i+1), i);
    }
    return ;
}


```


### 矩阵快速幂

矩阵乘法

```cpp
struct matrix{
    ll x[105][105];
    matrix(){ memset(x, 0, sizeof(x));}
};

matrix multiply(matrix& a, matrix& b){
    matrix c;
    
    for(int i=1; i<=n; i++){
        for(int j=1; j<=n; j++){
            
            for(int k=1; k<=n; k++){
                c.x[i][j] += a.x[i][k] * b.x[k][j];
            }
        }
    }
    return c;
}


int f(int a, int n){
    int res = 1;
    while(n>0){
        if( n&1) res = res * a;
        
        a = a*a;
        n>>=1;
    }
    return res;
}

matrix mpow(matrix& a, ll m){
    matrix res;
    
    for(int i=1; i<=n; i++) res.x[i][i] = 1; // eye

    while(m>0){
        if(m & 1) res = multiply(res, a);

        a = multiply(a, a);
        m >>=1;
    }
    return res;
}
```

### 树上问题，最近公共祖先，倍增算法
lca问题
min dis(u, v) = depth(u) + depth(v) - 2*depth(LCA(u,v))

LCA常见求法有：倍增算法、、Tarjan算法、ST算法、树链剖分。

朴素算法
depth(u, v) >= depth(u) & depth(v)

倍增转移方程
fa[i][j] = fa[ fa[i][j-1] ][j-1] ( j<=log_2^(depth))
向上取整， |log_2^(depth)|+1

倍增预处理time O(nlogn)，包含询问内的最差复杂度为 O( (n+q)logn)

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;

#define NN 500005
vector<int> e[NN];
int depth[NN];
int fa[NN][25];
int lg[NN];

// s is cur node, fn is parent.
void dfs(int s, int fn){
    fa[s][0] = fn;
    dep[s] = dep[fn] + 1;
    
    for(int i=1; i<=lg[dep[s]]+1; i++){
        fa[s][i] = fa[ fa[s][i-1]][i-1]; // 
    }
    for(int i=0; i<e[s].size(); i++){
        if( e[s][i]!=fn )  dfs(e[s][i], s); // to down
    }
}

// time logn
void pre(int n){
    lg[1] = 0;
    lg[2] = 1;
    for(int i=3; i<=n; i++){
        lg[i] = lg[i/2] + 1;
    }
}

int lca(int x, int y){
    if( dep[x]< dep[y]) swap(x, y);  // make sure x depth is bigger.
    
    while( dep[x]>dep[y]){
        x = fa[x][lg[dep[x]- dep[y]] ];
    }

    if( x==y) return ;
    
    for(int i=lg[dep[x]]; i>=0; i--){
        if( fa[x][i] !=fa[y][i]){
            x = fa[x][i];
            y = fa[y][i];
        }
    }
    return fa[x][0];
}

```
### 匈牙利算法
匈牙利算法有 递归dfs 和非递归 bfs 两种实现方法。
前者写起来简单。
后者在速度上更优。尤其是在 完全随机数据的稀疏图中，bfs的速度甚至接近是 dfs的 两倍。

https://www.renfei.org/blog/bipartite-matching.html

```cpp
bool hungary(int now){ // now is cur customer
    for(int i=0; i<edge[now].size(); i++){
        int to = edge[now][i];

        if( !vis[to]){
            vis[to] = 1;
            if( !food[to] | hungary( food[to])){ // if not arranged
                food[to] = now;
                return 1;
            }
        }
    }
    return 0;
    
}

void test(){
    int ans = 0;
    for(int i=1; i<=n; i++){

        memset( vis, 0, sizeof(vis));
        if( hungary(i))   ans++; 
    }
}

```


### 图的遍历

邻接矩阵的空间复杂度是 O(n^2),对于点较多的图论题，大概率会超出内存的限制。

```cpp
int edge[maxn][maxn];

void  add_a_edge(int start, int end, int value){
    edge[start][end] = value;
    // 若是无向图则 edge[start][end] = edge[end][start] = value
    // 若无权值，则可用 1表示有边，0表示无边
}

```

邻接表，用c++ vector容器来表示。邻接表比邻接矩阵常用得多。

有n个点则开 n大小的数组vector。

若需要表示权值，可以使用结构体。

```cpp
vector<int> edge[maxn];
void add_a_edge(int start, int end){
    edge[start].push_back( end);
}

struct node{
    int to;
    int value;
};
vector<node> edge[maxn];

void add_a_edge(int start, int end, int value){
    node a;
    a.to = end;
    a.value = value;
    edge[start].push_back(a);
}

```


```cpp
vector<int> edge[maxn];  // table
bool vis[maxn];

void dfs(int now){
    for(int i=0; i<edge[now].size(); i++){
        int next = edge[now][i];
        
        if( !vis[next]){

            vis[next] = 1;
            dfs(next);
        }
    }
}

void bfs(int start){
    queue<int> q;
    q.push(start);
    
    while( !q.empty()){
        int now = q.front();
        q.pop();

        for(int i=0; i<edge[now].size(); i++){
            int next = edge[now][i];

            if(!vis[next]){
                vis[next]=1;
                q.push( next);
            }
        }
    }
}


```

### 稀疏表 ST


处理区间GCD时，ST表与线段树的时间复杂度基本相近，但前者却显然要好写得多。

sparse table，常用来解决可重复贡献问题。
常见的可重复贡献问题有： 区间最值、区间按位和、区间按位或、区间GCD等。

区间和不是此类问题。

可重复贡献问题： 要你求10个数中的最大数，你完全可以先求前6个数的 max ，再求后7个数的 max，然后再对所求的两个最大数求 max 。虽然中间有几个数被重复计算了，但并不影响最后的答案。

用f[i][j] 表示[i, i+2^(j-1)]内的最值。
f[i][j] = max(f[i][j-1], f[i+2^{j-1}][j-1])
```cpp

n = read();
m = read();
pre();

for(int i=1; i<=n; i++){
    f[i][0] = read();
}
// 2^21 > 10^6
for(int j=1; j<21; j++){
    for(int i=1; i+ (1<<j) -1<=n; i++){
        f[i][j] = max(f[i][j-1], f[i+ (1<<j-1)][j-1] );
    }
}

```


```cpp
// bitwise and
f[i][j] = f[i][j-1] & f[i+(1<<(j-1))][j-1];
ans = f[l][lg] & f[r-(1<<lg)+1 ][lg];

// gcd
f[i][j] = gcd( f[i][j-1], f[i+(1<<(j-1))][j-1] );
ans = gcd( f[l][lg], f[r-(1<<lg)+1][lg]); 

```


### 倍增算法

在你面前的桌子上，摆着无数个重量为任意整数的胡萝卜；
接着告诉你一个数字 n，问你要怎么挑选，使得你选出的胡萝卜能够表示出 [1,n] 区间内的所有整数重量？

只要选择1，2， 4， 8， 16，的胡萝卜，就能表示【1， 2^n-1]内的所有数。值需要logn 个胡萝卜。

to[x][i] = to[ to[x][i-1] ][i-1]
carrot[x][i] = carrot[x][i-1] + carrot[ to[x][i-1] ][i-1]

```cpp
for(int x=1; x<=n; x++){
    to[x][0] = (x+k) % n + 1;
    carrot[x][0] = num[x];
}
for(int i=1; i<=64; i++){
    for(int x=1; x<=n; x++){
        to[x][i] = to[ to[x][i-1]][i-1];
        carrot[x][i] = carrot[x][i-1] ++ carrot[ to[x][i-1]][i-1];
    }
}

int p = 0;
int now = 1;
int ans = 0;
while( m){

    if( m & (1<<p)){
        ans += carrot[now][p];
        now = to[now][p];
    }
    m ^= (1<<p); // clear p-1 bit pos
    p++;
}
```

### 数论逆元
a-b = a+(-b);
a/b = a* (1/b);

a/b(mod m) = a * inv(b)(mod m)

b * inv(b) = 1(mod m)

费马小定理求逆元
```cpp
long long quickpow(long long a, long long n){
    a %=m;
    long long res = 1;
    while(n>0){
        if( n&1) res = (res*a)%mod;
        a = (a*a)%mod;
        n >>=1;
    }
    return res;
}

long long inv(long long a){
    return quickpow(a, mod-2);
}
```

拓欧求法
```cpp
long long exgcd(long long a, long long b, long long& x, long long& y){
    if(b==0){
        x=1;
        y=0;
        return a;
    }
    
    long long ans = exgcd(b, a%b, x, y);
    long long tmp = x;
    x = y;
    y = tmp - a/b*y;
    return ans;
}

long long inv(long long a, long long p){
    long long x, y;
    long long g = exgcd(a, p, x, y);
    
    if( g!=1) return -1;
    else return (x%p + p)%p;
}

```

### 树上问题-树的直径

两次dfs求直径

1437 树的最长路径
1985 奶牛马拉松

```cpp
#include<bits/stdc++.h>
using namespace std;
vector<int>edge[100005];
int vis[100005];
int dis[100005];
void dfs(int st)
{
    for(int i=0;i<edge[st].size();i++)
    {
        int to=edge[st][i];
        if(!vis[to])
        {
            vis[to]=1;
            dis[to]=dis[st]+1;//注意，本代码计算的是无权树的直径，所以边权为1
            //如果是有权树，则这里的1要改为边权
            dfs(to);
        }
    }
}
int main()
{
    ios::sync_with_stdio(false);
    int n;
    cin>>n;//n个点
    for(int i=1;i<=n-1;i++)//因为是树，有n-1条边
    {
        int u,v;
        cin>>u>>v;
        edge[u].push_back(v);
        edge[v].push_back(u);//无向图存储，若是有权树还要用结构体
    }
    dfs(1);//从1出发dfs一边
    int maxlen=0,Q,W,ans=0;
    for(int i=1;i<=n;i++)
    {
        if(dis[i]>maxlen) maxlen=dis[i],Q=i;
        dis[i]=vis[i]=0;
    }//找到直径的一个端点Q
    dfs(Q);//从Q出发
    for(int i=1;i<=n;i++)
        if(dis[i]>ans) ans=dis[i],W=i;//找到另一个端点W，同时得到直径长度
    return 0;
}

```
### 最小生成树

prim 在稠密图中比 Kruskal优，在稀疏图中劣。

prim_heap 在任何时候的时间复杂度都要优于朴素prim 和 Kruskal
(因为一遍图的 边数总是大于节点数的)，但优先队列维护的代价是更大的内存空间使用。

一般情况下就用prim_heap吧. time mlogm

prime_heap

```cpp
#include <bits/stdc++.h>
#define inf 1e8

using namespace std;
struct node{
    int to;
    int w;
};

vector<node> edge[5005];
int m,n;

int dis[5005];
int vis[5005];
int ans = 0;
typedef pair<int,int> pa;
priority_queue<pa, vector<pa>, greater<pa>> q;

void prim_heap(){
    for(int i=1; i<=n; i++){
        dis[i] = inf;
    }

    q.push( make_pair(0, 1));
    int nown = 0;
    
    while( !q.empty() && nown<n ){
        int now = q.top().second();
        int v = q.top().first;
        q.pop();
        
        if( vis[now]) continue;
        
        vis[now] = 1;
        nown++;
        ans += v;
        for(int i=0; i<edge[now].size(); i++){
            int to = edge[now][i].to;
            int w = edge[now][i].w;
            
            if( !vis[to] && w<dis[to]){
                dis[to] = w;
                q.push( make_pair(w, to));
            }
        }
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin >> n >> m;
    
    for(int i=1; i<=m; i++){
        int x, y, z;
        cin >> x >> y >> z;
        
        node in;
        in.to = y;
        in.w = z;
        
        edge[x].push_back(in);
        in.to = x;
        edge[y].push_back(in);
    }
    
    prim_heap();
    cout << ans << endl;
    return 0;
}


```

prim
```cpp

```

kruskal
```cpp
#include <bits/stdc++.h>
using namespace std;

struct node{
    int x;
    int y;
    int l;
};

vector<node> edge;

int m, n, fa[5005]; // union-find

bool cmp(node a, node b){
    return a.l < b.l;
}

void init(){
    for(int i=1; i<=n; i++){
        fa[i] = i;
    }
}


void ffa(int x){
    if( fa[x] != x){
        fa[x] = ffa( fa[x]);
        return fa[x];
    }
}


void unite(int x, int y){
    x = ffa(x);
    y = ffa(y);
    fa[y] = x;
}

int main(){
    ios::sync_with_stdio(false);
    cin >> n >> m;

    init();
    node in;
    
    for(int i=1; i<=m; i++){
        cin >> in.x >> in.y >> in.l;
        edge.push_back(in);
    }
    sort(edge.begin(), edge.end(), cmp);
    
    int nowk = n;
    int ans = 0;
    for(int i=0; i<m; i++){
        if( ffa(edge[i].x) != ffa(edge[i].y)){ // not united
            nowk--;
            ans += edge[i].l;
            unite(edge[i].x, edge[i].y);
        }

        if( nowk==1){
            cout << ans << endl;
            return 0;
        }
    }
    cout << "orz" << endl;
    return 0;

}


```


### 拓扑排序

对于有向无环图DAG，对图中所有节点进行排序，使图中所有边， 其起点在点序列中都在终点的前面。此为拓扑排序。

拓扑排序出的序列不唯一。

拓扑排序常用于解决有向图中的依赖解析问题。

也就是说，像“做A任务时必须先完成B任务”这样的限制条件，通过拓扑排序能够得到一个满足执行顺序的序列。
当不存在满足条件的序列时，也就代表图中出现了环，形成了循环依赖的情况。
循环依赖的意思是：做A任务必须先完成B任务，但做B任务也必须先完成A任务。则此时对于问题是无解的。

拓扑排序用 bfs来实现的。

```cpp
int in[n];
queue<int> q;
vector<int> edge[n];
vector<int> ans;

void topo_order(){
    for(int i=0; i<n; i++){
        if( in[i]==0) q.push(i);
    }

    while( !q.empty()){
        int f = q.front();
        q.pop();
        
        ans.push_back(f);
        for(int i=0; i<edge[f].size(); i++){
            in[ edge[f][i] ]--;
            if( in[ edge[f][i]]==0)  q.push( edge[f][i]);
            // 若入度为0，则加入队列
        }
    }
    
    if( ans.size()!=n)  cout << "no solution." << endl; 
    
}


```


### 组合数学-康托展开

康托展开用于求解以下问题：
给定一个1~n的排列，求该排列在1~n的全排列中，按字典序排名多少位。

举个例子：在1~5的排列中，求解34152的排名。

```cpp
int fac[100] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800};

int kangtuo(int n, vector<int>& s){
    int sum = 0;
    for(int i=0; i<n; i++){
        int t=0;

        for(int j=i+1; j<n; j++){
            if(s[i] > s[j]){
                t++;
            }
        }
        sum += t * fac[n-i-1];
    }
    return sum+1;
}

```

逆康托展开
```cpp
int factorial[100] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800};

void decantor(int x, int n){
    vector<int> v;
    vector<int> a;
    
    for(int i=1; i<=n; i++)  v.push_back(i);

    for(int i=m; i>=1; i--){
        int r = x % factorial[i-1];
        int t = x / factorial[i-1];
        x = r;
        
        sort( v.begin(), v.end());
        a.push_back( v[t]);
        v.erase( v.begin() + t);
    }
    
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
