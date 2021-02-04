


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

bfs + 状压dp 求解哈密尔顿路径

["MMMMM","MS#MM","MM#TO"]

["......", "M....M", ".M#...", "....M.", "##.TM.", "...O..", ".S##O.", "M#..M.", "#....."]

```cpp
// bug 
const int INF = 0x3f3f3f;

class Solution{
public:
const int dir[5]={-1,0,1,0,-1};
int cnt=0;
int id_cnt=0;
unordered_map<int, int> id;
int m;
int n;

void bfs(vector<string>& maze, vector<vector<int>>& dist, int i ,int j){
    queue<pair<int, pair<int, int>>> q; // shortest_dist to point
    dist[i][j]=0;
    q.push( {0, {i,j}}); //

    while(q.size()){
        auto& e =q.front();
        q.pop();
        
        int len=e.first;
        int ei = e.second.first;
        int ej = e.second.second;
        for(int k=0; k<4; k++){
            int ki=ei +dir[k];
            int kj=ej +dir[k+1];
            if(ki<0 || kj<0 || ki>=m || kj>=n) continue;
            if(maze[ki][kj]=='#') continue;  // maze
            if(dist[ki][kj] != INF) continue; // visited
            
            dist[ki][kj] = len+1;
            q.push( {len+1, {ki, kj}});
        }
    }
}

int minimalSteps(vector<string>& maze){
    if(maze.size()==0 || maze[0].size()==0) return 0;
    m = maze.size();
    n = maze[0].size();
    int fi=0;
    int fj=0;
    int id_s=0;

    vector<vector<int>> g(m, vector<int>(n));
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            g[i][j] = cnt;
            if(maze[i][j]=='S' || maze[i][j]=='M'){
                if(maze[i][j]=='S')  id_s = id_cnt;
                id[id_cnt++] =cnt; //
            }
            else if(maze[i][j]=='T'){
                fi=i; fj=j;
            }

            cnt++;
        }
    }

    vector<vector<int>> d(id_cnt, vector<int>(id_cnt, INF)); // adj of S/M
    for(int i=0; i<id_cnt; i++) d[i][i]=0;
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            // M > O -> M
            if(maze[i][j]=='O'){
                vector<vector<int>> dist(m, vector<int>(n, INF));
                bfs(maze, dist, i, j);

                for(int u=0; u<id_cnt; u++){ // M -> O->M
                    for(int v=u+1; v<id_cnt; v++){
                        int p1=id[u]; int p2=id[v]; // two points
                        //
                        int x1 =p1/n; int y1 =p1%n;
                        int x2 =p2/n; int y2 =p2%n;
                        d[v][u]=d[u][v]= min(d[u][v], dist[x1][y1]+dist[x2][y2]); //
                    }
                }
            }
        }
    }

    vector<vector<int>> dist_t(m, vector<int>(n, INF));
    bfs(maze, dist_t, fi, fj);
    int res=INF;
    
    vector<vector<int>> f(1<<id_cnt, vector<int>(id_cnt, INF));
    f[1<<id_s][id_s]=0;
    // cout << id_cnt << endl;

    for(int i=0; i<(1<<id_cnt); i++){ // all state
        for(int j=0; j<id_cnt; j++){  // all rest point
            if( (i>>j) & 1){
                for(int k=0; k<id_cnt; k++){
                    
                    if(!(i>>k & 1) ){ // this rest point can't be reached.
                        f[i|(1<<k)][k] =min(f[i|(1<<k)][k], f[i][j]+d[j][k]);
                    }
                }
            }
        }
    }
    for(int j=0; j<id_cnt; j++){
        int p=id[j];
        int x=p/n; int y=p%n;
        res = min(res, f[(1<<id_cnt)-1][j] + dist_t[x][y]); //
    }
    return res>=INF? -1: res;
}
};
```


```cpp
// ok
const int INF = 0x3f3f3f / 2;
class Solution {
public:
int cnt = 0;
int id_cnt = 0;
unordered_map<int, int> id;
int n, m;
const int dir[5]={-1,0,1,0,-1};

void bfs(vector<string>& maze, vector<vector<int>>& dist, int x, int y)
{
    queue<pair<int, pair<int, int>>> q;
    dist[x][y] = 0;
    q.push({0, {x, y}});

    while (q.size())
    {
        int dx[] = {0, 1, 0, -1}, dy[] = {1, 0, -1, 0};
        auto t = q.front();
        q.pop();

        int len = t.first;
        int xx = t.second.first, yy = t.second.second;
        for (int i = 0; i < 4; i ++ )
        {
            int a = xx + dx[i], b = yy + dy[i];
            if (a < 0 || a >= n || b < 0 || b >= m) continue;
            if (maze[a][b] == '#') continue;
            if (dist[a][b] != INF) continue;
            dist[a][b] = len + 1;
            q.push({len + 1, {a, b}});
        }
    }
}


int minimalSteps(vector<string>& maze) {
    n = maze.size(), m = maze[0].size();
    int fi = 0, fj = 0;
    int id_s = 0;

    vector<vector<int>> g(n, vector<int>(m));
    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < m; j ++ )
        {
            g[i][j] = cnt;
            if (maze[i][j] == 'S' || maze[i][j] == 'M')
            {
                if (maze[i][j] == 'S') id_s = id_cnt;
                id[id_cnt ++ ] = cnt;
            }
            else if (maze[i][j] == 'T')
                fi = i, fj = j;
            cnt ++ ;
        }
    
    vector<vector<int>> d(id_cnt, vector<int>(id_cnt, INF));
    for (int i = 0; i < id_cnt; i ++ )  d[i][i] = 0;

    for (int i = 0; i < n; i ++ )
        for (int j = 0; j < m; j ++ )
            if (maze[i][j] == 'O')
            {
                vector<vector<int>> dist(n, vector<int>(m, INF));
                bfs(maze, dist, i, j);

                for (int u = 0; u < id_cnt; u ++ )
                    for (int v = u + 1; v < id_cnt; v ++ )
                    {
                        int p1 = id[u], p2 = id[v];
                        int x1 = p1 / m, y1 = p1 % m;
                        int x2 = p2 / m, y2 = p2 % m;

                        d[v][u] = d[u][v] = min(d[u][v], dist[x1][y1] + dist[x2][y2]);
                    }
            }

    vector<vector<int>> dist_t(n, vector<int>(m, INF));
    bfs(maze, dist_t, fi, fj);
    int res = INF;

    vector<vector<int>> f(1 << id_cnt, vector<int>(id_cnt, INF));
    f[1 << id_s][id_s] = 0;

    for (int i = 0; i < 1 << id_cnt; i ++ )
        for (int j = 0; j < id_cnt; j ++ )
            if (i >> j & 1)
                for (int k = 0; k < id_cnt; k ++ )
                    if (!(i >> k & 1))
                        f[i | (1 << k)][k] = min(f[i | (1 << k)][k], f[i][j] + d[j][k]);
    
    for (int j = 0; j < id_cnt; j ++ )
    {
        int p = id[j];
        int x = p / m, y = p % m; 
        res = min(res, f[(1 << id_cnt) - 1][j] + dist_t[x][y]);
    }
    if (res >= INF) return -1;
    else return res;
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

LCP 15 游乐园的迷宫

凸包 + 叉积

```cpp
// 108 ms defeat 77%
using ll=long long;
struct Point{
    ll x;
    ll y;
    Point operator -(const Point& p){
        return {x-p.x, y-p.y};
    }
    
    ll operator *(const Point& p){
        return x*p.y - y*p.x; // cross mult
    }
};

vector<int> visitOrder(vector<vector<int>>& points, string dir){
    int n=points.size();
    vector<int> res;
    vector<Point> vp;
    vector<int> vis(n, 0);
    for(auto& e: points) vp.push_back({e[0], e[1]});
    
    int k=0;
    for(int i=1; i<n; i++){
        if(vp[i].x < vp[k].x)  k=i;
    }
    res.push_back(k);
    vis[k]=1;
    
    for(int i=0; i<n-2; i++){
        int t=-1;
        for(int j=0; j<n; j++){
            if(vis[j]) continue;
            if(t==-1) t=j;
            else{
                if(dir[i]=='L'){
                    if( (vp[t]-vp[k])*(vp[j]-vp[k])<0 ) t=j;
                }
                else{
                    if( (vp[t]-vp[k])*(vp[j]-vp[k])>0) t=j;
                }
            }
        }
        k =t;
        vis[t]=1;
        res.push_back(t);
    }
    for(int i=0; i<n; i++){
        if(!vis[i]) res.push_back(i);
    }
    return res;
}
```

LCP 最小跳跃次数

bfs, time n, space n
dp,  time n, space n

[2,5,1,1,1,1]

```cpp
// 220 ms
int minJump(vector<int>& jump){
    int n=jump.size();
    vector<int> dp(n, 0);
    for(int i=n-1; i>=0; i--){
        if(jump[i]+i>=n ) dp[i]=1;
        else  dp[i] = dp[jump[i]+i ]+1; // 
        
        for(int j=i+1; j<n && j<i+jump[i] && dp[j]>dp[i]; j++){ // 
            dp[j] =dp[i]+1;
        }
    }
    return dp[0];
}
```

糟糕的写法
// 习惯用 -1 作为行遍历结束的标志
```cpp
// 220 ms
static const int MAXN=1e6+5; // static must
int minJump(vector<int>& jump){
    int vis[MAXN]={0};
    int res=1;
    queue<int> q;
    q.push(0);
    q.push(-1);
    int tmp_max=0;

    while(1){
        int e = q.front();
        q.pop();
        // cout << "e" << e << endl;
        // cout << "res " << res << endl;
        // cout << "tmp_max " << tmp_max << endl;
        if(e==-1){
            res++;
            q.push(-1);
        }
        else{
            for(int j=tmp_max+1; j<e; j++){
                if(!vis[j]){
                    q.push(j);
                    vis[j]=1;
                }
            }
            int tmp=e +jump[e];
            if(tmp >= jump.size()) break; // >= out of dip
            else{
                if(!vis[tmp]){
                    q.push( tmp);
                    vis[tmp] =1;
                }
            }
            tmp_max = max(tmp_max, e);
        }
    }
    return res;
}
```




LCP 24 数字游戏

q0 表示小的那一半的元素的优先队列， 大顶堆。
q1 表示大的那一半的元素的优先队列。

默认降序，大顶堆。下面两句等价。
priority_queue<int, vector<int>, less<int>> q0; // dsc
priority_queue q0;


维护中位数，time nlogn, space n
两个优先队列来实时维护中位数

```cpp
typedef long long ll;
static constexpr int mod=1e9+7;
vector<int> numsGame(vector<int>& arr){
    int n=arr.size();
    if(n==1) return {0};
    for(int i=0; i<n; i++){
        arr[i] -= i; //
    }

    priority_queue<int, vector<int>, less<int>> q0; // dsc
    priority_queue<int, vector<int>, greater<int>> q1; // asc
    q0.push( min(arr[0], arr[1]));
    q1.push( max(arr[0], arr[1]));
    ll sum0 = q0.top();
    ll sum1 = q1.top();

    vector<int> res;
    res.push_back(0);
    res.push_back(static_cast<int>(sum1 -sum0));
    
    for(int i=2; i<n; i++){
        if(arr[i] <= q0.top()){
            q0.push( arr[i]);
            sum0 += arr[i];
        }
        else{
            q1.push( arr[i]);
            sum1 += arr[i];
        }

        if(q0.size()== q1.size()+2){
            int u=q0.top();
            q0.pop();
            sum0 -= u;

            q1.push( u);
            sum1 += u;
        }
        else if(q0.size()+1 ==q1.size()){
            int u=q1.top();
            q1.pop();
            sum1 -= u;

            q0.push(u);
            sum0 += u;
        }

        ll delta = (i&1)? sum1-sum0: sum1-sum0+q0.top();
        res.push_back(delta %mod);
    }
    return res;
}
```

LCP 25 古董键盘

dp
```cpp
typedef long long ll;
static const int mod=1e9+7;

int keyboard(int k, int n){
    vector<vector<ll>> dp(n+1, vector<ll>(27, 0L));
    for(int i=0; i<=26; i++) dp[0][i]=1;

    for(int i=1; i<=n; i++){
        for(int j=1; j<=26; j++){
            for(int t=0; t<=k; t++){
                if(i-t >=0)  dp[i][j] += dp[i-t][j-1]*combine(i, t);
            }
            dp[i][j] %=mod;
        }
    }
    return dp[n][26];
}

ll combine(int m, int n){
    int k=1;
    ll res=1;
    while(k<=n){
        res =((m-k+1)*res)/k;
        k++;
    }
    return res;
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


LCP 04 覆盖

状压dp

二分图最大匹配, 匈牙利算法
将各个相邻的点进行二分，使用二分图最大匹配算法即可。
```cpp
// 4 ms
static const int N=64;
vector<int> e[N];
int used[N];
int matched[N];
const int dir[5]= {-1,0,1,0,-1};
bool dfs(int u){
    for(auto& v: e[u]){
        if(used[v]) continue; // 

        used[v] =true;
        if( matched[v]==-1 || dfs(matched[v])){
            matched[v] = u;
            return true;
        }
    }
    return false;
}

int domino(int n, int m, vector<vector<int>>& broken){
    // int br[n][m] ={};
    vector<vector<int>> br(n, vector<int>(m, 0));
    for(auto& b: broken) br[b[0]][b[1]]=1;
    
    vector<int> bg;
    for(int i=0; i<n; i++){
        for(int j=i%2; j<m; j+=2){ //
            if( br[i][j]==1) continue;
            
            bg.push_back(i*m +j);
            for(int k=0; k<4; k++){
                int ki = i+dir[k];
                int kj = j+dir[k+1];
                if(ki==-1 || kj==-1 || ki==n || kj==m || br[ki][kj]==1) continue;
                e[i*m + j].push_back( ki*m + kj);
            }
        }
    }

    memset(matched, -1, sizeof(matched)); //
    int res=0;
    for(auto& i: bg){
        memset(used, 0, sizeof(used));
        if( dfs(i)) res++;
    }
    return res;
}
```

time n*(4^m)
space n*(2^m)
dp数组 n*2^m 项，填充每项的time 为 2^m

// dp[i][status],i 代表到第 i 行,status 代表当前行的覆盖情况
// bricks(x) 最多横着放的砖块计数
必须把 x==0 的判断用if 逻辑，而非放在for中。后者会导致少一次。
input
3
2
[[1, 1], [2, 1]]

expect -> 2

```cpp
// 0 ms
int ones(int x){
    int res=0;
    for(; x!=0; x= (x&(x-1)))  ++res;
    return res;
}

int bricks(int x){
    int res=0;
    while(x){
        int j= x& (-x);
        if( x& (j<<1)) ++res;
        x &= (~j);
        x &= ~(j<<1);
    }
    return res;
}

int domino(int n, int m, vector<vector<int>>& broken){
    int m1 = 1<<m;
    int max_v=0;
    vector<int> br(n+1, 0);
    vector<vector<int>> dp(n+1, vector<int>(1<<m, 0));
    dp[n][m1-1]=0;
    br[n]=m1-1;  // last line.
    
    for(auto v: broken) br[v[0]] |=(1<<v[1]);

    for(int l=n-1; l>=0; l--){
        for(int st=(~br[l])&(m1-1); ; st=(st-1)&(~br[l]) ){
            int max_cnt=0;
            int s = st & (~br[l+1]);
            for(int k=s; ; k=(k-1) &s){
                max_cnt = max(ones(k)+bricks(st& (~k)) + dp[l+1][br[l+1]|k], max_cnt);
                if(k==0) break; //
            }
            dp[l][(~st)&(m1-1)] =max_cnt;
            if(st==0) break;  // must
        }
    }
    for(int i=0; i<m1; i++) max_v=max(max_v, dp[0][i]);
    return max_v;
}
```


LCP 21 追逐游戏
bfs + dfs + 环

分情况讨论
ab右边，直接结束。
有环，且环长度 >=4, 无法抓住b
一定能抓住。如果一个点到A的最短距离大于到B的最短距离加一, 这个点就是B可以安全到达的点。

```cpp
// 488 ms defeat 100%
#define INF 0x3f3f3f3f

class Solution{
public:
vector<vector<int>> adj;
vector<int> depth;
vector<int> pa;
vector<bool> in_loop;
int m;
int loop=0;

void dfs(int u, int p){
    pa[u] =p; //
    depth[u] = depth[p] +1;
    
    for(int v: adj[u]){
        if(v==p) continue;
        
        if(!depth[v]) dfs(v, u);
        else if(depth[v] < depth[u]){ // circle exists if find auti-edge
            int cu = u;  
            while( cu!=v){
                in_loop[cu]=true;
                loop++;
                cu = pa[cu];
            }
            in_loop[v] = true;
            loop++; // 
        }
    }
}

// find shortest dist of all points to u if detect_loop=true
// find circle start point otherwise.
vector<int> bfs(int u, bool detect_loop){
    vector<int> dist(m+1, INF);
    queue<int> q;
    dist[u] =0;
    q.push(u);
    while( !q.empty()){
        int x =q.front();
        q.pop();
        
        if(detect_loop && in_loop[x]) return {x, dist[x]};
        for(int y: adj[x]){
            if(dist[y] <= dist[x]+1) continue;
            dist[y] = dist[x] +1;
            q.push(y);
        }
    }
    return dist;
}

int chaseGame(vector<vector<int>>& edges, int bg_a, int bg_b){
    m = edges.size();
    adj = vector<vector<int>>(m+1);
    for(auto& e: edges){
        adj[e[0]].emplace_back(e[1]);
        adj[e[1]].emplace_back(e[0]);
        // catch if bg-ed edge
        if(e[0]==bg_a && e[1]==bg_b) return 1;
        if(e[1]==bg_a && e[0]==bg_b) return 1;
    }
    // dfs find circle
    depth =vector<int>(m+1);
    pa =vector<int>(m+1);
    in_loop =vector<bool>(m+1);
    dfs(1, 0);

    // bfs to get shortest dist of a-b
    vector<int> da =bfs(bg_a, false);
    vector<int> db =bfs(bg_b, false);
    // can't catch b if circle len >=4
    if(loop>=4){
        vector<int> qb =bfs(bg_b, true);  // find bg of circle
        if(qb[1]+1 < da[ qb[0]])  return -1;
    }

    // can catch b.
    int res=0;
    for(int i=1; i<=m; i++){
        if(da[i] > db[i]+1) res=max(res, da[i]);
    }
    return res;
}
};
```

```cpp
vector<int> numGame(vector<int>& nums){

}
```

```cpp

```

```cpp

```

