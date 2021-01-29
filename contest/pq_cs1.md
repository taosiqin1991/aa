

778 水位上升的泳池中游泳

优先队列 + 并查集/bfs/dfs， time nnlogn, space nn.
二分查找 + bfs  ， time nnlogn, space nn.




直观理解： 路径上的最大值。所有路径中。求这个最优路径。

优先队列。
用优先队列保存下一步可以游向的平台，每次都选择高度最小的平台。以这种方式到达终点时，路径中遇到的最高平台就是答案。

优先队列 + 并查集
1、先把所有点坐标及相应的高度值入队列，h升序
2、遍历每个时间点，并将在高度和时间点相等时出队列，表示当次新增的点，将新增的点与周边能走通的点关联起来
3、涉及元素的关联，可以使用并查集处理，节点指定共父亲
4、每个时间点，所有关联关系建立之后，检查起始点和终点是不是共父亲，如果是的话，说明当前队列中出现过的点，就能够走通起点和终点


time nnlogn, space nn.
最大需要经过 nn 个节点，每个节点需要 logn 的时间来完成堆操作。

https://leetcode-cn.com/problems/swim-in-rising-water/solution/you-xian-dui-lie-bing-cha-ji-shen-sou-yan-sou-by-h/

```cpp
// 16 ms best
struct node{
    int x;
    int y;
    int h;
    node(int i, int j, int v): x(i),y(j),h(v){}
    bool operator <(const node& rhs) const{
        return h > rhs.h;
    }
};

vector<int> f;
const int d[4][2]={{0, -1}, {0,1}, {-1,0}, {1,0}};

void init(int n){
    f.resize(n);
    for(int i=0; i<n; i++) f[i]=i;
}

int find(int x){
    return f[x]==x? f[x]: f[x]=find(f[x]);
}

int swimInWater(vector<vector<int>>& g) {
    int n= g.size();
    int cnt= n*n;
    priority_queue<node> q;
    init(cnt);
    
    for(int i=0; i<cnt; i++){
        q.push( node(i/n, i%n, g[i/n][i%n]));   
    }

    for(int t=0; t<cnt; t++){
        while( !q.empty() && q.top().h==t ){  // 
            auto e = q.top();
            q.pop();

            int z = e.x *n + e.y;
            for(int i=0; i<4; i++){
                int nx = e.x + d[i][0];
                int ny = e.y + d[i][1];
                if(nx>=0 && nx<n && ny>=0 && ny<n && g[nx][ny]<=t){ //
                    int nz = nx * n +ny;
                    f[ find(nz)] = find(z); // 
                }
            }
        }
        if( find(0)== find(cnt-1)) return t;
    }
    return n*n-1;
}
```

```cpp
// dfs
int d[4][2] = {{-1, 0},{1, 0},{0, -1},{0, 1}};
bool dfs(grid, x, y, t, n, visited){
    if (x == n-1 && y == n-1) {
        return true;
    }
    for (int i = 0; i < 4; i++) {
        newx = x + d[i][0];
        newy = y + d[i][1];
        if (newx newy valid && visited[newx][newy] == 0 && grid[newx][newy] <= t) {
            visited[newx][newy] = 1;
            if( dfs(grid, newx, newy, t, n, visited)) {
                return true;
            }
        }
    }
    return false;
}

int swimInWater(vector<vector<int>>& grid) {
    int n = grid.size();
    // t 最小值为末尾元素高度
    for (t : N*N) {
        vector<vector<int>> visited(n, vector<int>(n, 0));
        if ( dfs(grid, 0, 0, t, n, visited)) {
            return t;
        }
    }
    return N*N-1;
}
```

```cpp
// bfs
int swimInWater(vector<vector<int>>& grid) {
    int n = grid.size();
    queue<pair<int, int>> cur;
    queue<pair<int, int>> next;
    next.push({0, 0});
    visited[0][0] = 0;

    for (t : N*N) {
        cur = next;
        while(!cur.empty()) {
            pair<int, int> loc = cur.top();
            cur.pop();
            if (loc == {N-1, N-1}) {
                flag;
                break;
            }
            for (int i = 0; i < 4; i++) {
                int newx = x + d[i][0];
                int newy = y + d[i][1];
                if (newx newy valid && grid[newx][newy] < t && visited[newx][newy] == 0) {
                    visited[newx][newy] = 1;
                    cur.push();
                    next.push();
                }
            }
        }
        if (flag) {
            return t;
        }
    }
    return N*N-1;
}
```


```cpp

```

```cpp

```



1102 得分最高的路径

类Prim的算法是基于BFS+优先级队列（基于最大堆）
类kersual算法是基于由大到小的顺序不停的进行并查集的合并操作，指导满足满足A[0][0]和A[maxI][maxJ]相连接即可
最小生成树是基于权重的最小值进行处理，而该题实际是基于权重的最大值进行处理。


优先队列 + bfs, 796 ms
并查集 + 降序，824 ms


```cpp
struct Dot{
    int val;
    int y;  // row
    int x;  // col
    Dot(int v, int y_, int x_): val(v), y(y_), x(x_){}
};

bool operator<(const Dot& a, const Dot& b){
    return a.val < b.val;
}

class Solution {
public:
priority_queue<Dot> pq;
vector<vector<int>> res;
vector<vector<bool>> vis;
int dx[4] = {0, 1, 0, -1};
int dy[4] = {1, 0, -1, 0};


int maximumMinimumPath(vector<vector<int>>& A) {
    int res1 = -1;
    
    int m = A.size();
    int n = A[0].size();
    vis.resize(m);
    for(int i=0; i<m; i++){
        vis[i].assign(n, false);
    }
    
    Dot t(A[0][0], 0, 0);
    pq.push(t);
    vis[0][0] = true;
    res1 = 0x7fffffff;
    
    while( !pq.empty()){
        auto cur_t = pq.top();
        pq.pop();
        res1 = min(res1, cur_t.val); // 
        
        if(cur_t.y ==m-1 && cur_t.x==n-1) return res1;

        for(int i=0; i<4; i++){
            int ny = cur_t.y + dy[i];
            int nx = cur_t.x + dx[i];
            if(ny>=0 && nx>=0 && ny<m && nx<n && !vis[ny][nx]){
                Dot c(A[ny][nx], ny, nx);
                pq.push( c);
                vis[ny][nx] = true;
            }
        }

    }
    return res1;
}

};
```

// ok
```cpp
class UnionFind{
private:
    vector<int> fa;

public:
    UnionFind(int n){
        fa.resize(n);
        iota(fa.begin(), fa.end(), 0);
    }

    ~UnionFind(){
        fa.clear();
    }

    int find(int idx){
        if( fa[idx]== idx) return idx;

        while( fa[idx]!= idx){
            idx = fa[idx];
        }
        return idx;
    }

    void connect(int a, int b){
        int af = find(a);
        int bf = find(b);
        if(af==bf) return ;

        if(af> bf){
            swap(af, bf);
        }
        fa[bf] = af; // af is small.
    }    
};

struct Vertex{
    int i;  // x, col
    int j;  // y, row
    int val;
    Vertex(int i_, int j_, int v_):i(i_), j(j_), val(v_){}
    bool operator<(const Vertex& rhs) const{
        return val < rhs.val;
    }
};

class Solution{
public:
const int dx[4] = {0, 1, 0, -1};
const int dy[4] = {1, 0, -1, 0};

int maximumMinimumPath(vector<vector<int>>& A){
    int m = A.size();
    int n = A[0].size();
    UnionFind uf(m * n);
    
    int max_v = min(A[0][0], A[m-1][n-1]);
    priority_queue<Vertex> pq;
    vector<vector<int>> color_set(m, vector<int>(n, 0));

    for(int j=0; j<m; j++){
        for(int i=0; i<n; i++){
            pq.push( Vertex(i, j, A[j][i]));
        }
    }
    while( uf.find(0)!= uf.find(m*n-1)){
        auto e = pq.top();
        pq.pop();
        
        max_v = min(e.val, max_v); //
        color_set[e.j][e.i] = 1;
        
        for(int k=0; k<4; k++){
            int ny = e.j + dy[k];
            int nx = e.i + dx[k];
            if(nx<0 || ny<0 || nx>=n || ny>=m || !color_set[ny][nx])  continue;

            uf.connect(e.j*n+e.i, ny*n + nx);
        }
    }
    return max_v;
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
