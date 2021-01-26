

5548 最小体力消耗路径

```cpp

```

```cpp
```

```cpp
```

5156 矩阵转换后的秩
1) 并查集 + 拓扑排序，time mn*log(mn) 瓶颈为排序

将每行及每列中相等的元素找出来，然后连并查集的边，不需要连所有边，只需相邻两个连边即可。后面只考虑并查集中每一个连通分量的根节点。

每行每列分别排序，根据排序后的大小关系连拓扑排序的边。

进行拓扑排序。每个点的秩初始值为1，在拓扑排序中，如果有一条 u-> v的边, 则将v的秩设置为 max(rank(v), rank(u)+1 ), 其余操作同一般的拓扑排序.

最后将矩阵中每个点的秩设置为 并查集中其所在的连通分量的 根节点的秩即可.

```cpp
// 820ms, 76MB
class UnionFind{
private:
    int n;
    vector<int> parent;
    vector<int> size;

public:
    UnionFind(int n){
        this->n = n;
        size = vector<int>(n, 1);
        parent = vector<int>(n);
        for(int i=0; i<n; i++){
            parent[i] = i;
        }

    }

    int find(int idx){
        if( parent[idx] == idx)  return idx;
        parent[idx] = find( parent[idx]);
        return parent[idx]; 
    }

    void connect(int a, int b){
        int fa = find(a);
        int fb = find(b);
        
        if(fa!= fb){
            if(size[fa] > size[fb]){
                parent[fb] = fa;
                size[fa] += size[fb];
            }
            else{
                parent[fa] = fb;
                size[fb] += size[fa];
            }
        }
    }
};


class Solution{
public:

vector<vector<int>> matrixRankTransform(vector<vector<int>>& mat){
    int n = mat.size();
    int m = mat[0].size();

    UnionFind uf(n*m);
    // row
    for(int i=0; i<n; ++i){
        map<int, vector<int>> mp;

        for(int j=0; j<m; j++){
            mp[ mat[i][j]].emplace_back(i*m + j);
        }
        for(auto& [num, vec]: mp){
            for(int k=0; k+1<vec.size(); k++){
                uf.connect( vec[k], vec[k+1]);
            }
        }
    }
    // col
    for(int j=0; j<m; ++j){
        map<int, vector<int>> mp;

        for(int i=0; i<n; i++){
            mp[ mat[i][j]].emplace_back(i*m + j);
        }
        for(auto& [num, vec]: mp){
            for(int k=0; k+1<vec.size(); k++){
                uf.connect( vec[k], vec[k+1]);
            }
        }
    }

    vector<vector<int>> adj(n*m);
    vector<int> indegree(n*m);
    // row
    for(int i=0; i<n; i++){
        vector<pair<int, int>> v(m);

        for(int j=0; j<m; j++){
            v[j] = make_pair( mat[i][j], j);
        }
        sort(v.begin(), v.end());
        for(int j=0; j+1<m; j++){

            if( v[j].first != v[j+1].first){
                int uu = uf.find(i*m + v[j].second);
                int vv = uf.find(i*m + v[j+1].second);
                adj[uu].emplace_back(vv);
                indegree[vv]++;
            }
        }
    }
    // col
    for(int j=0; j<m; j++){
        vector<pair<int, int>> v(n);

        for(int i=0; i<n; i++){
            v[i] = make_pair( mat[i][j], i);
        }
        sort(v.begin(), v.end());
        for(int i=0; i+1<n; i++){

            if( v[i].first != v[i+1].first){
                int uu = uf.find(v[i].second * m + j);
                int vv = uf.find(v[i+1].second *m + j);
                adj[uu].emplace_back(vv);
                indegree[vv]++;
            }
        }
    }

    vector<int> ans(n*m, 1);
    queue<int> q;
    for(int i=0; i<n*m; i++){
        if(uf.find(i)==i && indegree[i]==0){
            q.emplace(i);
        }
    }
    while( !q.empty()){
        int u = q.front();
        q.pop();
        
        for(int v: adj[u]){
            ans[v] = max(ans[v], ans[u]+1);
            indegree[v]--;
            
            if(indegree[v]==0)  q.emplace(v);
        }
    }
    
    vector<vector<int>> res(n, vector<int>(m));
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            res[i][j] = ans[uf.find(i*m + j)];
        }
    }
    return res;

}

};

```



5546 按键持续时间最长的键
1) 一次遍历, time n

```cpp
// 16ms, 10MB
char slowestKey(vector<int>& release_times, string& key_pressed){
    int pre = 0;
    int max_time = 0;
    int time = 0;
    char ans= 'a';

    for(int i=0; i<release_times.size(); i++){
        time = release_times[i] - pre;
        if(time > max_time){
            max_time = time;
            ans = key_pressed[i];
        }
        else if(time== max_time && key_pressed[i]> ans){
            ans = key_pressed[i];
        }

        pre = release_times[i];
    }
    return ans;
}

```

5547 等差子数组


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
