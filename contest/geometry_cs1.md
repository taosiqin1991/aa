

LCP 27 黑盒光线反射


重点预处理 O(m+n) 

单次 open()/close() 操作 O(log(m+n))。

```cpp
class BlackBox {
private:
    vector<pair<int, int>> g_pos;
    vector<pair<int, int>> g_neg;
    vector<map<int, int>> g_stat;

public:
    // 
    BlackBox(int m, int n){
        int pt_cnt=(n + m)*2;
        g_pos.assign(pt_cnt, {-1,-1}); //
        g_neg.assign(pt_cnt, {-1,-1});
        for(int i=0; i<pt_cnt; i++){
            if(i!=0 && i!=m+n && g_pos[i].first==-1){
                create_group(m, n, i, j);
            }
            if(i!=m && i!=m*2+n && g_neg[i].first==-1){
                create_group(m, n, i, -1);
            }
        }
    }

    void create_group(int m, int n, int idx, int dir){
        int g_id = g_stat.size();
        int g_loc=0;
        g_stat.emplace_back();
        
    }
    
    int open(int idx, int dir) {
        // insert
        auto [g_id, g_loc]=g_pos[idx];
        if(g_id !=-1) g_stat[g_id].emplace( g_loc, idx);
        
        
        auto [g_id, g_loc]
    }
    
    void close(int index) {

    }
};
```



