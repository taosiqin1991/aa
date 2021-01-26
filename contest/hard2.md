
478 在圆内随机生成点

1) 拒绝采样, time O(1), 最坏情况被一直拒绝O(inf),space O(1)
2) 计算分布函数

```cpp
class Solution{
private:
    double rad;
    double xc;
    double yc;
    //c++11 random floating point number generation
    mt19937 rng{random_device{}()};
    uniform_real_distribution<double> uni{0,1};

public:
    Solution(double radius, double x_c, double y_c){
        rad = radius;
        xc = x_c;
        yc = y_c;
    }

    vector<double> randPoint(){
        double x0 = xc - rad;
        double y0 = yc - rad;

        while(true){
            double xg = x0 + uni(rng)*2*rad;
            double yg = y0 + uni(rng)*2*rad;
            
            if(sqrt(pow(xg-xc, 2) + pow(yg-yc, 2)) <=rad){
                cout << "inner " << xg << "," << yg<< endl;
                return {xg, yg};
            }
            cout << "outer " << xg << "," << yg<< endl;
        }
    }
};
```


```cpp
class Solution{
private:
    double rad;
    double xc;
    double yc;
    mt19937 rng{random_device{}()};
    uniform_real_distribution<double> uni{0,1};

public:
    Solution(double radius, double x_c, double y_c){
        rad = radius;
        xc = x_c;
        yc = y_c;
    }

    vector<double> randPoint(){
        double d = rad * sqrt(uni(rng));
        double theta  = uni(rng) * (2*M_PI);
        return {d*cos(theta) + xc, d*sin(theta)+yc};
    }

};
```


470 用rand7()生成rand10()
time O(1), bad inf, space O(1)

rand7() return 1-7

```cpp
// 1-49, drop 41-49
int rand10(){
    int row = rand7();
    int col = rand7();
    int idx = col + (row-1)*7;
    while(idx>40){
        row = rand7();
        col = rand7();
        idx = col + (row-1)*7;
    }

    return 1+(idx-1)%10;
}

int rand10(){
    while(true){
        int num = (rand7()-1)*7 + rand7();
        
        if(num<=40) return 1+num%10;
        
        //41-49
        num = (num-40-1)*7+rand7();
        if(num<=60) return 1+num%10;
        // 61-63
        num = (num-60-1)*7+rand7();
        if(num<=20) return 1+num%10;
    }
}

```

710 黑名单中的随机数
1) 二分查找，预处理时间 BlogB，随机选择time logB，空间B 或者 1，根据是否原地排序。
2) 黑名单映射，[0, n-len(B)] 是要选择的区间， preprocess time O(B), rand_gen O(1), space O(B)


我们将黑名单分成两部分，第一部分 X 的数都小于 N - len(B)，需要进行映射；第二部分 Y 的数都大于等于 N - len(B)，这些数不需要进行映射，因为并不会随机到它们。



```cpp
// white_list overtime
// return w[rand()% w.size()];
class Solution{
private:
    unordered_map<int, int> m;
    int w_len;

public:
    Solution(int n, vector<int> b){
        w_len = n - b.size();
        unordered_set<int> w;
        for(int i=w_len; i<n; i++) w.insert(i);
        
        for(int x: b) w.erase(x);
        auto wi = w.begin();
        
        for(int x: b){
            if(x<w_len){
                m[x] = *wi++;
            }
            
        }
    }

    int pick(){
        int k = rand() % w_len;
        return m.count(k)? m[k]: k;
    }
};

```

二分查找
```cpp
class Solution{
private:
    int n;
    vector<int> b;
    
public:
    Solution(int N, vector<int> b_vec){
        n = N;
        sort(b_vec.begin(), b_vec.end());
        b = b_vec;
    }

    int pick(){
        int k = rand() % (n-b.size());
        int lo = 0;
        int hi = b.size()-1;

        while(lo<hi){
            int i = (lo+hi+1)/2;

            if(b[i]-i>k){
                hi = i-1;
            }
            else{
                lo = i;
            }
        }
        return lo==hi && (b[lo]-lo <=k)? k+lo+1: k;
    }
};

```


315 计算右侧小于当前元素的个数

线段树解法  有bug


```cpp
struct SegmentNode{
    int start;
    int end;
    int cnt;
    
    SegmentNode* left;
    SegmentNode* right;
    SegmentNode(int s, int e):start(s), end(e),left(nullptr), right(nullptr){}
};

class Solution{
private:

public:
    SegmentNode* build(size_t start, size_t end){
        if(start>end) return nullptr;

        SegmentNode* root = new SegmentNode(start, end);

        if(start<end){
            int mid = start + (end-start)/2;
            root->left = build(start, mid);
            root->right = build(mid+1, end);
        }
        return root;
    }

    void insert(SegmentNode* root, int idx){
        if(root==nullptr) return ;

        if(idx>=root->start && idx<= root->end){
            root->cnt++;
        }
        else{
            return ;
        }

        insert(root->left, idx);
        insert(root->right, idx);
    }

    int count(SegmentNode* root, int vmin, int vmax){
        if(root==nullptr || vmin> vmax) return 0;
        
        if( vmin==root->start && vmax==root->end) return root->cnt;
        
        int mid = root->start + (root->end - root->start)/2;
        int left_cnt = 0;
        int right_cnt = 0;

        if(vmax<= mid){
            left_cnt = count(root->left, root->start, vmax);
        }
        else{
            if(vmin <=mid){
                left_cnt = count(root->left, vmin, mid);
                right_cnt = count(root->right, mid+1, vmax);
            }
            else{
                right_cnt = count(root->right, vmin, vmax);
            }
        }

        return left_cnt + right_cnt;
    }

    vector<int> countSmaller(vector<int>& nums){
        vector<int> res;
        if(nums.empty()) return res;

        int n = nums.size();
        res.resize(n);
        int vmin = INT_MIN;
        int vmax = INT_MAX;
        for(int i=1; i<nums.size(); i++){
            vmin = min(vmin, nums[i]);
            vmax = max(vmax, nums[i]);
        }

        SegmentNode* root = build(vmin, vmax);

        for(int i=n-1; i>=0; i--){
            res[i]=count(root, vmin, nums[i]-1);
            insert(root, nums[i]);
        }
        return res;
    }
};



```

树状数组
「树状数组」是一种可以动态维护序列前缀和的数据结构，它的功能是：

单点更新 update(i, v)： 把序列 i 位置的数加上一个值v，在该题中 v = 1
区间查询 query(i)： 查询序列 [1⋯i] 区间的区间和，即 ii 位置的前缀和

```cpp
class Solution{
private:
    vector<int> c;
    vector<int> a;
    
    void init(int len){
        c.resize(len,0);
    }

    int low_bit(int x){
        return x &(-x);
    }

    void update(int pos){
        while(pos < c.size()){
            c[pos]+=1;
            pos += low_bit(pos);
        }
    }

    int query(int pos){
        int res = 0;
        
        while(pos>0){
            res += c[pos];
            pos -= low_bit(pos);
        }
        return res;
    }

    void discret(vector<int>& nums){
        a.assign(nums.begin(), nums.end());
        sort(a.begin(), a.end());
        
        a.erase( unique(a.begin(), a.end()), a.end());
    }

    int get_id(int x){
        return lower_bound(a.begin(), a.end(), x) - a.begin()+1;
    }

public:
    vector<int> countSmaller(vector<int>& nums){
        int n = nums.size();
        vector<int> res;

        discret(nums);
        init(n+5);
        
        for(int i=n-1; i>=0; i--){
            int id = get_id(nums[i]);

            res.push_back( query(id-1));
            update(id);
        }
        reverse(res.begin(), res.end());
        return res;
    }

};

```


归并排序

overtime

```cpp
class Solution{
public:
    vector<int> countSmaller(vector<int>& nums){
        int n = nums.size();
        vector<int> cnt(n, 0);
        vector<pair<int, int>> arr;
        for(int i=0; i<n; i++){
            arr.push_back( {nums[i], i});
        }
        merge_sort(arr, cnt);
        return cnt;
    }

    void merge_sort(vector<pair<int, int>>& vec, vector<int>& cnt){
        if(vec.size()<2) return ;
        
        int mid = vec.size()/2;
        vector<pair<int, int>> sub_vec1;
        vector<pair<int, int>> sub_vec2;
        for(int i=0; i<mid; i++){
            sub_vec1.push_back( vec[i]);
        }
        for(int i=mid; i<vec.size(); i++){
            sub_vec2.push_back( vec[i]);
        }
        
        merge_sort(sub_vec1, cnt);
        merge_sort(sub_vec2, cnt);

        vec.clear();
        merge(sub_vec1, sub_vec2, vec, cnt);
    }

    void merge(vector<pair<int, int>>& sub_vec1, vector<pair<int, int>>& sub_vec2, vector<pair<int, int>>& vec, vector<int>& cnt){
        int i=0;
        int j=0;
        int m = sub_vec1.size();
        int n = sub_vec2.size();
        while(i<m && j<n){
            if(sub_vec1[i].first <= sub_vec2[j].first){
                vec.push_back( sub_vec1[i]);
                cnt[ sub_vec1[i].second] +=j;
                i++;
            }
            else{
                vec.push_back(sub_vec2[i]);
                j++;
            }
        }

        while(i<m){
            vec.push_back( sub_vec1[i]);
            cnt[ sub_vec1[i].second] += j;
        }
        while(j<n){
            vec.push_back( sub_vec2[j]);
        }
    }
};

```

715 range模块

平衡树
需要快速定位，删除连续元素，统计前缀和

```cpp
/*
if(key < node->key) return rank(node->left, key);
int res = node->left->sum + node->val;
if(key==node->key) return res;
else return res + rank(node->right, key);
*/

class RangeModule{
private:
    map<int, int> right_left;
    queue<map<int, int>::const_iterator> q;

public:
    RangeModule(){}

    void addRange(int left, int right){
        auto it = right_left.lower_bound(left);
        while( it!=right_left.end() && it->second<=right){
            q.push( it);
            it++;
        }

        while( !q.empty()){
            auto it = q.front();
            q.pop();

            left = min(left, it->second);
            right = max(right, it->first);
            right_left.erase(it);
        }
        right_left.insert( {right, left});
    }

    bool queryRange(int left, int right){
        auto it = right_left.lower_bound(right);
        return it!= right_left.end() && it->second<=left;
    }

    void removeRange(int left, int right){
        auto it = right_left.upper_bound(left);
        while(it!=right_left.end() && it->second<right){
            q.push(it);
            it++;
        }

        while(!q.empty()){
            auto it=q.front();
            q.pop();
            
            int old_right = it->first;
            int old_left = it->second;
            right_left.erase(it);
            
            if(old_left < left){
                right_left.insert( {left, old_left});
            }
            if(old_right > right){
                right_left.insert( {old_right, right});
            }
        }
    }
};

```

1157 子数组占绝大多数的元素

1）分块，s=sqrt(2n), time (n+q)*sqrt(n)
2）线段树，20000个vector，单词query logn，预处理n，故time n+qlogn.
貌似由于常数 + 数据原因，分块比线段树还快。

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

