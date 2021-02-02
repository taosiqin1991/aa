
代码的deque也可以替换为list.
deque的底层是用分段的连续空间存储元素, list底层是用非连续空间存储元素.

在本个问题中, 使用deque耗时12ms, 而list耗时80ms, 原因是deque比list更适合前后增删数据. 那什么情况下list比deque好呢? 对中间数据插入删除时list的性能更好.

vector, list, deque 比较
http://www.cppblog.com/sailing/articles/161659.html

把二叉树看作一个拓扑图，一棵树的“根结点的数值”总是先于它的“左右子树中的结点的数值”被插入树中。

面试题 04.09 二叉搜索树序列
expect
[
   [2,1,3],
   [2,3,1]
]

```cpp
// wkcn 16 ms
// struct TreeNode {
//     int val;
//     TreeNode left;
//     TreeNode right;
//     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
// };
vector<vector<int>> BSTSequences(TreeNode* root){
    if(!root) return {{}};
    deque<TreeNode*> q;
    q.push_back( root);

    vector<int> buf;
    vector<vector<int>> res;
    dfs(q, buf, res);
    return res;
}

void helper(deque<TreeNode*>& q, vector<int>& buf, vector<vector<int>>& res){
    if(q.empty()){
        res.push_back( buf);
        return ;
    }
    
    int n=q.size();
    while( n--){ // not equal to q.empty()
        auto e = q.front();
        q.pop_front();

        int ch=0;
        buf.push_back(e->val);
        if(e->left){++ch; q.push_back(e->left);}
        if(e->right){++ch; q.push_back(e->right);}
        
        helper(q, buf, res);

        while( ch--){ q.pop_back(); }
        q.push_back(e);
        buf.pop_back();
    }
}
```

深搜，每一步知道有哪些选择，并且做出选择，把子节点加入，下次选择前要删除

```cpp
// 20 ms
vector<vector<int>> BSTSequences(TreeNode* root) {
    if( !root) return {{}};
    vector<TreeNode*> vec={root};
    vector<vector<int>> res;
    vector<int> tmp;
    helper(vec, tmp, res);
    return res; 
}

void helper(vector<TreeNode*>& vec, vector<int>& tmp, vector<vector<int>>& res){
    if(vec.size()==0){
        res.push_back(tmp);
        return ;
    }

    int n = vec.size();
    for(int i=0; i<n; i++){
        auto e= vec[i];
        vec.erase(vec.begin()+i); // become less
        
        if(e->left) vec.push_back(e->left);
        if(e->right) vec.push_back( e->right);
        tmp.push_back( e->val);
        
        helper(vec, tmp, res);

        tmp.pop_back();
        if(e->right) vec.pop_back();
        if(e->left)  vec.pop_back();
        vec.insert(vec.begin()+i, e);
    }
    return ;
}
```


```cpp

```

679 24 点游戏

回溯time 
C4_2  * C3_2 * C2_2 * 4 * 4 * 4 = 9000+

用双端队列记录正在处理的数字。
每次从队列中取出最前面的一个数，记为a, 剩下的数依次作为b，进行四则运算，得到的结果v放到双端队列的末尾，调用函数go进行下一轮的计算，计算结束后，将队列末尾元素（也就是v）弹出队列

input 8,3,3
8 / (3 - 8 / 3) = 24
8 / (3 - (8 / 3)) = 24


eps不能设得太小。当eps=10e-7时，对于3, 3, 8, 8这个样例，由于误差大于10e-7，是判断不出和24近似的.

```cpp
// 4 ms
const float eps=1e-5;
inline bool gen(deque<float>& q, float v){
    q.push_back(v);
    bool b =go(q);
    q.pop_back();  // recover
    return b;
}

bool judgePoint24(vector<int>& nums){
    deque<float> q;
    for(auto i: nums) q.push_back(i);
    return go(q);
}

bool go(deque<float>& q){
    const int n=q.size();
    if(n==1) return abs(q.front()-24) <eps; //
    
    for(int i=0; i<n; i++){
        float a=q.front();
        q.pop_front();

        for(int j=1; j<n; j++){
            float b=q.front();
            q.pop_front();
            if( gen(q, a+b) || gen(q, a-b) || gen(q, a*b) || (b && gen(q, a/b))){
                q.push_back(b);
                q.push_back(a);
                return true;
            }
            q.push_back(b);
        }
        q.push_back(a);
    }
    return false;
}

```

dfs 回溯
```cpp
// 16 ms
const float eps=1e-5;
bool judgePoint24(vector<int>& nums){
    bool res=false;
    vector<float> arr(nums.begin(), nums.end());
    helper(arr, res);
    return res;
}

void helper(vector<float>& arr, bool& res){
    if(res){ return; }

    if(arr.size()==1){
        if( abs(arr[0]-24) < eps) res=true;
        return ;
    }

    for(int i=0; i<arr.size(); i++){
        for(int j=0; j<i; j++){
            float p=arr[i];
            float q=arr[j];
            vector<float> t{p+q, p-q, q-p, p*q};
            if(p>eps) t.push_back( q/p);
            if(q>eps) t.push_back(p/q); // divide non_zero
            
            arr.erase(arr.begin()+i);
            arr.erase(arr.begin()+j);
            for(float e: t){
                arr.push_back( e);
                
                helper(arr, res);
                arr.pop_back();
            }
            arr.insert(arr.begin()+j, q);
            arr.insert(arr.begin()+i, p);
        }
    }
}
```


```cpp
// da-li-wang 8 ms
# define ADD +
# define SUB -
# define MUL *
# define DIV /
# define JUDGE(nums, target, a, b, op) \
    do { \
        nums.push_back(a op b); \
        if (judge(nums, target)) return true; \
        nums.pop_back(); \
    } while (0)

class Solution {
public:
    bool judge(const vector<double>& nums, double target) {
        int s = nums.size();
        if (s == 1)
            return (nums[0] < target + 1e-8) && (nums[0] > target - 1e-8);
        for (int i = 0; i < s; ++i) {
            for (int j = i + 1; j < s; ++j) {
                vector<double> new_nums;
                for (int k = 0; k < s; ++k) {
                    if (k != i && k != j)
                        new_nums.push_back(nums[k]);
                }
                double a = nums[i];
                double b = nums[j];
                JUDGE(new_nums, target, a, b, ADD);
                JUDGE(new_nums, target, a, b, MUL);
                JUDGE(new_nums, target, a, b, SUB);
                JUDGE(new_nums, target, b, a, SUB);
                if (b != 0) JUDGE(new_nums, target, a, b, DIV);
                if (a != 0) JUDGE(new_nums, target, b, a, DIV);
            }
        }
        return false;
    }
    bool judgePoint24(const vector<int>& nums) {
        vector<double> new_nums(nums.begin(), nums.end());
        return judge(new_nums, 24);
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

