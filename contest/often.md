
处理动态问题一般都会用到有序数据结构，比如平衡二叉搜索树和二叉堆，二者的时间复杂度差不多，但前者支持的操作更多。

既然平衡二叉搜索树这么好用，还用二叉堆干嘛呢？因为二叉堆底层就是数组，实现简单啊，详见旧文「二叉堆详解」。你实现个红黑树试试？操作复杂，而且消耗的空间相对来说会多一些。具体问题，还是要选择恰当的数据结构来解决。



快排思想：
选择一个数base（常第一个），low和high分别向中间扫描，小的数都在base左边，大的数都在base右边。小的数群之间的排序不做调整。这样一趟走完。
再对左右区间分区，重复上述过程。

```cpp
void quick_sort(int* array, int left, int right){
    if( left < right){
        int pivot = array[left];
        
        // all element scan.
        int lo = left;
        int hi = right; // array[lo] is saved.
        while(lo < hi){
            while( array[hi] >= pivot && lo < hi){
                hi--;
            }
            array[lo] = array[hi];  

            while(array[lo]<=pivot && lo<hi){
                lo++;
            }
            array[hi] = array[lo];
        }
        array[lo] = pivot;

        quick_sort(array, left, lo-1);
        quick_sort(array, lo+1, right);
    }
}

```


堆排序：堆在实现优先队列时是有优势的。因为插入或者删除某个元素，都有额外的其他操作。sink和操作。
（假定我们常用小顶堆）
堆是一颗完全二叉树。
堆中的某个节点总是不大于其子节点的值（最小堆）。不小于(最大堆)
插入元素：先放堆尾，再上浮swin。时间lgn
删除堆顶元素：把其与堆尾交换，删除其，然后对堆顶部下沉sink。时间lgn。
建堆的过程是依次在尾部添加元素，上浮的过程。




```cpp
//maxheap

bool less(int i, int j){
    return a[i] < a[j];
}

void exchange(int i, int j){
    int tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
}
// get indexs.
void parent(k){
    return k%2? (k/2) : (k/2-1);
}

void left(int k){
    return 2*k+1;
}

void right(int k){
    return 2*k + 2;
}

void swim(int k){
    // don't swim if it get to top.
    while(k>1 && less(parent(k),k)) ){
        // exchange if k_th element is bigger than parent.
        exchange(parent(k), k);

        k = parent(k);
    }
}

void sink(int k){
    // stop if sink to bottom.
    while( left(k)<=N ){
        // suppose left is bigger.
        int older = left(k);

        // if right(k) exist.
        if(right(k) <=N && less(older，right(k)) ){
            older = right(k);
        }

        if(older<k ) break; // is k is bigger, don't sink.
        exchane( k, older);

        k = older;
    }
}

```


归并排序：思想简单，写好不易。

```cpp
```



二分搜索：基本的。
```cpp

```

反转链表
```cpp
ListNode* reverse(ListNode* head) {
    ListNode* pre = null, cur* = head;
    while (cur != null) {
        ListNode* next = cur->next;
        cur->next = pre;
        
        pre = cur;
        cur = next;
    }
    return pre;
}
```




DFS遍历使用递归（递归隐含使用了系统的栈，我们不需要自己维护一个数据结构）
BFS遍历使用队列。

Dijkstra解决带权最短路径。
BFS解决无权最短路径。面试中能保证自己写对Dijkstra的人不多。

```cpp
void dfs(TreeNode* root){
    if( root==NULL) return ;
    // do_something

    dfs( root->left);
    dfs( root->right);
}

void bfs(TreeNode* root){
    queue<TreeNode*> que;
    
    que.push(root);
    while( !que.empty()){
        TreeNode* node = que.front();
        que.pop();
        // do something
        
        if(node->left!=NULL) que.push( node->left);
        if(node->right!=NULL) que.push( node->right);
    }
    
}

// dfs求解岛屿问题的模板
void dfs(vector<vector<int>>& grid, int r, int c){
    if( !in_area(grid, r, c)) return ;

    // avoid again
    if( grid[r][c]!=1) return;
    grid[r][c] = 2; // mark as handled.

    // up-down-left-right
    dfs(grid, r-1, c);
    dfs(grid, r+1, c);
    dfs(grid, r, c-1);
    dfs(grid, r, c+1);
}

bool in_area(vector<vector<int>>& grid, int r, int c){
    return r>=0 && r<grid.size() && c>=0 && c<grid[0].size();
}

```


BFS: 求解最短路径、层序遍历。
如果是遍历一棵树/一张图，BFS和DFS都可以。
广义来说，DFS都可以写成BFS的形式。
DFS主要优点是写起来更简单，空间复杂度更低。
有些任务是DFS做不到的，需要BFS，如最短路径、层序遍历。

BFS + 剪枝 就是 DP的形式了。



BFS求解二叉树的层序遍历
```cpp
vector<vector<int>> level_order(TreeNode* root){
    vector<vector<int>> res;
    queue<TreeNode*> que;
    if( root==NULL) return res;
    que.push( root);
    
    whiie(!que.empty()){
        vector<int> tmp;

        int n = que.size();
        for(int i=0; i<n; i++){
            TreeNode* node = que.front();
            que.pop(); 
            res.push_back(node->val);

            if( node->left) que.push(node->left);
            if( node->right) que.push(node->right);

        }
        res.push_back(tmp);
    }

}

```



BFS求解最短路径：

```cpp
void bfs(vector<vector<int>>& grid, int i, int j){
    queue<int*> 
}

```

DFS求解岛屿面积
```cpp
int island_max_area(vector<vector<int>>& grid){
    int res = 0;
    int M = grid.size();
    int N = grid[0].size();
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            if( grid[i][j]==1){
                int a = dfs(grid, i, j);
                res = max(res, a);
            }
        }
    }
    return res;
}

int dfs(vector<vector<int>>& grid, int r, int c){
    if( !in_area(grid, r, c)) return 0;


    if( grid[r][c]!=1) return 0;
    grid[r][c] = 2; 
    
    return 1 + dfs(grid, r-1, c) 
            + dfs(grid, r+1, c) 
            + dfs(grid, r, c-1) 
            + dfs(grid, r, c+1);
}

bool in_area(vector<vector<int>>& grid, int r, int c){
    return r>=0 && r<grid.size() && c>=0 && c<grid[0].size();
}
```

二叉树中序遍历来判断一颗树是不是平衡二叉树
```cpp
TreeNode prev; // 全局变量：指向中序遍历的上一个结点
boolean valid;

public boolean isValidBST(TreeNode root) {
    valid = true;
    prev = null;
    traverse(root);
    return valid;
}

void traverse(TreeNode curr) {
    if (curr == null) {
        return;
    }

    traverse(curr.left);

    // 中序遍历的写法，把操作写在两个递归调用中间
    if (prev != null && prev.val >= curr.val) {
        // 如果中序遍历的相邻两个结点大小关系不对，则二叉搜索树不合法
        valid = false;
    }
    // 维护 prev 指针
    prev = curr;

    traverse(curr.right);
}
```


回溯模板
```cpp

bool backtrack(vector<string>& board, int row) {
    // 触发结束条件
    if (row == board.size()) {
        res.push_back(board);
        return true;
    }
    ...
    for (int col = 0; col < n; col++) {
        ...
        board[row][col] = 'Q';

        if (backtrack(board, row + 1))
            return true;
        
        board[row][col] = '.';
    }

    return false;
}


```





全排列用回溯解
```python
def queen_dfs(n, queens, record):
    if n>=8:
        record.append( queeens.copy())
        return
    
    for i in range(8):
        # if same cols
        if i in queens:
            continue
        
        # if same diag
        flag = False
        for j in range(len(queens)):
            if abs(n-j)==abs(i-queens[j]):
                flag = True
                break
        if flag:
            continue
        
        # do_something
        queens.append(i)
        queen_dfs(n+1, queens, record)
        queens.pop()


def permute_dfs(permutation, array, origin):
    """
    origin: record raw
    mini: record answer, it means 
    """
    global mini
    
    if(len(array)==0): # if all into permutation
        # if legal and minumum
        if permutation > origin and permutation < mini:
            mini = permutation.copy()  # need copy because permutation will pop
            
    for i in array.copy():
        array.remove(i)
        permutation.append(i)

        permute_dfs(permutation, array, origin)
        array.append(i)
        permutation.pop()

```


动态规划-LCS最长公共子序列
时间复杂度N2，空间复杂度N.

很多时候二维dp比一维dp 写起来要容易些。

```cpp
int longest_common_seq(string& astr, string& bstr){
    if( !astr.size() || !bstr.size()) return 0;
    
    int M = astr.size();
    int N = bstr.size();
    vector<int> dp(N+1, 0));
    
    for(int i=1; i<=M; i++){
        int tmp =0;

        for(int j=1; j<=N; j++){
            int dp_j;
            if( astr[i-1] == bstr[j-1] ){
                dp_j = tmp + 1;
            }
            else{
                dp_j = max(dp[j], dp[j-1]); // key
            }
            tmp = dp[j];
            dp[j] = dp_j;
        }
    }
    return dp[N];
    
}
```

```cpp
连续子数组的最大和 dp[k] = max(dp[k-1] + a[k], a[k]); res= max(res, dp[k]);
最长波形子数组，建立两个,f[], g[], 其他同上。

最长公共子序列  dp[i][j] = max(dp[i-1][j-1]+1, dp[i-1][j], dp[i][j-1]);

最长公共子数组  dp[i][j] = max(dp[i-1][j-1]+1, 0);  res = max(res, dp[i][j]);

编辑距离
dp[i][j] = dp[i-1][j-1]
dp[i][j] = min( dp[i-1][j] +1, dp[i][j-1] +1, dp[i-1][j-1]+1)
编辑距离空间优化


一定有间隔的子数组最大和
dp[i] = max(dp[i-2] + a[i], dp[i-1])
```



reverse翻转
```cpp
void rotate(vector<int>& nums, int k){
    int N = nums.size();
    k = k % N;
    reverse(nums, 0, N);
    reverse(nums, 0, k);
    reverse(nums, k, N);
    
}

void reverse(vector<int>& nums, int begin, int end){
    for(int i=begin, j=end-1; i<j; i++,j--){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}

```



概率题
1）当你遇到第 i 个元素时，应该有 1/i 的概率选择该元素，1 - 1/i 的概率保持原有的选择
2）如果要随机选择 k 个数，只要在第 i 个元素处以 k/i 的概率选择该元素，以 1 - k/i 的概率保持原有选择即可。

```cpp
// don't know how many nums.
int get_rand(ListNode* head){
    if( !head)  return -1;
    
    int res = head->val;
    int cnt = 2;
    ListNode* node = head->next;
    
    double d = 0.0;
    while( node!=NULL){
        d = (double) rand() / RAND_MAX;
        
        if( d< (double)1.0/cnt){  // prob 1/i, choose it.
            res = node->val;
        }
        
        cnt++;
        node = node->next;
    }
    return res;
}


vector<int> get_k_rand(ListNode* head, int k){
    vector<int> res(k, 0);
    ListNode* p = head;
    
    // init with k node
    for(int i=0; i<k&& p!=NULL; i++){
        res[i] = p->val;
        p = p->next;
    }

    int i = k;

    while( p!=NULL){

        int j = rand() % (i+1);

        if( j<k){  //prob k/i, choose new data.
            res[j] = p->val;
        }
        p = p->next;
    }
    return res;

}

int rand_generator(vector<float>& probs){
    int m = probs.size();
    int i=0;
    int res = 0;
    
    while( i<m){
        float p = rand() / RAND_MAX;
        
        if( p< probs[i] ){
            res = i; // choose new
        }

        i++;

    }
    return res;

}

vector<int> vec;
int pick(int target){
    int c= 0;
    int idx = 0;
    
    for(int i=0; i<res.size();i++){
        if( res[i] == target){
            c++;
            
            if( rand() % c==0) idx = i;
        }
    }
    return idx;
}

```







区间合并
```cpp

vector<vector<int>> interval_merge(vector<vector<int>>& intervals){
    int m = intervals.size();

    vector<vector<int>> res;
    if( interval.size()<=1 ) return intervals;

    sort(intervals.begin(), intervals.end(), cmp);

    int i=0;
    while(i<m){
        int left = intervals[i][0];
        int right = intervals[i][1];
        
        // compare to next
        while(i<m-1 &&  right>=intervals[i+1][0]){
            right = max(right, intervals[i+1][1]);
            i++;
        }

        res.push_back( {left, right});
        i++;
    }
    return res;

}

static bool cmp(vector<int>& a, vector<int>& b){
    if( a[0] != b[0]){
        return a[0] < b[0];
    }
    else{
        return a[1] <= b[1];
    }
}


```



寻找消失的元素
1) 只要把所有的元素和索引做异或运算，成对儿的数字都会消为 0，只有这个落单的元素会剩下
```cpp
int missingNumber(int[] nums) {
    int n = nums.length;
    int res = 0;
    res ^= n;
    
    for (int i = 0; i < n; i++)
        res ^= i ^ nums[i];
    return res;
}
```


二分查找高效判定子序列
```cpp
bool isSubsequence(string s, string t) {
    int i = 0, j = 0;
    while (i < s.size() && j < t.size()) {
        if (s[i] == t[j]) i++;
        j++;
    }
    return i == s.size();
}
```


01背包，完全背包
f[j]表示当背包容量为j 时，可以获取的最大价值。
```cpp
for(int i=0; i<n; i++){
    for(int j=0; j>=V[i]; j++){
        f[j] = max(f[j], f[j- V[i]] + W[i]);
    }
}

for(int i=0; i<n; i++){
    for(int j=V[i]; j<=m; j++){
        f[j] = max(f[j], f[j-V[i]] + W[i]);
    }
}



```




```cpp
```


矩阵置零
```cpp
void set_arr_zero(vector<vector<int>>& arr){
    int m = arr.size();
    int n = arr[0].size();

    if( m==0 || n==0) return;

    vector<bool> row(m, false);
    vector<bool> col(n, false);
    
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            if( arr[i][j]==0){
                row[i] = true;
                col[j] = true;

            }
        }
    }


    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            if(row[i] || col[j]){
                arr[i][j] = 0;
            }
        }
    }
}

```



岛屿问题

岛屿周长

岛屿数量


分割等和子集


最大人工岛

岛屿最大面积


最短路径


```cpp
// bfs + priority_queue
double max_probability(int n, vector<vector<int>>& edges, vector<double>& succ_prob, int start, int end){
    
    auto graph = vector<vector<pair<int, double>>>(n);
    for(int i=0; i<edges.size(); i++){
        graph[ edges[i][0]].push_back( {edges[i][1], succ_prob[i]}); // start_point
        graph[ edges[i][1]].push_back( {edges[i][0], succ_prob[i]}); // end_point
    }

    auto mem = vector<double>(n, 0.0);

    priority_queue<pair<double, int>> que;
    que.push( {1.0, start});

    while( !que.empty()){
        auto tmp = que.top();
        que.pop();
        
        if( tmp.first<= mem[tmp.second] ) continue;
        
        mem[tmp.second] = tmp.first;
        for(auto e: graph[tmp.second]){
            que.push( {tmp.first * e.second, e.first});
        }

    }
    return mem[end];

}

class Solution_dfs{
public:
    double max_probability(int n, vector<vector<int>>& edges, vector<double>& succ_prob, int start, int end){
        int i=0;
        for(auto& e: edges){
            graph[ e[0]].push_back( {e[1], succ_prob[i]});
            graph[ e[1]].push_back( {e[0], succ_prob[i]});
            i++;
        }

        dfs(start, end, 1.0);
        return res;
    }

private:

    double res = 0.0;
    static const int N = 10001;
    vector<pair<int, double>> graph[N];
    bool visited[N];
    
    void dfs(int start, int end, double cur){
        if(start==end){
            res = max(res, cur);
            return ;
        }
        
        if(cur<=res || cur<1e-5){
            return ;
        }

        for(auto& nei: graph[start]){
            if(visited[nei.fist]) continue;

            visited[nei.first] = true;
            dfs(nei.first, end, cur* nei.second);

            visited[nei.first] = false;
        }
    }

}




class Solution_BellmanFord{
public:
    double max_probability(int n, vector<vector<int>>& edges, vector<double>& succ_prob, int start, int end){
        float prob[10010] = {0.0};
        prob[start] = 1.0;

        for((int i=0; i<n; i++){
            for(int j=0; j<edges.size(); j++){
                
                auto u = edges[j][0];
                auto v = edges[j][1];
                auto p = succ_prob[j];

                // break u- v into two directed edges
                // handle u-> v
                if(prob[u] != 0.0 && prob[u]*p > prob[v] ){
                    prob[v] = prob[u] * p;
                }

                // handle v-> u
                if(prob[v]!=0.0 && prob[v]*p > prob[u] ){
                    prob[u] = prob[v] * p;
                }
            }
        }
        return prob[end];


    }

private:



}




```


字符串替换
```cpp
string restore_str_1(string& s, vector<int>& indices){
    const int n = s.size();
    
    for(int i=0; i<n;){
        int idx = indices[i];
        if(idx != i){
            swap(s[i], s[idx]);
            swap(indices[i], indices[idx]);
        }
        else{
            i++;
        }
    }
    return s;
}



```




稀疏矩阵乘法
```cpp

vector<vector<int>> multiply(vector<vector<int>>& A, vector<vector<int>>& B) {
    int m = A.size();
    int n = B[0].size();
    int k = B.size();
    // result m*n

    auto res = vector<vector<int>>(m, vector<int>(n, 0));
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            int sum = 0;
            
            for(int t=0; t<k; t++){
                if( A[i][t]==0 || B[t][j]==0) continue;
                sum += A[i][t] * B[t][j];
            }
            res[i][j] = sum;  
        }
    }
    return res;
    
}
```


单调栈实现下一个大的数/下一个全排列
```cpp
vector<int> next_greater_element(vector<int>& nums){
    vector<int> ans(nums.size());
    stack<int> s;

    for(int i=nums.size()-1; i>=0; i--){
        while( !s.empty() && s.top() <=nums[i]){
            s.pop();
        }

        ans[i] = s.empty()? -1:s.top();
        s.push(nums[i]);
    }
    return ans;
}

```


```cpp
vector<int> next_permutation(vector<int>& nums){{
    int i = nums.size()-1;
    int ii = nums.size();

    while(--i>=0 && nums[i]>=nums[i+1]);

    if(i>=0){
        while( nums[i]>= nums[--ii]);

        swap(nums[ii], nums[i]);
    }

    std::reverse(nums.begin()+i+1, nums.end());

}
```


两个栈实现一个队列
```cpp
class CQueue {
public:
    CQueue() {

    }
    
    void appendTail(int value) {
        stk1.push(value);
    }
    
    int deleteHead() {
        if( !stk2.empty()){
            int a = stk2.top();
            stk2.pop();
            return a;
        }
        else if( !stk1.empty()){
            while( !stk1.empty()){
                stk2.push( stk1.top());
                stk1.pop();

            }
            int a = stk2.top();
            stk2.pop();
            return a;
        }
        else{
            return -1;
        }
    }

    bool empty(){
        return stk1.empty() && stk2.empty();
    }

private:
    stack<int> stk1;
    stack<int> stk2;
};

```











