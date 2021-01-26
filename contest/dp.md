

二维区域和检索
sum(abcd) = sum(od) + sum(oa) - sum(ob) - sum(oc)
sum(abcd) is mat[i][j], single element.
sum(od) is dp[i][j]

dp[i][j] = dp[i][j-1] + dp[i-1][j] + mat[i][j] - dp[i-1][j-1]

```cpp
NumMatrix(vector<vector<int>>& mat){
    if(mat.size()==0) return;  // special

    int m = mat.size();
    int n = mat[0].size();
    dp = vector<vector<int>>(m+1, vector<int>(n+1, 0));
    
    for(int i=1; i<m+1; i++){
        for(int j=1; j<n+1; j++){
            dp[i][j] = dp[i-1][j] + dp[i][j-1] + mat[i-1][j-1] - dp[i-1][j-1];
        }
    }

}

int sumRegion(int row1, int col1, int row2, int col2){
    return dp[row1][col1] + dp[row2+1][col2+1] - dp[row2+1][col1] - dp[row1][col2+1];
}

private:
vector<vector<int>> dp;


int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
    int area1 = (C-A)*(D-B);
    int area2 = (G-E) * (H-F);
    // cross
    int m1 = max(A, E);
    int n1 = max(B, F);
    
    int m2 = min(C, G);
    int n2 = min(D, H);
    
    if( m1> m2 || n1>n2) return area1 + area2;
    else{
        return area1 - (m2-m1)*(n2-n1) + area2;
    }

}

```



二叉树中序遍历

```cpp
class Solution{
public:


    int getMinimumDifference(TreeNode* root){
        int ans = INT_MAX;
        int pre = -1;
        dfs(root, pre, ans);

    }

    void dfs(TreeNode* root, int& pre, int& ans){
        if( root==NULL) return;
        
        dfs(root->left, pre, ans);
        
        if(pre==-1){
            pre = root->val;
        }
        else{
            ans = min(ans, root->val - pre); // compute util second node
            pre = root->val;
        }
        
        dfs(root->right, pre, ans);
    }
}


```

中序遍历三种形式，递归，栈，Morris
Morris能将中序遍历的空间复杂度从O(n)变成O(1)
```cpp
class Solution{
public:
    vector<int> inorderTraversal(TreeNode* root){
        vector<int> res;
        inorder(root, res);
        return res;
    }

    void inorder(TreeNode* root, vector<int>& res){
        if( root==NULL) return ;

        inorder(root->left, res);

        res.push_back( root->val);

        inorder(root->right, res);
    }

}


class Solution{
public:
    vector<int> inorderTraversal(TreeNode* root){
        vector<int> res;
        if(root==NULL) return res;
        
        stack<TreeNode*> stk;
        TreeNode* p = root;        
        
        while(p!=NULL || !stk.empty()){
            
            while(p!=NULL){
                stk.push( p);
                p = p->left;
            }
            
            p = stk.top();
            stk.pop();
            res.push_back( p->val);

            p = p->right;
            
        }
        return res;
    }
    
}

class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        TreeNode *predecessor = nullptr;

        while (root != nullptr) {
            if (root->left != nullptr) {
                // predecessor 节点就是当前 root 节点向左走一步，然后一直向右走至无法走为止
                predecessor = root->left;
                while (predecessor->right != nullptr && predecessor->right != root) {
                    predecessor = predecessor->right;
                }
                
                // 让 predecessor 的右指针指向 root，继续遍历左子树
                if (predecessor->right == nullptr) {
                    predecessor->right = root;
                    root = root->left;
                }
                // 说明左子树已经访问完了，我们需要断开链接
                else {
                    res.push_back(root->val);
                    predecessor->right = nullptr;
                    root = root->right;
                }
            }
            // 如果没有左孩子，则直接访问右孩子
            else {
                res.push_back(root->val);
                root = root->right;
            }
        }
        return res;
    }
};




```

形状不同的岛屿数量

1)找到每个岛屿，并记录其坐标(dfs, bfs, 并查集)
2)将坐标按照要求平移（水平/垂直）
3)坐标集里每个坐标减去最小坐标(x,y)后，相同形状的岛屿，其相对坐标集肯定是一样的


可以用set来统计，也可以排序去重的数据结构
set<vector<int>> islands; 用 i*col +j 来存储坐标。
vector<vector<pair<int, int>>> profiles;

sort(profiles.begin(), profiles.end());
unique( profiles.begin(), profiles.end()) - profiles.beigin();

auto new_end = unique( profiles.begin(), profiles.end());  // just put element to last.
profiles.erase( new_end, profiles.end());  // delete 

```cpp

```


字符串
```cpp
string reverse_words(string& s){
    stringstream ss(s);
    string res;
    string word;
    
    while(ss>> word){
        res = "" + word + res;
    }
    if(res[0]=='_'){
        res.erase(0, 1);
    }
    return res;
}

```


反转链表
```cpp
    ListNode* reverseList(ListNode* head) {
        if(head==NULL || head->next==NULL) return head;
        
        ListNode* p = head;
        ListNode* pre = NULL;
        ListNode* fut = head->next;

        while(p!=NULL){
            p->next = pre;

            pre = p;
            p = fut;
            if(fut!=NULL){
                fut = fut->next;
            }
            
        }
        return pre;
    }
```

 
最短路径  
标准bfs解法
```cpp
void BFS()
{
    定义队列;
    定义备忘录，用于记录已经访问的位置；

    判断边界条件，是否能直接返回结果的。

    将起始位置加入到队列中，同时更新备忘录。

    while (队列不为空) {
        获取当前队列中的元素个数。
        for (元素个数) {
            取出一个位置节点。
            判断是否到达终点位置。
            获取它对应的下一个所有的节点。
            条件判断，过滤掉不符合条件的位置。
            新位置重新加入队列。
        }
    }

}


struct Node{
    int x;
    int y;    
};

int shortestPathBinaryMatrix(vector<vector<int>>& grid){
    int ans = 0;
    queue<Node> myq;
    int M = grid.size();
    int N = grid[0].size();

    if(grid[0][0]==1 || grid[M-1][N-1]==1){
        return -1;
    }

    vector<vector<int>> mem(M, vector<int>(N, 0));
    myq.push({0,0});
    mem[0][0] = 1;
    
    while( !myq.empty()){

        int size = myq.size();
        for(int i=0; i<size; i++){
            Node cur = myq.front();
            int x = cur.x;
            int y = cur.y;
            
            // end condition
            if(x==(N-1) && y==(M-1)) return ans+1;
            
            vector<Node> next_nodes={{x+1, y}, {x-1,y}, {x,y+1}, {x,y-1},
                                    {x+1, y+1},{x+1,y-1}, {x-1,y+1},{x-1, y-1}};
                                    
            for(auto& n: next_nodes){
                if(n.x<0 || n.x>=N || n.y<0 || n.y>=M) continue;
                
                if(mem[n.y][n.x]==1) continue;
                
                if(grid[n.y][n.x]==1) continue;

                // candidate
                mem[n.y][n.x] = 1;
                myq.push(n);
            }
            
            myq.pop();
        }
        ans++;

    }
    return -1;

}


```



```cpp

```



最长公共子序列
```cpp
int longest_common_seq_dp_two(string& str1, string& str2){
    int n1 = str1.size();
    int n2 = str2.size();

    vector<vector<int>> dp(n1+1, vector<int>(n2+1, 0));
    
    for(int i=1; i<n1+1; i++){
        for(int j=1; j<n2+1; j++){
            
            if( str1[i-1]==str2[j-1]){
                dp[i][j] = dp[i-1][j-1] +1;
            }
            else{
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[n1][n2];
}

```
    
最长上升子序列
1) O(n^2) dp[i] 表示arr[i]必须被选取的最长上升子序列的长度。
2) 贪心+ 二分查找，O(nlogn)
```cpp
int len_of_LIS(vector<int>& nums){
    int n = nums.size();
    if( n==0) return 0;
    
    vector<int> dp(n, 0);
    for(int i=0; i<n; i++){
        dp[i]=1;

        for(int j=0; j<i; j++){
            if( nums[j]< nums[i]){
                dp[i] = max(dp[i], dp[j]+1);
            }
        }
    }
    return *max_element( dp.begin(), dp.end());
}

int len_of_LIS_greedy(vector<int>& nums){
    int n = nums.size();
    if(n==0) return 0;

    vector<int> d(n+1, 0);
    int len = 1;
    d[len] = nums[0];

    for(int i=1; i<n; ++i>{
        if( nums[i]> d[len]){
            d[++len] = nums[i];

        }
        else{
            // nums[i] is smallest if it can't be found.
            int l = 1;
            int r = len;
            int pos =0;
            
            while( l<=r){
                int mid = l + (r-l)/2;
                if( d[mid] < nums[i]){
                    pos = mid;  // record pos in d arr.
                    l = mid + 1;
                }
                else{
                    r = mid -1;
                }
            }
            d[pos+1] = nums[i];

        }
    }
    return len;  

}


```

三角形最小路径和
1）一般空间和时间都是O(n2)
2)空间可以降到O(n)

第i行有i+1个数。f[i][j]表示从三角形顶部走到位置(i,j)的最小路径和。
每一步只能移动到下一行相邻的节点上。
f[i][j] = min( f[i-1][j-1], f[i-1][j]) + c[i][j]
用一维度来存储的话
f[i] = min(f[i-1], f[i]) + c[i][j]

```cpp
int minimum_total(vector<vector<int>>& triangle){
    int n = triangle.size();
    vector<int> f(n);
    f[0] = triangle[0][0];
    
    for(int i=1; i<n; i++){
        f[i] = f[i-1] + triangle[i][i];

        for(int j=i-1;j>0; --j){
            f[j] = min(f[j-1], f[j]) + triangle[i][j];
        }
        f[0] += triangle[i][0];
        
    }
    return *min_element( f.begin(), f.end());
}

```

连续子数组的最大和
1) f[i]表示以 i位置的数结尾的连续子数组的最大和。 f[i] = max( f[i-1] + a, a)
2) 分治方法类似于 线段树求解LCIS问题的pushUp操作。

```cpp
// time O(n), space O(1)
int maxSubArray(vector<int>& nums){
    int res = nums[0];
    int sum = 0;
    for(int x: nums){
        if(sum>0){
            sum += x;
        }
        else{
            sum = x;
        }

        res = max(res,  sum);
    }
    return res;
    
}

// time O(logn), space O(logn)
class Solution{
public:
    struct Status{
        int lsum;
        int rsum;
        int msum;
        int isum;
    };

    Status pushUp(Status l, Status r){

    };

    Status get(vector<int>& a, int l, int r){
        
    }

    int maxSubArray(vector<int>& nums){
        return get(nums, 0, nums.size()-1).msum;

    }

}


```

线段树能求解区间合并问题，多次询问的区间最长连续 上升序列问题，区间最大子段和问题等。

```cpp

```


```cpp

```


最长上升子序列
```cpp

```



最长上升子序列
```cpp

```



最长上升子序列
```cpp

```


最长上升子序列
```cpp

```


不同的岛屿数量

1) 不用维护visited矩阵，直接将每次访问过的陆地置为0即可。
2) 其他可以尝试把路径用str存起来，用set来去重。
   
```cpp
class Solution{
public:
    int numDistinctIslands(vector<vector<int>>& grid){
        if(grid.size()==0 || grid[0].size()==0) return 0;
        int M = grid.size();
        int N = grid[0].size();

        set<vector<int>> islands;
        for(int i=0; i<M; i++){
            for(int j=0; j<N; j++){
                
                vector<int> island;
                dfs(grid, i, j, i, j, island);  // find an island
                
                if( !island.empty()) islands.insert( island);

            }
        }
        
    }

private:
    void dfs(vector<vector<int>>& grid, int sr, int sc, int r, int c, vector<int>& island){
        if( r<0 || c<0 || r>=grid.size() || c>=grid[0].size() ) return ;
        
        if(grid[r][c]==0) return;

        grid[r][c] = 0;
        
        island.push_back( (r-sr)*N + (c-sc));
        
        dfs(grid, sr, sc, r-1, c, island);
        dfs(grid, sr, sc, r+1, c, island);
        dfs(grid, sr, sc, r, c-1, island);
        dfs(grid, sr, sc, r, c+1, island);

    }
    

};

```

单词搜索
1) dfs
```cpp
bool exist(vector<vector<char>>& board, string& word){
    if( board.empty() || word.empty()) return false;
    if( board[0].size()==0) return false;

    int M = board.size();
    int N = board[0].size();
    vector<vector<int>> f(M, vector<int>(N, 0));  // visited
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            if( dfs(board, word, 0, i, j, f)){
                return true;
            }
            
        }
    }
    return false;
    
}

bool dfs(vector<vector<char>>& board, string& word, int idx, int x, int y, vector<vector<int>>& f){
    if( idx==word.size() )  return true;

    if(x<0 || y<0 || x>=board.size() || y>=board[0].size()) return false;

    if( f[x][y] || board[x][y] !=word[idx]) return false;

    f[x][y] = 1;

    bool b1 = dfs(board, word, idx+1, x+1, y, f);
    bool b2 = dfs(board, word, idx+1, x-1, y, f);
    bool b3 = dfs(board, word, idx+1, x, y+1, f);
    bool b4 = dfs(board, word, idx+1, x, y-1, f);
    if(b1 || b2 || b3 || b4) return true;

    // if(dfs(board, word, idx+1, x+1, y, f) ||
    //  dfs(board, word, idx+1, x-1, y, f) || 
    //  dfs(board, word, idx+1, x, y+1, f) || 
    //  dfs(board, word, idx+1, x, y-1, f)) return true;

    f[x][y] = 0;
    return false;
}


```



统计封闭岛屿的数量
1）被1 包围的区域。靠边的0不能算。

```cpp
int closedIsland(vector<vector<int>>& grid){
    if(grid.size()==0 || grid[0].size()==0 ) return 0;
    int M = grid.size();
    int N = grid[0].size();
    int num = 0;

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            
            if( grid[i][j]==0){ // find 
                if( !dfs(grid, i, j)){
                    num++;
                }
            }
        }
    }
    return num; 
    
}
// whether get to border
bool dfs(vector<vector<int>>& grid, int r, int c){
    if(r<0 || c<0 || r>=grid.size() || c>=grid[0].size()) return true;
    
    // 0 water, 1 land, 2 visited. we hope water.
    if( grid[r][c]!=0) return false;

    grid[r][c] = 2;

    bool b1 = dfs(grid, r-1, c);
    bool b2 = dfs(grid, r+1, c);
    bool b3 = dfs(grid, r, c-1);
    bool b4 = dfs(grid, r, c+1);
    return b1 || b2 || b3 || b4;
}

```


最大人工岛
可以人工把填一块土。求之后的岛最大面积。
注意超时。
1) 对于每个0，可以将其变成1， 然后统计这个连通块大小。
最大面积肯定出现在跟这个0有关的连通块面积上。

```cpp
class Solution{
public:
    int largestIsland(vector<vector<int>>& grid){
        n = grid.size();
        
        for(int i=0; i<n; i++){
            for(int j=0; j<n; ++j){
                if( grid[i][j]==1){
                    
                }
            }
        }
    }



private:
    int n;
    int color = 2;
    int max_area = 0;
    vector<vector<int>> dir={{1,0}, {-1, 0}, {0,1},{0,-1}};
    unordered_map<int, int> m; // color area

    

}


```

最长上升子序列
```cpp

```


最大岛屿周长
```cpp
int islandPerimeter(vector<vector<int>>& grid){
    int res = 0;
    int M = grid.size();
    int N = grid[0].size();
    
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            
            if(grid[i][j]==1){
                res = max(res, dfs(grid, i, j));

            }
        }
    }
    return res;
    
}

int dfs(vector<vector<int>>& grid, int i, int j){
    // get to border, add 1
    if(i<0 || j<0 || i>=grid.size() || j>=grid[0].size()) return 1;

    // get to water, add 1
    if(grid[i][j]==0) return 1;

    if(grid[i][j] == 2) return 0; // visited

    grid[i][j] = 2;
    
    return dfs(grid, i-1, j) +
            dfs(grid, i+1, j) + 
            dfs(grid, i, j-1) + 
            dfs(grid, i, j+1);
  
}

```

路径是否相交
```cpp
bool isPathCrossing(string path){
    
}

int get_hash(int x, int y){
    return x*20001 + y;
}

```

分割等和子集
dp[i][j] 表示从数组[0,i]这个区间挑选一些正整数，使其和为j
dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i]];

```cpp
bool canPartition(vector<int>& nums){
    int n = nums.size();
    if(n==0) return false;

    int sum=0;
    for(int e: nums){
        sum += e;
    }

    if(sum%2==1) return false;
    int target = sum/2;

    vector<vector<bool>> dp(n, vector<bool>(target+1,0) );
    dp[0][0] = true;
    
    if(nums[0]==target){
        dp[0][nums[0] ] = true;
    }
    for(int i=1; i<n; i++){
        for(int j=0; j<=target; j++){ // huge
            
            dp[i][j] = dp[i-1][j];

            if(nums[i] <=j){
                dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]];
            }
        }

        // end condition
        if(dp[i][target]){
            return true;
        }
        
    }
    return dp[n-1][target];

}

```


零钱兑换
S总金额，N是硬币种类。
1) time O(sn), space O(s)
f[s] 表示总和为s 所需的最少硬币数量
f[s] = f[s-c] + 1

f[3] = min( f[3-c1], f[3-c2], f[3-c3]) + 1


```cpp
int coinChange(vector<int>& coins, int amount){
    int max = amount +1;
    int n = coins.size();
    
    vector<int> dp(amount+1, max);
    dp[0] = 0;
    
    for(int i=1;  i<=amount; i++){
        for(int j=0; j<n; j++){

            if( coins[j] <=i ){
                dp[i] = min(dp[i], dp[i-coins[j]] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1: dp[amount];
    
}

```

重复的子字符串
KMP算法, time O(n)

KMP 算法虽然有着良好的理论时间复杂度上限，但大部分语言自带的字符串查找函数并不是用 KMP 算法实现的。这是因为在实现 API 时，我们需要在平均时间复杂度和最坏时间复杂度二者之间权衡。普通的暴力匹配算法以及优化的 BM 算法拥有比 KMP 算法更为优秀的平均时间复杂度；

```cpp


```


迷宫

深搜只能调到28ms，广搜4ms

思路1:广度优先搜索，68ms、16.5MB，80%、75%

队列BFS
遇到1之前沿着同一方向继续前进
遇到1后上一个到达的位置加入队列，并标记
思路2:深度优先搜索，68ms、17MB，80%、67%

已访问点跳过
碰到墙壁后回弹一位，继续搜索

```cpp

```

最长上升子序列
```cpp

```



```cpp

```


### 线段树
time O(nlogn), space n


699 掉落的方块
```cpp
class SegmentTree(object):
    def __init__(self, N, update_fn, query_fn):
        self.N = N
        self.H = 1
        while i<< self.H <N:
            self.M += 1

        self.update_fn = update_fn
        self.query_fn = query_fn
        self.tree = [0] *(2*n)
        self.lazy = [0] *N

    def _apply(self, x, val):
        self.tree[x] = self.update_fn(self.tree[x])
        if x < self.N:
            self.lazy[x] = self.update_fn(self.lazy[x], val)
        
    def _pull(self, x):
        while x>1:
            x/=2
            self.tree[x] = self.query_fn(self.tree[x*2], self.tree[x*2+1])
            self.tree[x] = self.update_fn(self.tree[x], self.lazy[x])
            
    def _push(self, x):
        for h in xrange(self.H, 0 , -1):
            y = x>>h
            if self.lazy[y]:
                self._apply( y*2, self.lazy[y])
                self._apply( y*2+1, self.lazy[y])
                self.lazy[y] = 0

    def update(self, L, R, h):
        L += self.N
        R += self.N
        L0, R0 = L, R
        while L <=R:
            if L & 1:
                self._apply(L, h)
                L += 1
            if R & 1==0:
                self._apply(R, h)
                R -=1
            L /=2;R /=2
        self._pull(L0)
        self._pull(R0)
        
    def query(self, L, R):
        L += self.N
        R += self.N
        self._push( L)
        self._push( R)
        ans = 0
        
        while L <=R:
            if L & 1:
                ans = self.query_fn( ans, self.tree[L])
                L += 1
            if R & 1==0:
                ans = self.query_fn( ans, self.tree[R])
                R -=1
            L /=2; R/=2
        return ans

class Solution(object):
    def falling_squares(self, pos):
        tree = Segment( len(index), max, max)
        best = 0
        ans = []

        for left,size in pos:
            L,R = index[left], index[left+ size-1]
            h = tree.query(L, R) + size
            tree.update(L, R, h)
            best = max( best, h)
            ans.append( best)

        return ans
            
```



850 矩形面积

```python
global X
X = set()

class Node(object):
    def __init__(self, start, end, X):
        self.start = start
        self.end = end
        self.total = 0
        self.count = 0
        self._left = None
        self._right = None

    @property
    def mid(self):
        return self.start + (self.end-self.start)/2

    
    def left(self):
        self._left = self._left or Node(self.start, self.mid)
        return self._left
        
    def right(self):
        self._right = self._right or Node(self.mid, self.end)
        return self._right

    def update(self, i, j, val):
        if i>=j: return 0
        
        if self.start==i and self.end==j:
            self.count += val
        else:
            self.left.update(i, mid(self.mid, j), val)
            self.right.update( max(self.mid, i), j, val)
        
        if self.count >0:
            self.total = X[self.end] - X[self.start]
        else: 
            self.total = self.left.total + self.right.total
        
        return self.total

class Solution(object):
    def rectangle_area(self, rectangle):
        OPEN, CLOSE = 1, -1
        events = []

        for x1, y1, x2, y2 in rectangles:
            events.append( (y1, OPEN, x1, x2))
            events.append( (y2, CLOSE, x1, x2))
            X.append(x1)
            X.append(x2)
        events.sort()
        
        X = sorted(X)
        Xi = {x: i for i, x in enumerate(X)}
        
        active = Node(0, len(X)-1)
        ans = 0
        cur_x_sum = 0
        cur_y = events[0][0]
        
        for y, typ, x1, x2 in events:
            ans += cur_x_sum  * (y- cur_y)
            cur_x_sum = active.update( Xi[x1], Xi[x2], typ)
            cur_y = y

        return ans % (10**9 + 7)
    

```



### 并查集
```cpp

```


前k 个 高频词
可以用堆排序来求解，或者用字典树TrieTree


1）排序好后是NlogN
2）堆排序是 Nlogk


692
```python
def topK_freq(self, words, k):
    count = collections.Counter( words)
    heap = [ (-freq,word) for word,freq in count.items()]
    
    heapq.heapify( heap)
    return [ heapq.heappop( heap)[1] for _ in range(k)]
```



692
```cpp
struct TrieNode{
    bool is_end;
    TreNode* branch[26];
    // int times;
    // string s;
    pair<int, string> show;

    TrieNode():is_end(false){
        show.first = 0;
        for(int i=0; i<26; i++){
            branch[i] = nullptr;
        }
    }

};

class Solution{
public:
    vector<string> topK_frequent(vector<string>& words, int k){
        root = new TrieNode();
        for(auto it: words){
            insert(it); // setup
        }

        K = k;
        vector<string> res(k);
        dfs(root);
        
        for(int i=0; i<k; i++){
            res[i] = p.top().second();
            p.pop();
        }

        reverse(res.begin(), reg.end());
        return res;

    }

    void insert(string& word){
        TrieNode* node = root;
        for(auto it: word){
            if(node->branch[it - 'a']== nullptr){
                node->branch[it - 'a'] = new TrieNode();
            }
            node = node->branch[it-'a'];
        }
        node->is_end = true;
        node->show.first++;
        node->show.second = word;

    }

    void dfs(TrieNode* root){
        if( root==nullptr) return ;
        
        if(root->is_end){// setup heap
            p.push( root->show);
            if( p.size()>K){
                p.pop();
            }

        }

        for(int i=0; i<26; i++){ // deep
            if(root->branch[i] != nullptr){
                dfs( root->branch[i]);
            }
        }
    }

    struct comp{
        bool operator()(pair<int, string> a, pair<int, string> b){
            if( a.first != b.first){
                return a.first > b.first;
            }
            else{
                return a.second < b.second;
            }
            
            
        }
    }



private:
    TrieNode* root;
    int K;
    priority_queue<pair<int, string>, vector<pair<int, string>>, cmp> p;

}


```

745 前缀和后缀搜索
解决方案
1）成对的单词查找树
2）后缀修饰的单词查找树
两种time 都是 NK^2 + QK, space NK^2

```python
Trie = lambda: collections.defaultdict(Trie)
WEIGHT = False

# class WordFilter(object):
#     def __init__(self, words):
#         pass

#     def f(self, prefix, suffix):
#         pass

class WordFilter(object):
    def __init__(self, words):
        self.trie = Trie()

        for weight,word in enumerate(words):
            cur = self.trie
            cur[WEIGHT] = weight
            for i,x in enumerate(word):
                # put all prefixs and suffixs

                tmp = cur
                for letter in word[i:]:
                    tmp = tmp[letter, None]
                    tmp[WEIGHT] = weight
                # advance letters
                cur = cur[x, word[~i]]
                cur[WEIGHT] = weight
                
    def search(self, prefix, suffix):
        cur = self.trie
        for a,b in map(None, prefix, suffix[::-1]):
            if (a,b) not in cur:
                return -1
            cur = cur[a, b]
        return cur[WEIGHT]

```


```python
```

```py
Trie = lambda: collections.defaultdict(Trie)
WEIGHT = False

class WordFilter(object):
    def __init__(self, words):
        self.trie = Trie()
        
        for weight,word in enumerate(words):
            word += "#"
            for i in range( len(word)):
                cur = self.trie
                cur[WEIGHT] = weight
                for j in range(i, 2*len(word)-1):
                    cur = cur[word[j % len(word)]]
                    cur[WEIGHT] = weight
    
    def f(self, prefix, suffix):
        cur = self.trie
        for letter in suffix + "#" + prefix:
            if letter not in cur:
                return -1
            cur = cur[letter]
        return cur[WEIGHT]

        
```


目标和 494

n是数组的长度。
1）dfs，time 2^n, space n, 未递归所使用的栈。
2) 动态规划，time和space都是 n*sum
3) 动态规划+空间优化，time n*sum， space sum
4) dfs + 减枝

d[i][j] 表示数组中前i 个元素，组成和为 j的方案数。
d[i][j] = d[i-1][j-nums[i]] + d[i-1][j+nums[i]]
递推形式为

d[i][j+nums[i]] += d[i-1][j]
d[i][j-nums[i]] += d[i-1][j]

由于所有数的和不超过1000，那么j最小值可达到 -1000.可以先加1000.

d[i][j+nums[i]+1000] += d[i-1][j+1000]
d[i][j-nums[i]+1000] += d[i-1][j+1000]

由于d[i][...] 只和 d[i-1][...]有关，可以优化空间。用两个一维度数组即可。


```cpp
class Solution{
public:
    int count = 0;
    int findTargetSumWays(vector<int>& arr, int s){
        dfs(nums, 0, 0, s);
        return count;
    }

    void dfs(vector<int>& nums, int pos, int sum, int s){
        if(pos==nums.size() && sum==s){
            count++;
            return ;
        }
        if(pos>=nums.size()) return ;

        dfs(nums, pos+1, sum+ nums[pos], s);
        dfs(nums, pos+1, sum- nums[pos], s);
    }

    int find_target_sum_ways(vector<int>& nums, int s){
        vector<int> d(2001, 0);
        d[nums[0]+1000] += 1;
        d[-nums[0]+1000] +=1;

        for(int i=1; i<nums.size(); i++){
            vector<int> next(2001, 0);
            
            for(int sum=-1000; sum<=1000; sum++){
                if( d[sum+1000]>0){
                    next[sum+nums[i]+1000] += d[sum+1000];
                    next[sum-nums[i]+1000] += d[sum+1000];

                }
            }
            d = next;
        }
        return s>1000? 0: d[s+1000];
    }

}

```


```cpp
/*
 * @lc app=leetcode.cn id=494 lang=cpp
 *
 * [494] 目标和
 *
 * 思路：dfs+剪枝，dfs部分比较简单，一个数字就是正或负。
 * 剪枝是统计后面所有的数字之和，当前结果全加或全减后面数字之和能否变号，如果不能就剪去。对数组从大到小排序后效果更好。
 *
 */

// @lc code=start
class Solution {
public:
    bool check(const long long rest, const long long sum)
    {
        return (rest+sum) * (rest-sum) <= 0;
    }

    // 这里的sum指的是从pos+1位之后所有数的和
    void dfs(const vector<int>& nums, const int pos, const long long rest,
             long long sum, int& cnt)
    {
        if (pos == nums.size())
        {
            if (rest == 0)
                cnt++;
            return;
        }
        sum -= nums[pos];
        if (check(rest+nums[pos], sum))
            dfs(nums, pos+1, rest+nums[pos], sum, cnt);
        if (check(rest-nums[pos], sum))
            dfs(nums, pos+1, rest-nums[pos], sum, cnt);
    }

    int findTargetSumWays(vector<int>& nums, int S) {
        sort(nums.begin(), nums.end(), [](const int& a, const int& b) { return a > b; });
        long long sum = 0;
        for(const int& n: nums)
            sum += n;

        int cnt = 0;
        dfs(nums, 0, S, sum, cnt);
        return cnt;
    }
};
// @lc code=end

```



朋友圈
1）dfs，n^2,整个矩阵都要访问。space n，visited数组的大小。
2）bfs，n^2,整个矩阵都要访问。space n，queue和visited数组的大小。
3）union-find，time n^3, 访问整个矩阵一次，并查集操作一次最坏需要n时间。
space n. parent大小为n

```cpp
// dfs
int find_circle_num(vector<vector<int>>& M){
    int n = M.size();
    vector<int> visited(n, 0);

    for(int i=0; i<n; i++){
        if( visited[i]==0){
            dfs(M, visited, i); // get  all his friend visited.
            count++;
        }
    }
    return count;
}

// m*m arr
void dfs(vector<vector<int>>& M, vector<int>& visited, int i){
    for(int j=0; j<M.size(); j++){

        if( M[i][j]==1 && visited[j]==0){
            visited[j]=1;
            dfs(M, visited, j);
        }
    }
}


// bfs
int find_circle_num_bfs(vector<vector<int>>& M){
    int n = M.size();
    vector<int> visited(n, 0);
    
    int count=0;
    queue<int> q;

    for(int i=0; i<n; i++){

        if(visited[i]==0){

            q.push( i);
            while( !q.empty()){
                int a = q.front();
                q.pop();

                visited[a]=1;
                for(int j=0; j<n; j++){
                    if( M[a][j]==1 && visited[j]==0){
                        q.push( j);
                        // cout << i << "," << j << "," << count << endl;
                    }
                }
            }

            count++;
            
        }
    }
    return count;
}


// union-find
class Solution{
public:
    int find(vector<int>& parent, int i){
        if(parent[i]==-1)  return -1;

        return find(parent, parent[i]);
    }

    void union(vector<int>& parent, int x, int y){
        int xset = find(parent, x);
        int yset = find(parent, y);
        
        if(xset !=yset){
            parent[xset] = yset;
        }
    }

    int find_circle_num(vector<vector<int>>& M){
        int n = M.size();
        vector<int> parent(n);
        for(auto& e: parent)  e= -1;

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                
                if(M[i][j]==1) && i!=j){
                    union(parent, i, j);
                }
            }
        }

        int count = 0;
        for(int i=0; i<n; i++){
            if(parent[i]==-1){
                count++;
            }
        }
        return count;
    }

}


```


261  以图判树

图方面的算法比较薄弱，在此对评论区几种方法做个总结：
对于该题目的思路，主要有两种

是连通图且不存在环
是连通图且边的个数==节点数-1
实现方式：
对于连通图的判定，有两种方式(代码都有)：
以广度优先搜索或者深度优先搜索的方式，遍历一遍图。如果存在没有遍历到的节点，那么是非连通图，返回false.
并查集：最后如果有多个头目，则是非连通图，返回false.
存在环的判定：
深度优先遍历，把边给数一下。因为数的时候，会数生成树最少的边数(形成环的边会因为节点被访问过而不计算,如下图：深度遍历时，只会遍历1,2和2,3之间的边，13之间的边不会遍历)，所以最终数出来的边数<总边数，则形成环。广度优先遍历，同理。
1
|  \
2 --- 3
并查集：如果并查集建立的过程中发生合并，则一定有环形成。
边的数==节点数-1的判定：很直白了，无需多说。
上述思路都是重叠的，广度优先/深度优先/并查集都可以实现，而且使用不同的思路。


```cpp
class Solution_dfs {
public:
    bool validTree(int n, vector<vector<int>>& edges) {
        if (edges.size() != n - 1) {
            return false;
        }

        visited = vector<bool>(n, false);
        vector<unordered_set<int>> graph(n);
        for (const auto& edge : edges) {
            graph[edge[0]].insert(edge[1]);
            graph[edge[1]].insert(edge[0]);
        }

        dfs(0, graph);
        return all_of(visited.begin(), visited.end(), [](bool i) { return i;});
    }

private:
    vector<bool> visited;
   void dfs(int i, vector<unordered_set<int>>& graph) {
       if (visited[i]) {
           return;
       }

       visited[i] = true;
       for (const auto& neighbor : graph[i]) {
           dfs(neighbor, graph);
       }
   }
};


class Solution {
public:
    bool validTree(int n, vector<vector<int>>& edges) {
        if (edges.size() != n - 1) {
            return false;
        }

        vector<bool> visited(n, false);
        vector<unordered_set<int>> graph(n);
        for (const auto& edge : edges) {
            graph[edge[0]].insert(edge[1]);
            graph[edge[1]].insert(edge[0]);
        }

        queue<int> q{{0}};
        while (!q.empty()) {
            auto cur = q.front(); q.pop();
            visited[cur] = true;
            for (const auto& neighbor : graph[cur]) {
                graph[neighbor].erase(cur);
                q.push(neighbor);
            }
        }

        return all_of(visited.begin(), visited.end(), [](bool i) { return i;});
   }
};


class Solution {
public:
    bool validTree(int n, vector<vector<int>>& edges) {
        if (edges.size() != n -1) {
            return false;
        }

        UnionFind uf(n);
        for (const auto& edge : edges) {
            auto u = edge[0];
            auto v = edge[1];
            auto pu = uf.find(u); 
            auto pv = uf.find(v); 
            if (pu == pv) {
                return false;
            }

            uf.unite(pu, pv);
        }

        return true;
    }

private:
    class UnionFind {
    public:
        UnionFind(int n) {    
            parent = vector<int>(n, 0);
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }
            

        void unite(int x, int y) {
            auto px = find(x);
            auto py = find(y);
            if (px != py) {
                parent[px] = py;
            }
        } 

        int find(int x) {
            if (x == parent[x]) {
                return x;
            }

            return find(parent[x]);
        }  
    private:
        vector<int> parent;
    };
};


```


最长递增子串的长度

```cpp

```


999 可以被一步捕获的棋子数
1) time n^2, space n
遍历棋盘需要n^2, ，模拟车在四个方向上捕获颜色相反的卒需要n，总共需要n^2

```cpp
int numRookCaptures(vetor<vector<char>>& board){
    int cnt = 0;
    int st=0;
    int ed=0;
    int dx[4] = {0, 1, 0, -1};
    int dy[4] = {1, 0, -1, 0};
    for(int i=0; i<8; i++){
        for(int j=0; j<8; j++){
            if(board[i][j]=='R'){
                st = i;
                ed = j;
                break;
            }
        }
    }
    
    for(int i=0; i<4; i++){

        for(int step=0; ;step++){
            int tx = st + step*dx[i];
            int ty = ed + step*dy[i];

            if(tx<0 || tx>=8 || ty<0 || ty>=8 ||board[tx][ty]=='B'){
                break;
            }

            if(board[tx][ty]=='p'){
                cnt++;
                break;
            }
            
        }
    }
    return cnt;
}

```

1005 k次取反后最大化的数组和
2) 原地排序,这个速度 8ms
1）小顶堆，每次改变最小的元素，再插入堆中  20ms

```cpp
int largestSumAfterKNegations(vector<int>& A, int K){
    sort(A.begin(), A.end());
    int sum = 0;
    int min_val = INT_MAX;
    for(auto& val: A){
        if(val<0 && K>0){
            K--;
            val = -val;
        }

        sum += val;
        min_val = min(min_val, val);
    }

    if(K>0 && (K&1)){
        sum = sum - 2*min_val;
    }
    return sum;
}

int largestSumAfterKNegations(vector<int>& A, int K){
    int ans = 0;
    priority_queue<int, vector<int>, greater<int> > my_q;
    
    for(auto& a: A) my_q.push(a);

    while(K--){
        int tmp = -my_q.top();
        my_q.pop();

        myq.push(tmp);
    }

    while( !my_q.empty()){
        ans += my_q.top();
        my_q.pop();
    }
    return ans;

}
```


1024 视频拼接
1）动态规划，time T*N
2) 贪心，time T+N

```cpp
int videoStitching(vector<vector<int>>& clips, int T){
    vector<int> maxn(T);
    int last = 0;
    int pre = 0;
    int res = 0;

    for(auto& it: clips){
        if(it[0] < T){
            maxn[ it[0]] = max( maxn[it[0]], it[1]);
        }
    }

    for(int i=0; i<T; i++){
        last = max(last, maxn[i]);
        if(i==last) return -1;
        
        if(i==pre){
            res++;
            pre = last;
        }
    }
    return res;
}

int videoStitching(vector<vector<int>>& clips, int T){
    vector<int> d(T+1, INT_MAX-1);
    d[0] = 0;
    
    for(int i=1; i<=T; i++){

        for(auto& it: clips){
            if( it[0]<i && i<=it[i]){
                d[i] = min(d[i], d[it[0]] +1);
            }
        }
    }
    return d[T]==INT_MAX-1? -1: d[T];
}

```

跳跃游戏
1）贪心 time n, space 1
```cpp
bool canJump(vector<int>& nums){
    int n = nums.size();
    int right_most = 0;

    for(int i=0; i<n; i++){
        if(i<=right_most){
            right_most = max(right_most, i+nums[i]);
            
            if(right_most>=n-1){
                return true;
            }
        }
    }
    return false;
}

```

440 字典序的第K小数字
1）十叉树
k表示要找的第k个元素，起始下标是0.
获取以prefix 开头的数字个数，包括其本身
如果数字个数大于k，下移，在prefix*10 的子树进行查找
如果数字个数小于k，右移，在 prefix +1 的子数进行查找
根节点 [prefix, prefix+1)
第一层 [prefix*10, (prefix+1)*10)
第二层 [prefix*100,  min(n+1, (prefix+1)*100) )

```cpp
typedef long long LL;

int findKthNumber(int n, int k){
    int prefix = 1;
    k--;

    while(k>0){
        int cnt = get_cnt(n, prefix);
        
        if(cnt<=k){  // right
            k -=cnt;
            prefix++;
        }
        else{  // down
            k--;
            prefix *=10;
        }
    }
    return prefix;
}

int get_cnt(LL n, LL prefix){
    LL cnt=0;
    
    LL first = prefix;
    LL second = prefix+1;
    while(first<=n){
        cnt += min(n+1, second)-first;
        
        first *=10;
        second *=10;
    }
    return cnt;
}
```



```cpp

```


```cpp

```


```cpp

```





