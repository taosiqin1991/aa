
28 实现 strStr()

O(n)有两种方式
Rabin-Karp，通过哈希算法实现常数时间窗口内字符串比较。
比特位操作，通过比特掩码来实现常数时间窗口内字符串比较。

time O(m+n)
```cpp
// kmp
vector<int> get_next(string& p){
    int n=p.size();
    vector<int> next(n, 0);
    int i=1;
    int len=0;
    while(i< n){  // O(n)
    
        if(p[i]== p[len]){
            ++len;
            next[i++] =len;
        }
        else if( len==0){  // match failed
            next[i++]=0;
        }
        else{   //match failed
            len = next[len-1];
        }
    }
    return next;
}


int kmp(string& s, string& p){
    vector<int> next=get_next(p);
    int ns =s.size();
    int np =p.size();
    int i=0;
    int len=0;  // j
    while(i< ns){  // O(m)
        if(s[i]== p[len]){ 
            i++; 
            len++;
            if(len==np) return i-len; 
        }
        else if(len==0 ) ++i;
        else  len= next[len-1];
    }    
    return -1;
}

int strStr(string s, string p){
    if(p.size()==0) return 0;
    return kmp(s, p);
}n
```

1392 最长快乐前缀
kmp
```cpp
string longestPrefix(string s){
    int n = s.size();
    vector<int> next(n, 0);
    for(int i=1; i<n; i++){
        int tmp = next[i-1];
        
        while(tmp>0 && s[tmp]!=s[i]){
            tmp = next[tmp-1]; //
        }
        if(s[tmp]==s[i]) ++tmp;
        
        next[i] = tmp;
    }
    return s.substr(0, next.back()); // 
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





无重复字符的最长子串

1）滑动窗口  time O(N)
2）
3）
```cpp
int len_of_longest_substring(string s){
    unordered_set<char> occ;
    int n = s.size();

    int rk = -1;
    int ans = 0;
    
    for(int i=0; i<n; i++){
        if(i!=0){
            // left pointer move
            occ.erase( s[i-1]);
        }

        while( rk+1<n && !occ.count( s[rk+1]) ){
            // move right pointer
            occ.insert( s[rk+1]);
            rk++;
        }

        // [i, rk+1]
        ans = max(ans, rk-i+1);
    }
    return ans;
}


```

右边第一个比它大的数
1)单调栈处理
```cpp
vector<int> greater(vector<int>& arr){
    int n = arr.size();
    vector<int> res(n, -1);
    
    stack<int> stk;
    stk.push(0);

    for(int i=1; i<n; i++){
        while( !stk.empty() && arr[i] > arr[ stk.top()]){
            res[ stk.top()] = arr[i];
            stk.pop();
        }
        stk.push(i);
    }
    return res;
}

```




链表数相加
1）将较短的链表头部补充0，考虑进位
1）不补齐，考虑进位

```cpp
ListNode* add_two_num(ListNode* l1, ListNode* l2){
    ListNode head = ListNode(-1);
    ListNode* p = &head;
    
    int sum = 0;
    bool carry = 0;
    while(l1!=nullptr || l2!=nullptr){
        sum = 0;
        
        if(l1!=nullptr){
            sum += l1->val;
            l1 = l1->next;
        }
        if(l2!=nullptr){
            sum += l2->val;
            l2 = l2->next;
        }
        if(carry){
            sum++;
        }
        
        p->next = new ListNode( sum%10);
        p = p->next;
        carry = sum>=10? true: false;
    }

    if(carry)  p->next = new ListNode(1);
    return head.next;
}


```

最长回文子串
1）动态规划 time n^2, space n^2
2) 中心扩展法 time n^2, space n
3) Manacher算法 time n
   
动态规划  P(i,j) = true，如果[i,j]是回文串
状态转移方程为  p(i,j) = p(i+1, j-1)&&(si==sj)  两个条件同时满足才真

边界条件
p(i, i) = true
p(i, i+1) = (s_i==s_(i+1))
答案为 p(i,j)=true中 j-i+1 的最大值


优秀解法
```cpp
string longestPalindrome(string s){
    if(s=="") return "";
    int n = s.size();
    int idx = 0;
    int maxl= 0;
    int begin = 0;
    
    while(idx < n){
        int right = idx;
        int left = idx;
        
        while(idx < n && s[idx+1]==s[idx]){
            idx++;
            right++;
        }

        while(right<n && left>=0 && s[right]==s[left]){
            right++;
            left--;
        }

        right--;
        left++;

        if(right-left+1> maxl){
            maxl = right-left+1;
            begin = left;
        }

        idx++;

    }
    return s.substr(begin, maxl);
}

```


```cpp
// manacher
string longestPalindrome(string s){
    int start = 0;
    int end = -1;
    string t = "#";
    for(char c: s){
        t += c;
        t +="#";
    }
    t+= "#";
    s = t;

    vector<int> arm_len;
    int right = -1;
    int j = -1;
    for(int i=0; i<s.size(); i++){
        int cur_arm_len;
        if(right>=i){
            int i_sum = 2*j-i;
            int min_arm_len = min( arm_len[i_sym, right-i]);
            cur_arm_len = expand(s, i-min_arm_len, i+min_arm_len);

        }
        else{
            cur_arm_len = expand(s, i, i);
        }

        arm_len.push_back( cur_arm_len);
        if(i + cur_arm_len > right){
            j = i;
            right = i + cur_arm_len;
        }
        if(cur_arm_len * 2 + 1 > end -start){
            start = i- cur_arm_len;
            end = i + cur_arm_len;
        }
    }

    string ans;
    for(int i=start; i<=end; i++){
        if( s[i]!="#"){
            ans += s[i];
        }
    }
    return ans;
}

int expand(const string& s, int left, int right){
    while(left>=0 && right<s.size() && s[left]==s[right]){
        --left;
        ++right;
    }
    return (right-left-2)/2;
}

```

超出时间限制
```cpp
// c++  s.substr(pos, n);
string longestPalindrome(string& s){
    int n = s.size();
    vector<vector<bool>> d(n, vector<bool>(n, false));
    string ans;
    
    for(int l=0; l<n; l++){
        
        for(int i=0; i+l<n; i++){
            int j = i+l;
            if(l==0){
                d[i][j] = true;
            }
            else if(l==1){
                d[i][j] = (s[i]==s[j]);
            }
            else{
                d[i][j] = (s[i]==s[j]) && d[i+1][j-1];
            }

            if( d[i][j] && l+1> ans.size()){
                ans = s.substr(i, l+1);
            }
        }
        return ans;
    }

}
```

42 接雨水
1) 双指针 time n， space 1
2) 栈，time n，space n
3) 动态编程，time n，space n
暴力解法  time n^2

```cpp

int trap(vector<int>& height){
    int l = 0;
    int r = height.size()-1;
    int max_l = 0;
    int max_r = 0;
    int ans =0;
    while(l < r){
        if(height[l] < height[r]){ // l is small
        
            if(height[l]>= max_l){  //update
                max_l = height[l];
            }
            else{  // water
                ans += max_l - height[l];
            }

            l++;
        }
        else{  // r is small
            if(height[r]>= max_r){  // upddate
                max_r = height[r];
            }
            else{  // water
                ans += max_r - height[r];
            }
            r--;
        }
    }
    return ans;
}


// stack overtime
int trap(vector<int>& height){
    int ans = 0;
    int cur = 0;
    stack<int> stk;

    while(cur< height.size()){
        
        while(!stk.empty() && height[cur]>height[stk.top()]){
            int a = stk.top();
            stk.pop();
            if(stk.empty()){
                break;
            }

            int dis = cur - stk.top() -1;
            int bound_h = min(height[cur], height[stk.top()]) - height[a];
            ans += dis * bound_h;
        }
    }
    return ans;
}

```



407 2D接雨水

1-D的接雨水问题有一种解法是从左右两边的边界往中间不断进行收缩，收缩的过程中，对每个坐标（一维坐标）能接的雨水进行求解。

2-D的接雨水问题的边界不再是线段的两个端点，而是矩形的一周，所以我们用优先队列维护所有边界点，收缩时，也不仅仅只有左右两个方向，而是上下左右四个方向，并且维护一个visit的数组，记录哪些坐标已经被访问过，不然会造成重复求解。

```cpp
Q<x,y,h>：优先级队列；
将地图的四周作为第一个边界，存入Q；
ans = 0;总储水量;
while(Q不空){
    <x,y,h> = Q弹出堆顶;
    for(<nx,ny> in <x,y>的上下左右){
        if(<nx,ny> 在图上 且 在边界内部){
            ans = ans + max(0,h - <nx,ny>的高度);
            新边界位置<nx,ny,max(h,<nx,ny>的高度)>入Q;
        }
    }
}

/*
    首先去最外面的边围起来
    然后按照最矮的边向里面扩缩
        比过来的边矮说明就可以积水，因为当前的这个边高度是当前这个包围圈的最矮高度
        将这个边放到队列中，取得高度应该是两者高度的最大值
*/

// 36 ms slow
typedef pair<int, pair<int, int>> PIII;
class Solution {
public:
    int trapRainWater(vector<vector<int>>& heightMap) {
        if(heightMap.empty())
            return 0;
        priority_queue<PIII, vector<PIII>, greater<PIII>> heap;
        int R = heightMap.size();
        int C = heightMap[0].size();
        vector<vector<int>> visited(R, vector<int>(C, 0));
        for(int r = 0; r < R; r++){
            for(int c = 0; c < C; c++){
                if(r == 0 || r == R - 1 || c == 0 || c == C - 1){
                    heap.push(make_pair(heightMap[r][c], make_pair(r, c)));
                    visited[r][c] = 1;
                    }
            }
        }
        int dx[5] = {1, 0, -1, 0, 1};
        int ans = 0;
        while(!heap.empty()){
            PIII iter = heap.top();
            heap.pop();
            for(int i = 0; i < 4; i++){
                int r = iter.second.first + dx[i];
                int c = iter.second.second + dx[i + 1];
                if(r >= 0 && r < R && c >= 0 && c < C && !visited[r][c]){
                    if(heightMap[r][c] < iter.first)
                        ans += iter.first - heightMap[r][c];
                    // 这里面的max很关键
                    heap.push(make_pair(max(iter.first, heightMap[r][c]), make_pair(r, c)));
                    visited[r][c] = 1;
                }
            }
        }
        return ans;
    }
};


```

并查集，time max(mn + V)
```cpp
// 36 ms
class Solution {
public:
    int fa[120 * 120], vis[120 * 120], sz[120 * 120];
    int n, m;
    int find(int x)
    {
        if(fa[x] == x ) return x;
        return fa[x] = find(fa[x]);
    }
    bool check(int x, int y)
    {
        if(x < 0 || x >= n || y < 0 || y >= m) return 0;
        return 1;
    }
    int trapRainWater(vector<vector<int>>& heightMap) {
        n = heightMap.size(), m = heightMap[0].size();
        for(int i = 0; i <= n * m; i++) fa[i] = i, sz[i] = 1;
        sz[n * m ] = 0;
        memset(vis, 0, sizeof vis);
        int V = 0;
        vector<int> pii[20010];
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++) 
                pii[heightMap[i][j]].push_back(i * m + j),
                V = V < heightMap[i][j] ? heightMap[i][j]: V;
        int cnt = 0, res = 0;
        int dir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for(int v = 0; v < V; v++)
        {
            for(int i = 0; i < pii[v].size(); i++)
            {
                cnt++;
                int x = pii[v][i] / m, y = pii[v][i] % m;
                vis[x * m + y] = 1;
                for(int k = 0; k < 4; k++)
                {
                    int dx = x + dir[k][0], dy = y + dir[k][1];
                    int flag = 0;
                    if(!check(dx, dy) || vis[dx * m + dy]) 
                    {
                        int fx, fy;
                        if(!check(dx, dy)) fx = find(n * m);
                        else fx = find(dx * m + dy);
                        fy = find(x * m + y);
                        if(fx == fy) continue;
                        sz[fx] += sz[fy];
                        fa[fy] = fx;
                    }
                }
            }
            // cout << cnt << endl;
            res += cnt - sz[find(n * m)];
        }
        return res;
    }
};

作者：zhe-ge-xian-qia-bu-tai-leng

```




11 盛水最多的容器
双指针，time n，space 1
```cpp
int maxArea(vector<int>& height){
    int l = 0;
    int r = height.size()-1;
    int ans = 0;
    while(l<r){
        int area = min(height[l], height[r]) * (r-l);
        ans = max(ans, area);

        if(height[l] <=height[r]){
            l++;
        }
        else{
            r--;
        }
    }
    return ans;
}

```


```cpp

````






k个一组翻转链表

```cpp
ListNode* reverseKGroup(ListNode* head, int k){
    ListNode dummy(0);
    dummy.next = head;

    ListNode* cur = &dummy;;
    ListNode* last = &dummy;;

    int cnt = 0;
    while( cur){
        cnt++;
        cur = cur->next;

        if(cnt==k && cur){  // reverse
            ListNode* bak = last->next;

            while(--cnt){
                ListNode* tmp = last->next;
                
                last->next = tmp->next;
                tmp->next = cur->next;
                cur->next = tmp;
            }
            cur = bak;
            last = bak;
        }

    }
    return dummy.next;
}

```


删除倒数第n 个节点
1) 双指针 time n, space 1
2) 用栈 time n, space n
   
```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    if(!head | !head->next) return nullptr;
    ListNode* p=head;// slow
    ListNode* q=head;// fast

    for(int i=0; i<n; i++){
        q = q->next;
    }
    if(!q){  // delete head
        return head->next;
    }
    
    while(q->next){
        p = p->next;
        q = q->next;
    }
    // delete raw p->next node. 
    p->next = p->next->next;
    return head;
}

```


爬楼梯
1) 动态规划，扫描，f(x) = f(x-1) + f(x-2), time n, space 1
2) 快速幂 time logn, space 1
3) 通项公式  time 1

调试没过

```cpp
int climbStairs(int n){
    int p=0;
    int q=0;
    int r=1;

    for(int i=1; i<=n; i++){
        p=q;
        q=r;
        r=p+q;
    }
    return r;
}

/*
*  |1 1 | |f(n)   | = | f(n) + f(n-1)  | = | f(n+1) |
*  |1 0 | |f(n-1) |   | f(n)           |   | f(n)   |
*
*  M = |1 1|
*      |1 0|
*
*/

// 老是报错，将 int 改为 long long就通过了。

int climbStairs(int n){
    vector<vector<int>> q = {{1,1},{1,0}};
    vector<vector<int>> res = pow(q, n);
    return res[0][0];
}

vector<vector<int>> pow(vector<vector<int>>& a, int n){
    vector<vector<int>> ans = {{1,0}, {0,1}};
    while( n>0){
        if( (n&1)==1 ){
            ans = multiply(ans, a);
        }
        n= n>>1;
        a = multiply(a, a);
    }
    return ans;
}

vector<vector<int>> multiply(vector<vector<int>>& a, vector<vector<int>>& b){
    vector<vector<int>> c(2, vector<int>(2,0));
    
    for(int i=0; i<2; i++){
        for(int j=0; j<2; j++){
            cout << i << "," << j << endl;
            c[i][j] = a[i][0] * b[0][j] + a[i][1]*b[1][j];
        }
    }
    cout << c[0][0] << ","<< c[0][1]<<"," << c[1][0] <<","<< c[1][1] << endl;
    return c;
}

```




```cpp
bool isValid(string s) {
    int n = s.size();
    if (n % 2 == 1) {
        return false;
    }

    unordered_map<char, char> pairs = {
        {')', '('},
        {']', '['},
        {'}', '{'}
    };
    stack<char> stk;

    for (char ch: s) {
        if (pairs.count(ch)) {
            if (stk.empty() || stk.top() != pairs[ch]) {
                return false;
            }
            stk.pop();
        }
        else {
            stk.push(ch);
        }
    }
    
    return stk.empty();
}


// notice stk.empty
bool isValid(string s){
    map<char, char> m_left = {{')', '('}, {']', '['}, {'}','{'}};
    stack<char> stk;
    set<char> s_left = {'(','[','{'};
    
    for(int i=0; i<s.size(); i++){
        if( s_left.find(s[i])!= s_left.end()){
            stk.push(s[i]);
        }
        else{
            if( m_left.find(s[i]) !=m_left.end() ){
                if( !stk.empty() && m_left[s[i]] == stk.top()){
                    stk.pop();
                }
                else{
                    return false;
                }
                    
            }
        }
    }
    return stk.empty();
}

```


搜索旋转数组
1) 二分查找,先找到最小值，再二分查找
2) 用二分以及 arr[0]做参考

优先处理好处理的情况。复杂的都放else
注意二分边界 l<=r

```cpp
int search(vector<int>& nums, int target){
    int l = 0;
    int r = nums.size()-1;

    while(l <= r){
        int mid = l + (r-l)/2;

        if(nums[mid]==target) return mid;
        if(nums[0]<= nums[mid]){
            if(nums[0] <=target && target < nums[mid]){ // ordered
                r = mid-1;
            }
            else{
                l = mid+1;
            }

        } 
        else{ // nums[mid] bigger than nums[0]
            if(nums[mid] < target && target <=nums[n-1] ){ // ordered
                l = mid+1;
            }
            else{
                r = mid-1;
            }
        }       
        
    }
    return -1;
}

```


买卖股票
1) 有顺序

```cpp
int maxProfit(vector<int>& prices){
    int n = prices.size();
    if(n<=1) return 0;

    int vmin = prices[0];
    int ans = 0;
    for(int i=1; i<n; i++){
        ans = max(ans, prices[i] - vmin);

        if(prices[i] < vmin){
            vmin = prices[i];
        }
    }
    return ans;
}

```


两数之和
1) 暴力，time n^2
2) 哈希表，time n

```cpp
vector<int> twoSum(vector<int>& nums, int target){
    map<int, int> a; // v,idx
    vector<int> res(2, -1);

    for(int i=0; i<nums.size(); i++){
        if( a.count(target-nums[i])>0){
            res[0] = a[target-nums[i]];
            res[1] = i;
            break;
        }
        a[nums[i]] = i;
    }
    return res;
}

```


三数之和
1）排序加双指针， nlogn + n*n = o(n^2)
如果重复数字太多，不适合用hash set,去重并不简单

```cpp
vector<vector<int>> threeSum(vector<int>& nums){
    vector<vector<int>> res;
    sort(nums.begin(), nums.end());

    int tmp = -1;
    for(int i=0; i<nums.size(); i++){
        if(nums[i]>0) return res; // no result

        // unique
        if(i>0 && nums[i]==nums[i-1]){
            continue;
        }

        int l = i+1;
        int r = nums.size()-1;
        // nums[i], nums[l], nums[r]
        while( r>l){
            tmp = nums[i] + nums[l] + nums[r];
            if(tmp>0)  r--;
            else if(tmp <0) l++;
            else{
                // 
                res.push_back(vector<int>{nums[i], nums[l], nums[r]});
                
                while(r>l && nums[r]==nums[r-1]) r--;
                while(r>l && nums[l]==nums[l+1]) l++;

                // find answer, next
                r--;
                l++;
            }
        }  
    }

    return res;
}


```

子序列问题除了动态规划没什么好办法。


115 不同的子序列
d[i][j] 表示s1的前i 个字符中，出现tj字符串组成的最多个数。

d[i][j] = d[i-1][j-1] + d[i-1][j];
d[i][j] = d[i-1][j];


注意溢出，用 long long 替代 int

```cpp
int numDistinct(string s, string t){
    int m = s.size();
    int n = t.size(); 
    vector<vector<int>> d(m+1, vector<int>(n+1, 0));
    
    for(int i=0; i<=m; i++){ // 0 row all zero.
        d[i][0] = 1;
    }

    for(int i=1; i<=m; i++){
        for(int j=1; j<=n; j++){

            if(j>i) continue;  // notice

            if(s[i-1]==t[j-1]){
                d[i][j] = d[i-1][j-1] + d[i-1][j];
            }
            else{
                d[i][j] = d[i-1][j];
            }
        }
    }
    return d[m][n];
}

```

不同的子序列
正则表达式匹配
编辑距离
通配符匹配
最长公共子序列






有序数组的平方
1）乘完直接排序，nlogn
2）双指针， n

```cpp
vector<int> sortedSquares(vector<int>& A){
    int n = A.size();
    vector<int> ans(n);
    
    int i=0;
    int j=n-1;
    int pos = n-1;
    while( i<=j){
        if( A[i]*A[i] > A[j]*A[j]){
            ans[pos] = A[i]*A[i];
            i++;
            pos--;
        }
        else{
            ans[pos]=A[j]*A[j];
            j--;
            pos--;
        }

    }
    return ans;
}

```


全排列
dfs,  time n*n!, space n

backtrack调用次数，调用次数为 n的k排列。\sum_{k=1}^{n} P(n,k) = n! + n!/1! + n!/2! + ... + n!/(n-1)! < 2n! + n!(1/2 + 1/4 + ...) < 3n!
backtrack调用次数是 O(n!)

对于每个backtrack的叶节点，我们需要把n个节点复制到答案中，相乘时间复杂度是 O(n*n!)

空间复杂度，递归调用深度是n，因此栈空间深度为n。

```cpp
vector<vector<int>> res;
vector<vector<int>> permute(vector<int>& nums){
    int n = nums.size();
    res.clear();
    vector<int> tmp;
    vector<bool> used(n, false);
    dfs(nums, used, tmp);
    return res;
}

void dfs(vector<int>& nums, vector<bool>& used, vector<int>& tmp){
    if(tmp.size()==nums.size()){
        res.push_back( tmp);
        return ;
    }
    
    for(int i=0; i<nums.size(); i++){
        if(used[i]==false){
            tmp.push_back( nums[i]);
            used[i] = true;
            dfs( nums, used, tmp);

            tmp.pop_back();
            used[i] = false;
        }
    }
}

```



```cpp

```


```cpp

```








