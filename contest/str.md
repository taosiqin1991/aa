

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



最大子序和
1) 动态规划, f[i] = max(f[i-1]+ai, ai)
   time n, space 1
2) 分治，time n, 类似线段树求解 LCIS问题的 pushUp操作

二叉树深度为logn
遍历二叉树上所有节点，总时间 \sum_{i=1}^{logn} 2^(i-1) = n ，递归会使用logn的栈空间。

但是仔细观察「方法二」，它不仅可以解决区间 [0,n−1]，还可以用于解决任意的子区间 [l,r] 的问题。如果我们把 [0, n - 1][0,n−1] 分治下去出现的所有子区间的信息都用堆式存储的方式记忆化下来，即建成一颗真正的树之后，我们就可以在 O(logn) 的时间内求到任意区间内的答案，我们甚至可以修改序列中的值，做一些简单的维护，之后仍然可以在 O(logn) 的时间内求到任意区间内的答案，


```cpp
class Solution{
public:
    struct Status{
        int lsum;
        int rsum;
        int msum;
        int isum;
    };

    Status pushUp(Status l, Status r){
        int isum = l.isum + r.isum;
        
        int lsum = max(l.lsum, l.isum + r.lsum);
        int rsum = max(r.rsum, r.isum + l.rsum);
        
        int msum = max(max(l.msum, r.msum), l.rsum + r.lsum);
        return (Status){lsum, rsum, msum, isum};
    }

    Status get(vector<int>& a, int l, int r){
        if(l==r) return (Status){a[l], a[l], a[l], a[l]};

        int m = (l+r)>>1;
        Status lsub = get(a, l, m);
        Status rsub = get(a, m+1, r);
        return pushUp(lsub, rsub);
    }

    int maxSubArray(vector<int>& nums){
        return get(nums, 0, nums.size()-1).msum;
    }
    
};


```



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


LRU缓存机制
哈希链表


因此有哈希链表 = 双向链表 + 哈希表。
python中是 OrderedDict
Java中是  LinkedHashMap


LFU缓存
哈希表 + 平衡二叉树
双哈希表

在 C++ 语言中，我们可以直接使用 std::set 类作为平衡二叉树

有bug

```cpp
struct DLinkNode{
    int key;
    int val;

    DLinkNode* next;
    DLinkNode* pre;
    DLinkNode(int k, int v):key(k), val(v), pre(nullptr), next(nullptr){}
    DLinkNode(int k, int v, DLinkNode* pp, DLinkNode* np):key(k), val(v), pre(pp), next(np){}

};

class LRUCache{
private:
    DLinkNode* head;
    DLinkNode* tail;
    int cap;
    int cnt;
    unordered_map<int, DLinkNode*> mp;

public:
    LRUCache(int c){
        cap = c;
        cnt = 0;
        head = new DLinkNode(100, 100);
        tail = new DLinkNode(-1, -1);
        head->next = tail;
        tail->pre = head;
    }

    ~LRUCache(){
        mp.clear();
        DLinkNode* tmp = nullptr;
        while( head!=nullptr){
            tmp = head->next;
            delete head;
            head = head->next;
        }
        
    }

    void put(int k, int v){
        if(mp.count(k)){
            mp[k]->val = v;
            move_from_mid(mp[k]);
            move_to_head(mp[k]);
        }
        else{
            if(cnt==cap) delete_tail();
            else cnt++;

            add_head(k, v);
        }
    }

    int get(int k){
        if(mp.count(k)==0) return -1;
        
        move_from_mid(mp[k]);
        move_to_head(mp[k]);
        return mp[k]->val;
    }
    
    // head, x, head->next.
    void add_head(int& k, int& v){
        DLinkNode* l = head;
        DLinkNode* r = head->next;
        DLinkNode* mid = new DLinkNode(k, v, l, r);
        l->next = mid;
        r->pre = mid;
        mp[k] = mid;
    }

    // tail->pre, x, tail.
    void delete_tail(){
        DLinkNode* mid = tail->pre;
        DLinkNode* l = tail->pre->pre;
        DLinkNode* r = tail;

        l->next = r;
        r->pre = l;
        
        mp.erase(mid->key);
        delete mid;
    }

    void move_from_mid(DLinkNode* mid){
        DLinkNode* l = mid->pre;
        DLinkNode* r = mid->next;
        
        mid->pre = nullptr;
        mid->next = nullptr;
        l->next = r;
        r->next = l;
    }

    void move_to_head(DLinkNode* mid){
        DLinkNode* l = head->pre;
        DLinkNode* r = head;
        mid->pre = l;
        mid->next = r;
        l->next = mid;
        r->pre = mid;
    }

};

```

```cpp

class LRUCache {
public:
    struct node {
        int val;
        int key;
        node* pre;
        node* next;
        node(){}
        node(int key, int val):key(key), val(val), pre(NULL), next(NULL){}
    };

    LRUCache(int capacity) {
        this->capacity = capacity;
        head = new node();
        tail = new node();
        head->next = tail;
        tail->pre = head;
    }
    
    void inserttohead(node* cur)
    {
        node* next = head->next;
        head->next = cur;
        cur->pre = head;
        next->pre = cur;
        cur->next = next;
    }

    node* nodedelete(node* cur)
    {
        cur->pre->next = cur->next;
        cur->next->pre = cur->pre;
        return cur;
    }
    void movetohead(node* cur)
    {
        node* temp = nodedelete(cur);
        inserttohead(temp);
    }
    int get(int key) 
    {
        int ret = -1;
        if ( ump.count(key))
        {
            node* temp = ump[key];
            movetohead(temp);
            ret = temp->val;
        }
        return ret;
    }

    void put(int key, int value) {
        if ( ump.count(key))
        {
            node* temp = ump[key];
            temp->val = value;
            movetohead(temp);
        }
        else
        {
            node* cur = new node(key, value);
            if( ump.size()== capacity )
            {
                node *temp = nodedelete(tail->pre);
                ump.erase(temp->key);
            }
            inserttohead(cur);
            ump[key] = cur;

        }
    }
    unordered_map<int, node*> ump;
    int capacity;
    node* head, *tail;
};


```



get logn, put logn, 性能瓶颈。space cap.

```cpp
// hashtable + binary_tree
struct Node{
    int cnt;  // freq
    int time; 
    int key;
    int val;
    
    Node(int c, int t, int k, int v):cnt(c), time(t), key(k), val(v){}

    bool operator<(const Node& b) const{
        return cnt==b.cnt? time< b.time: cnt<b.cnt;
    }
};

class LFUCache{
private:
    int cap;
    int time;
    unordered_map<int, Node> key_table;
    set<Node> s;

public:
    LFUCache(int c){
        cap = c;
        time = 0;
        key_table.clear();
        s.clear();
    }

    int get(int key){
        if(cap==0) return -1;
        
        auto it = key_table.find(key);
        if(it==key_table.end()) return -1;
        // old
        Node cache= it->second;
        s.erase(cache);
        cache.cnt ++;
        cache.time = ++time;

        // new
        s.insert(cache);
        it->second = cache;
        return cache.val;
    }

    void put(int key, int val){
        if(cap==0) return ;
        
        auto it = key_table.find(key);
        if( it==key_table.end()){
            // create
            if(key_table.size()==cap){
                // delete first in tree
                key_table.erase( s.begin()->key);
                s.erase(s.begin());
            }

            Node cache = Node(1, ++time, key, val);
            
            key_table.insert( {key, cache});
            s.insert( cache);
        }
        else{ // modify old
            Node cache = it->second;
            s.erase( cache);
            cache.cnt +=1;
            cache.time = ++time;
            cache.val = val;
            
            s.insert(cache);
            it->second = cache;
        }
    }    
};

```

双哈希表实现LFU

```cpp
struct Node{
    int key;
    int val;
    int freq;
    Node(int k, int v, int f): key(k), val(v), freq(f){}
};

class LFUCache{
private:
    int min_freq;
    int cap;
    unordered_map<int, list<Node>::iterator> key_table;
    unordered_map<int, list<Node>> freq_table;

public:
    LFUCache(int c){
        min_freq = 0;
        cap = c;
        key_table.clear();
        freq_table.clear();
    }

    int get(int key){
        if(cap==0) return -1;
        
        auto it = key_table.find(key);
        if( it==key_table.end()) return -1;

        list<Node>::iterator node = it->second;
        int val = node->val;
        int freq = node->freq;

        freq_table[freq].erase(node);
        // delete if link is empty
        if(freq_table[freq].size()==0){
            freq_table.erase(freq);
            if(min_freq==freq) min_freq +=1;
        }

        // insert freq+1
        freq_table[freq+1].push_front( Node(key, val, freq+1));
        key_table[key] = freq_table[freq+1].begin();
        return val;

    }

    void put(int key, int val){
        if(cap==0) return ;
        
        auto it = key_table.find(key);
        if(it==key_table.end()){  // insert new
            if(key_table.size()==cap){
                // delete
                auto it2 = freq_table[min_freq].back();
                key_table.erase( it2.key);
                freq_table[min_freq].pop_back();

                if(freq_table[min_freq].size()==0){
                    freq_table.erase( min_freq);
                }
            }

            freq_table[1].push_back( Node(key, val, 1));
            key_table[key] = freq_table[1].begin();
            min_freq = 1;
        }
        else{
            list<Node>::iterator node = it->second;
            int freq = node->freq;
            freq_table[freq].erase(node);

            if(freq_table[freq].size()==0){
                freq_table.erase(freq);
                if(min_freq==freq) min_freq +=1;
            }
            freq_table[freq+1].push_front( Node(key, val, freq+1));
            key_table[key] = freq_table[freq+1].begin();
        }
    }
};

```


```cpp

```


```cpp

```








