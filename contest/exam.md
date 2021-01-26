


### 排序
堆排序：
1）根据初始数组构造初始堆（构造完全二叉树，保证所有的父节点都比他的children数值大）
2）每次交换第一个和最后一个元素，输出最后一个元素（最大值），然后把剩下的元素重新调整为大根堆。


```cpp
void quick_sort(vector<int>& nums, int left, int right){
    if(left>=right) return ;

    int i = left;
    int j = right;
    int base = nums[left];
    while( i < j){

        while(nums[j]>=base && j>left){
            j--;
        }
        nums[i] = nums[j];
        
        while(nums[i]<=base && i<right){
            i++;
        }
        nums[j] = nums[i];
    }
    nums[i] = base;
    quick_sort(nums, left, i-1);
    quick_sort(nums, i+1, right);
}


void heap_sort(vector<int>& nums){
    int len = nums.size();
    // construct heap
    for(int i=len/2-1; i>=0; i--){
        heap_adjust(nums, i, len);
    }

    // n-1 loop for sort
    for(int i=len-1; i>0; i==){
        swap(nums[i], nums[0]);
        
        heap_adjust(nums, 0, i);
    }
    
}


// (sink) max_heap, find bigger num.
void heap_adjust(vector<int>& nums, int parent, int len){
    int tmp = nums[ parent];
    int child = 2* parent + 1;  // left child
    
    while( child<len){
        
        // if has right_child and right_child bigger
        if(child+1 <len && nums[child]<nums[child+1]) child++;
        
        if(tmp>= list[child]) break;

        nums[parent] = nums[child];
        parent = child;
        child = 2*parent + 1;
    }
    nums[parent] = tmp;

}

void swap(int& a, int& b){
    int tmp = a;
    a = b;
    b = tmp;
}

void merge_sort(vector<int>& nums, int left, int right){
    if( left>=right) return ;

    int mid = left + (right- left)/2;
    merge_sort(nums, left, mid);
    merge_sort(nums, mid+1, right);
    
    merge(nums, left, mid, right);
}

// [left, mid] [mid+1, right]
void merge(vector<int& nums, int left, int mid, int right){
    if(left >=mid) return ;
    
    int len = right - left+1;
    vector<int> tmp(len, 0);
    for(int k=0; k<len; k++) tmp[k] = nums[k+left];

    int i=left;
    int j = mid;
    int k=0;
        
    while(i<=mid && j<=right){

        if( nums[i] < nums[j]){
            tmp[k++] = nums[i++];
        }
        else{
            tmp[k++] = nums[j++];
        }
    }

    while(i<=mid) tmp[k++] = nums[i++];
    
    while(j<=right) tmp[k++] = nums[j++];
        
    

    for(int k=0; k<len; k++){
        nums[k+left] = tmp[k];
    }

}

int binary_search(vector<int>& nums, int val){
    
    int len = nums.size();
    int low = 0;
    int high = nums.size()-1;
    
    while( low <= high){
        int mid = low + (high - low)/2;

        if( nums[mid]==val){
            return mid;
        }
        else if( nums[mid]>val){
            low = mid+1;
        }
        else{
            high = mid -1;
        }   

    }
    return -1;
}


```


### 中序遍历

```cpp

class TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int v){val=v; left=right=NULL;}
}


vector<vector<TreeNode*>> level_traversal(TreeNode* root){
    vector<vector<TreeNode*>> res;
    if(root==NULL) return res;

    queue<TreeNode*> que;
    que.push(root);

    
    while( !que.empty()){
        vector<TreeNode*> tmp;

        int len = que.size();
        for(int i=0; i<len; i++){
            TreeNode* m = que.front();
            que.pop();
            
            if( m->left!=NULL) que.push(m->left);
            if( m->right!=NULL) que.push(m->right);
            tmp.push_back( m);
        }

        res.push_back(tmp);
    }

    return res;
}

```

反转链表

```cpp

class ListNode{
    int val;
    ListNode* next;
    ListNode(int v){val=v; next=NULL;}    

};


void inverse_list(ListNode* head){
    ListNode* p_cur = head;
    ListNode* p_pre = NULL;;
    ListNode* p_next = head->next;

    while( p_cur!=NULL){
        p_cur->next = p_pre;

        p_pre = p_cur;
        p_cur = p_next;
        p_next = p_next->next;

    }
    return p_pre;

}

 


```


最长公共子序列
```cpp
string find_longest_common_seq(string& a, string& b){
    int m = a.size();
    int n = b.size();

    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    
    for(int i=1; i<m+1; i++){
        for(int j=1; j<n+1; j++){

            if( a[i-1]==b[j-1]){
                dp[i][j] = dp[i-1][j-1] + 1;
            }
            else{
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }

        }
    }
    
    return dp[m][n];
    
}


子序列
```cpp
vector<vector<int>> res;

vector<vector<int>> subsets(vector<int>& nums){
    vector<int> track;
    backtrack(nums, 0, track);
    return res;

}

void backtrack(vector<int>& nums, int start, vector<int>& track){
    res.push_back( track);
    
    for(int i=start; i<nums.size(); i++){
        track.push_back( nums[i]);
        backtrack(nums, i+1, track);
        
        track.pop_back();
    }

}




```

组合
```cpp
vector<vector<int>> res;
// C_4^2
vector<vector<int>> combine(int n, int k){
    if( k<=0 || n<=0 ) return res;

    vector<int> track;
    backtrack(n, k, 1, track);
    return res;
}

void backtrack(int  n, int k, int start, vector<int>& track){
    // get to bottom of tree
    if( k==track.size() ){
        res.push_back( track);
        return ;
    }

    // from start.
    for(int i=start; i<=n; i++){
        track.push_back(i);
        
        backtrack(n, k, i+1, track);
        
        track.pop_back();
    }
}


class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        res.clear();

        vector<int> tmp;
        backtrack(n, k, 0, tmp);
        return res;

    }

    void backtrack(int n, int k, int start, vector<int>& tmp){
        if( tmp.size()==k){
            res.push_back( tmp);
            return ;
        }

        for(int i=1; i<=n; i++){
            tmp.push_back( i);

            backtrack(n, k, i+1, tmp);
            tmp.pop_back();
        }

    }

private:
    vector<vector<int>> res;
    
};



```


全排列
```cpp

class Solution{
public:

    vector<vector<int>> permute(vector<int>& nums){
        paths_.clear();
        
        vector<int> used(nums.size(), false);
        vector<int> path;

        permute_helper(nums, used, path);
        return paths_;
    }

private:
    vector<int> path_;
    vector<vector<int>> paths_;

    void permute_helper(vector<int>& nums, vector<int>& used, vector<int>& path){
        if( path.size()== nums.size()){
            paths_.push_back( path);
            return ;
        }

        for(int i=0; i<nums.size(; i++){
            if( used[i]) continue;

            used[i] = true;
            path.push_back( nums[i]);
            permute_helper(nums, used, path);

            path.pop_back();
            used[i] = false; 
        }

    }

}




```



LRU设计
1) 设计数据结构，接收capacity参数作为缓存的最大容量
2）实现put和get方法，时间复杂度要求 O(1)

LRU缓存算法的数据结构是哈希链表，即 双向链表 + 哈希表。
插入数据时，数据插入list的头部。


```cpp
class LRUCache{
public:

    LRUCache(int capacity): cap_(capacity){}
    
    int get(int key){
        if( map.find(key) == map.end()){
            return -1;
        }
        
        int val = map[key]->second;
        // put data front
        put(key, val);
        return val;
    }

    void put(int key, int value){
        pair<int, int> x = {key, value};
        
        if(map_.find(key) != map.end()){
            // delete old
            cache_.erase( map[key]);

            // new insert to front
            cache_.emplace_front(x);
            // update map.
            map[key] = cache_.begin();
        }
        else{
            if( cap_==cache_.size()){
                // delete the last
                pair<int, int> last = cache_.back();
                cache_.pop_back();
                map.erase( last.first);
            }
            
            // directly add
            cache_.emplace_front(x);
            map_[key] = cache_.bagin();
        }
    }



private:
    // key ->   iterator to pair(key, val) in the list
    unordered_map<int, list<pair<int, int>>::iterator > map_;
    
    list<pair<int, int>> cache_;
    int cap_;
    
}

```


寻找素数
```cpp
// the best, n* loglog(n)
// sieve of eratosthenes
int count_primes(int n){
    vector<bool> is_prime(n, true);
    
    for(int i=2; i*i <n; i++){
        if( is_prime[i]){

            for(int j=i*i; j<n; j=j+i){
                is_prime[j] = false;
            }
        }
    }

    int count =0;
    for(int i=2; i<n; i++){
        if( is_prime[i])  count++;
    }
    return count;
}

```


编辑距离
s1变成s2，最小的操作数
```cpp
// template
if s1[i]==s2[j]:
    do_nothing(skip)
    i, j all move
else:
    choose one:
        insert_
        delete_
        replace_
//

def min_distance(s1, s2):

    def dp(i, j):
        if i==-1: return j+1
        if j==-1: return i+1
        
        if s1[i]==s2[j]:
            return dp(i-1, j-1)
        else:
            return min( dp(i,j-1)+1,      // insert
                        dp(i-1,j) + 1,    // delete
                        dp(i-1, j-1)+1)   // inplace

    return dp(len(s1)-1, len(s2)-1)
    

class Solution{
public:
    int min_distance(string& s1, string& s2){
        int m = s1.size();
        int n = s2.size();

        vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
        
        // if s2 is empty. need clear s1 to equal s2.
        for(int i=1; i<=m; i++)  dp[i][0] = i;

        for(int j=1; j<=n; j++)  dp[0][j] = j;

        for(int i=1; i<=m; i++){
            for(int j=1; j<=n; j++){
                if( s1[i-1]== s2[j-1])  dp[i][j] = dp[i-1][j-1];
                
                else{

                    dp[i][j] = min(
                        dp[i-1][j] + 1,
                        dp[i][j-1] + 1,
                        dp[i-1][j-1] + 1
                    )
                }

            }
        }
        return dp[m][n];
        
    }

}

```

接雨水
1）用双指针解法
2）暴力解法 + 备忘录
water[i] = min( max(height,[0,...,i]), max(height[i,...,end])) - height[i]
左边最高的柱子，右边最高的柱子。


```cpp
int trap(vector<int>& height){
    if (height.size()==0 ) return 0;
    
    int res;
    int left = 0;
    int right = height.size()-1;
    
    int max_l = height[left];
    int max_r = height[right];
    
    while( left < right){
        max_l = max(max_l, height[left]);
        max_r = max(max_r, height[right]);
        
        if( max_l< max_r ){
            res += max_l - height[left];
            left++;
        }
        else{
            res += max_r -  height[right];
            right--;
        }
    }

    return res; 
}

```



去除有序数组的重复元素
1）数组的话，将要删除的数据换到末尾再删除。

```cpp
int remove_vec_duplicates(vector<int>& nums){
    int n = nums.size();
    if( n==0) return 0;
    
    int slow = 0;
    int fast = 1;

    while( fast<n){
        if( nums[slow]!= nums[fast]){
            slow++;            
            nums[slow] = nums[fast];  // good

            fast++;
        }
        else{
            fast++;
        }

    }
    return slow+1;
}

ListNode* rmove_list_duplicated(ListNode* head){
    if( head==NULL)  return NULL;
    
    ListNode* slow = head;
    ListNode* fast = head->next;
    while(fast!=NULL){
        // different
        if( fast->val != slow->val){
            slow->next = fast;

            show = slow->next;

        }
        
        fast = fast->next;
    }

    slow->next = NULL;
    return head;

}

```


最长回文子串
1) 动态规划，时间O(n^2), 空间O(n^2) dp_table
2) manacher's algorithm

寻找回文串是从中间向两端扩展，判断回文串是从两端向中间收缩。对于单链表，无法直接倒序遍历，可以造一条新的反转链表，可以利用链表的后序遍历，也可以用栈结构倒序处理单链表。

具体到回文链表的判断问题，由于回文的特殊性，可以不完全反转链表，而是仅仅反转部分链表，将空间复杂度降到 O(1)

```cpp
string longest_palindrome(string& s){
    string res;
    
    for(int i=0; i<s.size(); i++){
        string s1 = palindrome(s, i, i);
        
        string s2 = palindrome(s, i, i+1);

        res = res.size() > s1.size()? res: s1;
        res = res.size() > s2.size()? res: s2;
    }
    return res;
}

string palindrome(string& s, int l, int r){
    // index not overflow
    while( l>=0 && r<s.size() && s[l]==s[r]){
        // to left, to right
        l--;
        r++;
    }
    
    return s.substr(l+1, r-1 -1);
}


```

K个一组反转链表
1) 从head开始反转K个元素
2) 在k+1 个元素作为head递归调用上面。

给定这个链表：1->2->3->4->5
当 k = 2 时，应当返回: 2->1->4->3->5
当 k = 3 时，应当返回: 3->2->1->4->5


```cpp
ListNode* reverse_k_group(ListNode* head, int k){
    if( !head) return NULL;
    ListNode* begin = head;
    ListNode* end = head;

    for(int i=0; i<k; i++){
        if( !end){
            return head; // do nothing
        }
        end = end->next;
    }
    
    ListNode* new_head = reverse(begin, end); //  2, 4...
    // end ... begin
    begin->next = reverse_k_group(end, k);
    return new_head;

}

// [a,b)
ListNode* reverse(ListNode* begin, ListNode* end){
    ListNode* pre = NULL;
    ListNode* cur = begin;
    ListNode* next = begin->next;

    while(cur !=end){
        // reverse 
        cur->next = pre;

        pre = cur;
        cur = next;
        next = cur->next;
    }
    return pre;
}

```

判定括号合法性
[](){}
1）数量偶数，奇偶位置括号能对应上。

```cpp
bool is_one_valid(string& s){
    int left = 0;
    for(char c: str){
        if(c=='(') left++;
        else if(c==')')  left--;

        if(left<0) return false;
    }
    return left==0;

}

bool is_valid(string& str){
    stack<char> left;
    
    for(char c: str){
        if(c=='(' || c=='[' || c=='{'){
            left.push(c);
        }
        else if(c==')' || c==']' || c=='}' ){
            if( !left.empty()&& left_of(c)== left.top() )  left.pop();
            else  return false;
        }
    }
    return left.empty();
}

char left_of(char c){
    if(c==')') return '(';
    if(c==']') return '[';
    return '{';
}

```


寻找缺失和重复的元素
nums = [1,2,2,4]，算法返回 [2,3]

1）遍历一次数组，用hashmap记录每个数字出现的次数，然后遍历hashmap，看看哪个缺失哪个重复。O(N),时间复杂度和空间复杂度
2）时间复杂度没法降低
3）考虑降低空间复杂度

数组问题，关键点在于元素和索引是成对儿出现的，常用的方法是排序、异或、映射。

映射的思路就是我们刚才的分析，将每个索引和元素映射起来，通过正负号记录某个元素是否被映射。

排序的方法也很好理解，对于这个问题，可以想象如果元素都被从小到大排序，如果发现索引对应的元素如果不相符，就可以找到重复和缺失的元素。

异或运算也是常用的，因为异或性质 a ^ a = 0, a ^ 0 = a，如果将索引和元素同时异或，就可以消除成对儿的索引和元素，留下的就是重复或者缺失的元素。可以看看前文「寻找缺失元素」，介绍过这种方法。

```cpp
// return {dup, missing}
vector<int> find_error_nums(vector<int>& nums){
    

}

```


最短路径
dijkstra
Astar
bfs
动态规划


LeetCode64 最小路径和
```cpp

```

```cpp

```


Union-Find
解决图论中的动态连通性的。判断有几个连通的。

1) 写union和connected两个api
2) 将小的树接到大一些的树下面，使树深度不太深，并且更加平衡。
3) 
```cpp

class UF{
public:
    UF(int n);
    void union(int p, int q); // O(N) to  1
    int  find(int x);
    bool connected(int p, int q);

    int count();


private:
    int count_; // the left count
    vector<int> parent_;   // node x is parent[x]

}

///////  O(N)
UF::UF(int n){
    this->count_ = n;  // suppose all is alone.

    parent_.resize(n);
    for(int i=0; i<n; i++)  parent[i] = i;

}

void UF::union(int p, int q){
    int root_p = find(p);
    int root_q = find(q);
    if( root_p == root_q) return ;
    
    // combine two to one
    parent_[root_p ] = root_q;
    count_--;

}

int find(int x){
    while( parent_[x] != x){
        x = parent_[x];
    }
    return x;
}

bool UF::connected(int p, int q){
    int root_p = find(p);
    int root_q = find(q);

    return root_p == root_q;
}

int UF::count(){
    return count_;

}
/////////////////// O(1)
class UF{
private: 
    int count_;
    vector<int> parent_;
    vector<int> size_;

public:
    UF(int n){
        this->count_ = n;
        parent_.resize(n);
        size_.resize(n);
        
        for(int i=0; i<n; i++){
            parent_[i] = i;
            size[i] = 1;
        }
    }


    void union(int p, int q){

    }

    bool find(int p, int q){

    }

    int count(){
        return count_;
    }
    
    
}




```






