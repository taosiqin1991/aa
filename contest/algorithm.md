
## 基本框架

数组：连续存储，可以随机访问，节省空间。扩容需要先分配更大空间然后把原数据拷贝过去，相对有时间开销。插入删除中间数据时间开销大，是O(n)
链表：元素不连续，靠指针指向下一个元素位置，所以不存在扩容问题。插入删除中间元素快。因为存储空间不连续，无法根据索引直接计算得到对应元素的地址，所以不能随机访问。存储空间比数组大。

以下复杂数据结构都可通过数组或者链表来实现。

队列、栈：这两种数据结构都可以用链表或数组实现。用数组需要处理扩容缩容问题，用链表实现，需要比数组大一些的空间。

图：有两种表示方法。邻接表就是链表，邻接矩阵就是二维数组。
邻接表比较节省空间，但很多操作的效率上比不过邻接矩阵。
邻接矩阵判断连通性很快，并可以进行矩阵运算，比较方便。如果图比较稀疏则比较浪费空间。

散列表：
拉链法需要链表特性， 操作简单，需要额外的空间存指针。
线性探查法需要数组特性，便于连续寻址，不需要指针存储空间，但操作稍微复杂些。

树：用数组实现就是堆。因为堆是一个完全二叉树，用数组存储不需要节点指针。用链表实现的是常见的。

Redis数据库，提供列表、字符串、集合等常用等数据结构。对于每种数据结构，底层但存储方式都至少有两种，根据实际需求选择。


树的大多问题用递归来求解。



链表遍历框架，迭代或递归
二叉树遍历框架，非线性递归遍历。
回溯算法是N叉树的前后序遍历问题，如N皇后问题。

二分图，可以用 map<string, vector<string>>存储。
BFS广度优先来查找最短路径。
Bellman-Ford算法，可用于寻找负权重环。如w 变成 -ln(w) 如果前者w大于1，则后者小于0. 可以利用此寻找套汇机会。


滑动窗口：来解决最小覆盖子串、字符串中所有字母异位词、无重复字符的最长子串
双指针：判断链表是否有环、环的起始点、链表中点、链表的倒数第k个元素
左右指针：二分查找、两数之和、反转数组、滑动窗口算法
回溯：N皇后、全排列
动态规划：背包问题、高楼扔鸡蛋、最长公共子序列、
并查集Union-Find：解决图论中的动态连通性问题。

反转链表：需要三个指针同时移动p_cur, p_pre, p_next, 还需要一个指针记录NULL前的一个节点。共需要4个指针。





```cpp
// LinkedList 
struct ListNode{
    int val;
    ListNode* next;
} ListNode;

void traverse(ListNode* head){
    ListNode* p = head;
    while( p!=NULL){
        // todo
        p = p->next;
    }
}

void traverse(ListNode* head){
    traverse( head->next);
}

```

```cpp
// tree
struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(){val=0; left=right=NULL;}
};

void traverse(TreeNode* root){
    traverse( root->left);
    traverse( root->right);
}

struct NTreeNode{
    int val;
    vector<TreeNode*> children;
    
};

void traverse(NTreeNode* root){
    for(NTreeNode* child: root->children){
        traverse( child);
    }
}

```

用后序遍历求解二叉树最大路径和。LeetCode 124
```cpp
int ans = INT_MIN;
int one_side_max(TreeNode* root){
    if( root==NULL) return 0;
    int left = max(0, one_side_max(root->left));
    int right = max(0, one_side_max(root->right));
    
    ans = max(ans, left + right + root->val);
    return max(left, right) + root->val;
}
```

LeetCode105, 根据前序遍历和中序遍历的结果还原一颗二叉树。本质是前序遍历。
```cpp
TreeNode build_tree(vector<int>& pre_order, int pstart, int pend, vector<int>& in_order, int istart, int iend, map<int, int>& in_map){
    
}
```

LeetCode99题，恢复一颗BST树。根据中序遍历来确定。
```
```




递归

有时递归是高效的，有时是低效的。因为堆栈会消耗额外空间。

```
```

归并排序
```cpp
vector<int> tmp;

void sort(vector<int>& nums, int lo, int hi){
    if( lo>=hi) return;
    int mid = lo + (hi - lo)/2;
    sort(nums, lo, mid);
    sort(nums, mid+1, hi);  // [low, high]
    merge(nums, lo, mid, hi);
}

void merge(vector<int>& nums, int start, int mid, int end){
    int start1 = start;
    int end1 = mid;
    int start2 = mid+1;
    int end2 = end;

    int len = end - start + 1;
    tmp.resize(len);
    int i = start1;
    int j = start2;
    int k = 0;
    while( i<end1 && j<end2){
        tmp[k++] = nums[i] < nums[j] ? nums[i++] : nums[j++];
    }
    while( i<end1){
        tmp[k++] = nums[i++];
    }
    while( j<end2){
        tmp[k++] = nums[j++];
    }

    for(int k=0; k<len; k++) nums[k] = tmp[k];
}

```

```cpp
// 归并排序完整版本

class Solution{
public:

    // don't create vector in merge func. because it's often called.
    vector<int> res;

    void sort_interface(vector<int>& nums){
        res.resize(nums.size());
        sort(nums, 0, nums.size()-1);  // sort inplace.
    }


    void sort(vector<int>& nums, int lo, int hi){
        if(lo >= hi) return;
        int mid = lo + (hi - lo)/2;
        sort(nums, lo, mid);
        sort(nums, mid+1, hi);
        merge(nums, lo, mid, hi);
    }

    // handle [lo,mid), [mid, hi]
    void merge(vector<int>& nums, int lo, int mid, int hi){
        int i = lo;
        int j = mid+1;
        for(int k=lo; k<=hi; k++) res[k] = nums[k];
        
        for(int k=lo; k<=hi; k++){
            if( i>mid)          nums[k]=res[j++];    // todo
            else if( j>hi)      nums[k] = res[i++];
            else if( nums[j])   nums[k] = res[j++];
            else                nums[k] = res[i++];
        }   
    }



};

```

```cpp
void traverse(TreeNode* root){
    if( root==NULL) return;
    traverse( root->left);
    traverse( root->right);
}

void traverse_ntree(TreeNode* root){
    if( root==NULL) return ;
    for(auto child: root->children){
        traverse( child);
    }
}
```

给定二叉树和target值，节点上的值有正有负，返回树中和等于目标值的路径条数。
```cpp
int path_sum(TreeNode* root, int sum){
    if( root==NULL) return 0;
    
    int path_self_begin = count(root, sum);
    int path_left_sum = path_sum(root->left, sum);
    int path_right_sum = path_sum(root->right, sum);
    return path_left_sum + path_right_sum + path_self_begin;
}

int count(TreeNode* node, int sum){
    if( node==NULL) return 0;
    
    int is_me = (node->val == sum)? 1:0; // self
    
    int left_brother = count( node->left, sum-node->val);  // self + left
    int right_brother = count( node->right, sum - node->val); // self + right
    return is_me + left_brother + right_brother;
    
}

```


## 排序

### 快排
```
```


### 堆排
```
```


### 归并排序

```
```


## 二分搜索
```cpp

int binary_search(vector<int>& nums, int target){
    int left = 0;
    int right = nums.size()-1;
    
    while( left <=right){
        int mid = left + (right - left)/2;
        if( nums[mid]== target){
            return mid;
        }
        else if( nums[mid]<target){
            left = mid + 1;
        }
        else if( nums[mid] > target){
            right = mid -1;
        }
        return -1;
    }

}

```

## 双指针
链表有环的环起始位置、

```cpp
ListNode* detect_cycle(ListNode* head){
    ListNode* fast, slow;
    fast = head;
    slow = head;
    
    while( fast!=NULL && fast->next->NULL){
        fast = fast->next->next;
        slow = slow->next;
        if( fast==slow){
            break;
        }
    }
    
    slow = head;
    while( slow!= fast){
        fast = fast->next;
        slow = slow->next;
    }
    return slow;
}

```

## 滑动窗口
```cpp
int left = 0;
int right = 0;

while( right < s.size()){
    windows.add( s[right]);
    right++;

    while( valid){
        windows.remove( s[left]);
        left++;
    }
}

```

## DFS
```
```

## BFS
```
```

## 树的遍历
```
```

## 字符串问题
```
```




## 动态规划
```python
def dp(n):
    for coin in coins:
        dp(n - coin)

```

## 回溯
解决排列、子组合、子集问题。



```cpp
// todo  N queens.
void backtrack(vector<int>& nums, vector<>)
```


```python
result = []
def backtrack(path, choice_list):
    if end_conditon:
        result.add( path)
        return 
    
    for choice in choice_list:
        make_choice
        backtrack(path, choice_list)
        cancel_choice     
        
```


```cpp
vector<vector<int>> res;

vector<vector<int>> subsets(vector<int>& nums){
    // record paths 
    vector<int> track;
    backtrack(nums, 0, track);
    return res;
}

void backtrack(vector<int>& nums, int start, vector<int>& track){
    res.push_back( track);
    for(int i=start; i<nums.size(); i++){
        // make choice
        track.push_back( nums[i]);
        // backtrack
        backtrack(nums, i+1, track);
        // cancel choice
        track.pop_back();
    }
}

```

## 前缀和
统计班上同学成绩在不同分数段的百分比、

```cpp
int n = nums.size();
vector<int> pre_sum(n, 0);
for( int i=0; i<n; i++){
    pre_sum[i+1] = pre_sum[i] + nums[i];
}



```

```
```



## 等概率采样
```
```











