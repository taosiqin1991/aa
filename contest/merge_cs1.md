

493 翻转对
array a, i< j and ai> 2*aj, so (i, j) is a pair.
输入: [1,3,2,3,1]
输出: 2

输入: [2,4,3,5,1]
输出: 3

「327 题的题解：区间和的个数」中，我们介绍了归并排序、线段树、树状数组以及平衡搜索树等多种解法

归并排序
树状数组
都 nlogn, space n

```cpp
int res=0;
int reversePairs(vector<int>& nums){
    if(nums.size()==0) return 0;
    merge_sort(nums, 0, nums.size()-1);
    // for(auto& v: nums){
    //     cout << v << " ";
    // }
    return res;
}

void merge_sort(vector<int>& arr, int l, int r){
    if(l >= r) return ;
    int mid = l + (r-l)/2;
    merge_sort(arr, l, m);  // increasing
    merge_sort(arr, m+1, r); // increasing 
    merge(arr, l, r);
}


int merge(vector<int>& arr, int l, int r){
    int n1 = (m-l+1); //[l,m]
    int n2 = (r-m); // [m+1, r]
    
    int l
    return
}

```


树状数组

```cpp
// bueryt
// 256 ms
int reversePairs(vector<int>& nums){
    if( !nums.size()) return 0;
    int res=0;
    int n =nums.size();
    int k=1;
    vector<pair<long, int>> a;
    vector<int> b(n, 0);
    vector<long> c(1, LONG_MIN); //
    
    for(int i=0; i<n; i++){
        a.push_back( {nums[i], i});
    }
    sort(a.begin(), a.end());

    c.push_back( 2*a[0].first);
    for(int i=0; i<n; i++){
        if( i&& a[i].first !=a[i-1].first){
            b[a[i].second] =++k;
            c.push_back( 2*a[i].first);  // 
        }
        else b[a[i].second] =k;
    }
    
    vector<int> tree(k+1, 0);
    for(int i=n-1; i>=0; i--){
        int x =lower_bound(c.begin(), c.end(), nums[i]) -c.begin();
        res += find(x, tree);
        add(tree, b[i]);
    }
    return res;
}

int find(int x, vector<int>& tree){
    int p=0;
    for(int i=x-1; i>0; i-= (i&(-i)) ){
        p += tree[i];
    }
    return p;
}

void add(vector<int>& tree, int x){
    int l = tree.size();
    for(int i=x; i<l; i+=(i&(-i))){
        ++tree[i];
    }
}
```


合并两个有序链表
1) time O(m+n), space O(1) 迭代方式
2) time O(m+n), space O(m+n) 递归方式

递归调用 mergeTwoLists 函数时需要消耗栈空间，栈空间的大小取决于递归调用的深度。结束递归调用时 mergeTwoLists 函数最多调用 n+m 次，因此空间复杂度为 O(n+m)


```cpp
ListNode* mergetwoLists(ListNode* l1, ListNode* l2){
    ListNode prehead(-1);
    ListNode* pre = &prehead;

    while(l1!=nullptr && l2!=nullptr){
        if(l1->val < l2->val){
            pre->next = l1;
            l1 = l1->next;
            pre = pre->next;
        }
        else{
            pre->next = l2;
            l2 = l2->next;
            pre = pre->next;
        }

    }
    pre->next = l1==nullptr? l2: l1;
    return prehead.next;
}



ListNode* mergeTwoLists(ListNode* l1, ListNode* l2){
    if( l1==nullptr) return l2;
    else if(l2==nullptr) return l1;

    if(l1->val < l2->val){
        l1->next = mergeTwoLists(l1->next, l2);
        return l1;
    }
    else{
        l2->next = mergeTwoLists(l1, l2->next);
        return l2;
    }
}
```


合并k个有序的链表
1) 优先队列实现，time O(knlogk), space O(k)
   优先队列元素不超过k个，insert和delete时间为logk，最多有kn个点。对于每个点都插入删除各一次，故渐进时间复杂度为 knlogk

2）分治合并，time O(knlogk), space O(logk) 栈空间
3）顺序合并，time O(k^2 *n), space O(1)

故最好的办法是分治合并。

分治合并，第一轮合并k/2组链表，每一组时间代价是O(2n)
第二轮合并k/4组链表，每一组时间代价是O(4n)
总时间是 sum_(i=1)^(inf) k/(2^i) * 2^i * n = knlogk

```cpp
ListNode* mergeKLists(vector<ListNode*>& lists){
    return merge(lists, d0, lists.size()-1);
}

ListNode* merge(vector<ListNode*>& lists, int l, int r){
    if(l==r) return lists[l];
    if(l>r) return nullptr;

    int mid = l + (r-l)/2;
    ListNode* p1 = merge(lists, l, mid);
    ListNode* p2 = merge(lists, mid+1, r);
    return mergeTwoLists( p1, p2);
}

// bad case
ListNode* mergeKLists_order(vector<ListNode*>& lists){
    ListNode* ans = nullptr;
    for(int i=0; i<lists.size(); i++){
        ans = mergeTwoLists(ans, lists[i]);
    }
    return ans;
}

ListNode* mergeTwoLists(ListNode* 11, ListNode* l2){
    if(l1==nullptr) return l2;
    else if(l2==nullptr) return l1;

    ListNode* a = l1;
    ListNode* b = l2;
    ListNode tmp(INT_MIN);

    ListNode* p = &tmp;
    while(a && b){
        if(a->val < b->val){
            p->next = a;
            a = a->next;
            p = p->next;
        }
        else{
            p->next = b;
            b = b->next;
            p = p->next;
        }
    }
    p->next = a? a: b;
    return tmp.next;
}

```

优先队列合并，需要维护每个链表没有被合并的元素的最前面一个。
k个链表最多有k个这样的元素。

```cpp

// priority_queue<int, vector<int>, less<int> > max_heap;
// priority_queue<int, vector<int>, greator<int> > min_heap;

struct ListNode{
    int val;
    ListNode* next;
    ListNode(int v){val=v; next=nullptr;}
};

struct cmp{
    bool operator()(ListNode* a, ListNode* b){
        if(a.x == b.x) return a.y > b.y;
        return a.x > b.x;
    }
}

priority_queue<ListNode, vector<ListNode*>, cmp> p;


class Solution{
private:
    struct Status{
        int val;
        ListNode* ptr;
        bool operator<(const Status& a) const{
            return val > a.val;
        }
    };
    priority_queue<Status> q;

public:
    ListNode* mergeKLists(vector<ListNode*>& lists){
        for(auto node: lists){
            if(node)  q.push( {node->val, node});
        }
    
        ListNode tmp;
        ListNode* p = &tmp;

        while( !q.empty()){
            auto f = q.top();
            q.pop();

            p->next = f.ptr;
            p = p->next;

            if(f.ptr->next) q.push( {f.ptr->next->val, f.ptr->next});
        }
        return tmp.next;
    }
};
```





