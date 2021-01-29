
420 强密码检验器
1259 强密码检验器
time n, space 1 的方案

1）下界是缺失的字符类型Nc
2）如果长度l 小于6，则补齐 6-l 个字符。则结果是 max(6-l, Nc)
3) if l<=20, 可以通过替换来去连续。长度为s的连续串，需要 s/3次替换。
4) l>20，则必须使用删除 l-20次。还需要去连续。  max(Nm, Nc) + (l-20)
5) 等等 3n+1, 3n+2

```cpp
int strongPasswordChecker(string s){
    bool has_digit= false;
    bool has_lower=false;
    bool has_upper=false;
    int len = 0;
    char c;

    int cnd_mod[3] = {0, 0, 0};
    int n_modify = 0;

    while( (c=s[len])!='\0'){
        if(c>='0' && c<='9'){
            has_digit = true;
        }
        else if(c>='a' && c<='z'){
            has_lower = true;
        }
        else if(c>='A' && c<='Z'){
            has_upper = true;
        }

        int i=len;
        
        while(s[++i]==c);
        int l = i - len; // repeated len
        
        if(l>=3){
            n_modify += l/3;
            cnt_mod[l%3]++;
        }
        
        len = i;
    }

    int n_miss_type = !has_digit + !has_upper + !has_lower;
    
    if(len<6)  return max(6-len, n_miss_type);

    if(len<=20)  return max(n_modify, n_miss_type);

    // two long
    int n_remove = len-20;
    if(n_remove < cnt_mod[0]){
        return max(n_modify - n_remove, n_miss_type) + len-20;
    }
    
    n_remove -= cnt_mode[0];
    n_modify -= cnt_mode[0];

    if(n_remove< cnt_mod[1] * 2){
        return max(n_modify - n_remove/2, n_miss_type) + len-20;
    }

    n_remove -= cnt_mod[1]*2;
    n_modify -= cnt_mod[1];

    return max(n_modify- n_remove/3, n_miss_type) + len-20;
}

```


465 最优账单平衡
NP难问题，可以用 暴力回溯求解

1、谁的剩余钱为 0，说明他已经平账了，可以不管他了；
2、谁的钱是正数，说明他需要还给别人多少钱；
3、谁的钱是负数，说明他需要等着别人还钱。

```cpp
int minTransfers(vector<vector<int>>& transactions){
    int res = INT_MAX;
    unordered_map<int, int> m;
    for(auto& t: transactions){
        m[ t[0]] -= t[2];
        m[ t[1]] += t[2];
    }

    vector<int> ac_cnt;
    for(auto& a: m){
        if(a.second !=0)  ac_cnt.push_back(a.second);

    }
    helper(ac_cnt, 0, 0, res);
    return res;
}

void helper(vector<int>& ac_cnt, int start, int cnt, int& res){
    int n = ac_cnt.size();
    while(start<n && ac_cnt[start]==0 ) start++;
    
    if(start==n){
        res = min(res, cnt);
        return ;
    }

    for(int i=start+1; i<n; i++){
        if( (ac_cnt[i]<0 && ac_cnt[start]>0) || (ac_cnt[i]>0 && ac_cnt[start]<0) ){
            ac_cnt[i] += ac_cnt[start];
            
            helper(ac_cnt, start+1, cnt+1, res);
            ac_cnt[i] -= ac_cnt[start];
        }
    }

}

```

完全二叉树的节点数
1）深度遍历，time n
2）左子树乘右子树的深度， time logn * logn
3）二分，time logn * logn




```cpp
// time logn * logn
int countNodes(TreeNode* root){
    if(root==nullptr) return 0;

    int d = count_depth(root);
    if(d==1) return 1;
    
    int l =1;
    int r=pow(2, d-1);
    int mid;
    while(l<=r){
        mid = l +(r-l)/2;
        
        if(check(mid, d, root)){
            l = mid+1;
        }
        else{
            r = mid-1;
        }
    }

    return pow(2, d-1) -1 + r;
    
}

int count_depth(TreeNode* root){
    int d = 0;
    while(root){
        root = root->left;
        d++;
    }
    return d;
}

bool check(int idx, int d, TreeNode* root){
    int l = 1;
    int r = pow(2, d-1);
    
    for(int i=0; i<d-1; i++){
        int pivot = l + (r-l)/2;

        if(idx <=pivot){
            r = pivot;
            root = root->left;
        }
        else{
            l = pivot + 1;
            root = root->right;
        }
    }
    return root;
}


```


```cpp
//time logn * logn
int countLevels(TreeNode* root){
    int levels = 0;
    while(root){
        root = root->left;
        levels += 1;
    }
    return levels;
}

int countNodes(TreeNode* root){
    if(root==nullptr) return 0;

    int l_level = countLevels(root->left);
    int r_level = countLevels(root->right);
    
    if(l_level == r_level){  // l_tree is full
        return countNodes(root->right) + (1<<l_level);
    }
    else{ // r_tree full
        return countNodes(root->left) + (1<<r_level);
    }
}

```

```cpp
// time n
void countNodes(TreeNode* root){
    if(!root) return 0;
    return countNodes(root->left) + countNodes(root->right) + 1;
}
```




1259 不相交的握手

d[i]表示i 个人的握手数。
d[n] = 2* d[n-2] + sum( d[k-2]*d[n-k] )

d[n] =\sum  d[2j-2] * d[n-2j],  j in [1,n/2]

```cpp
/* for(i=4; i<=n; i+=2)
*      d[i] += 2*d[i-2]
*      for(k=4; k<i-2; k+=2)
*           d[i] += d[k-2] * d[n-k]
*
*   return d[n]
*
*  (a + b) % mode = (a % mode + b % mode) % mode
*  (a * b) % mode = (a % mode * b % mode) % mode
*
*/

int numberOfWays(int num_people){
    int mod = 1e9 + 7;
    int n = num_people;
    vector<long long> d(n+1, 1);
    
    for(int i=2; i<=n; i+=2){
        d[i] = 0;
        
        for(int j=1; j<i; j+=2){
            d[i] = (d[i] + (d[j-1]*d[i-j-1]) %mod) %mod;
        }
    }
    return d[n];
}

```



```cpp

```

俄罗斯套娃信封问题
1）x升序，y降序。在y中寻找最长递增子序列，time nlogn.
排序和LIS 问题的time 都是 nlogn
LIS问题，  状态数n * 转移数 logn，故 time nlogn

```cpp
// 880 ms
static bool cmp(vector<int>& a, vector<int>& b){
    return a[0]<b[0] || (a[0]==b[0] && a[1]>b[1]);
}


int maxEnvelopes(vector<vector<int>>& envelopes){
    if( envelopes.size()==0) return 0;
    
    int n = envelopes.size();
    sort(envelopes.begin(), envelopes.end(), cmp);  // [&](auto& a, auto& b){return a[0]<b[0];}

    vector<int> h(n, 0);
    for(int i=0; i<n; i++){
        h[i] = envelopes[i][1];
    }
    return len_of_LIS(h);
}

int len_of_LIS(vector<int>& nums){
    int n = nums.size();
    int res = 0;
    vector<int> d(n+1, 1);

    for(int i=1; i<=n; i++){
        
        for(int j=1; j<i; j++){
            if(nums[i-1] > nums[j-1]){
                d[i] = max(d[i], d[j]+1);
            }
        }
        res = max(res, d[i]);
    }
    return res;
}


```


327 区间和的个数
1）暴力求解，n^2
2) 用线段树或者树状数组，nlogn

lower <= pre_sum - x <= upper
so,  pre_sum - upper <= x <= pre_sum-lower

```cpp
// average_tree  nlogn, space n
int countRangeSum(vector<int>& nums, int lower, int upper){
    int n = nums.size();
    int64_t pre_sum = 0;
    multiset<int64_t> s;
    s.insert(0);

    int res =0;
    for(int i=0; i<n; i++){
        pre_sum += nums[i];
        
        res += distance(s.lower_bound(pre_sum - upper), s.upper_bound(pre_sum-lower));
        s.insert( pre_sum);
    }

}

```


```cpp
// merge sort, nlogn
int countRangeSum(vector<int>& nums, int lower, int upper){
    int n = nums.size();
    vector<int64_t> s(n+1, 0);
    vector<int64_t> assist(n+1, 0);
    for(int i=1; i<=n; i++) s[i] = s[i-1] + nums[i-1];

    return merge(s, assist, 0, n, lower, upper);
}

int merge(vector<int64_t>& s, vector<int64_t>& assist, int l, int r, int lo, int hi){
    if(L>=R) return 0;

    int cnt = 0;
    int m = l+ (r-l)/2;
    cnt += merge(s, assist, l, m, lo, hi);
    cnt += merge(s, assist, m+1, r, lo, hi);
    
    int left = l;
    int lower = m+1;
    int upper = m+1;
    while(left <=m){
        
        while(lower<=R && s[lower]-s[left]<lo) lower++;
        
        while(upper<=R && s[upper]-s[left]<=hi ) upper++;

        cnt+= (upper-lower);
        left++;
    }

    // merge process
    left = l;
    int right = m+1;
    int pos = l;
    while(left<=m || right<=r){
        if(left>m)  assist[pos] = s[right++];

        if(right>r && left<=m ) assist[pos]=s[left++];
        
        if(left<=m && right<=r){
            if(s[left]<=s[right])  assist[pos]=s[left++];
            else assist[pos]=s[right++];
        }
        pos++;
    }
    for(int i=l; i<=r; i++) s[i] = assist[i];
    return cnt;
}
```


权值线段树

```cpp
//
private:
    vector<int> val;
    void add(int v, int o, int l, int r){
        if(l==r) val[o]++;
        else{
            int m = l + (r-l)/2;
            if(v<=m)  add(v, 2*o, l, m);
            else   add(v, 2*o+1, m+1, r);
            val[o] = val[2*o] + val[2*o+1];

        }
    }

    int query(int ql, int qr, int o, int l, int r){
        if(ql<=l && qr>=r) return val[o];

        int ans = 0;
        int m = l +(r-l)/2;
        if(ql<=m) ans += query(ql, qr, 2*o, l, m);
        if(qr>m)  ans += query(ql, qr, 2*o+1, m+1, r);
        return ans;
    }

public:
int countRangeSum(vector<int>& nums, int lower, int upper){
    int n = nums.size();
    int res = 0;
    vector<int64_t> s(n+1, 0);
    for(int i=1; i<=n; i++) s[i] = s[i-1] + nums[i-1];

    sort(s.begin(), s.end());
    s.erase( unique(s.begin(), s.end()), s.end());

    int len = s.size();
    val = vector<int>(4*(len+1), 0);
    int64_t pre_sum =0;

    int lo = lower_bound(s.begin(), s.end(), pre_sum) - s.begin()+1;
    // int up = upper_bound(s.begin(), s.end(), pre_sum) - s.begin();
    
    add(lo, 1, 1, len);
    for(int i=0; i<n; i++){
        pre_sum += nums[i];
        int high = lower_bound(s.begin(), s.end(), pre_sum-upper) - s.begin()+1;
        int low = upper_bound(s.begin(), s.end(), pre_sum-lower) - s.begin();
        if(high<=low){
            res += query(high, low, 1, 1, len);
            int new_low = lower_bound(s.begin(), s.end(), pre_sum)- s.begin()+1;
            add(new_low, 1, 1, len);
        }

    }
    return res;
    
}

```



```cpp

```

```cpp

```


