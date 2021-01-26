

### 区间调度算法

最多不相交的区间

选择一个区间x, x.end是所有区间中结束最早的,end最小
把与x相交的区间都删除
重复上述两步,直到原集合为空.

故需要将end 升序排列. 排序time nlogn, 后续遍历n, 故time nlogn
```cpp
static void cmp_end(vector<int>& a, vector<Int>& b){
    return a[1] < b[1] || (a[1]==b[1] && a[0]<b[0]);
}

int intervalSchedule(vector<vector<int>>& intvs){
    if( intvs.size()==0) return 0;

    sort(intvs.begin(), intvs.end(), cmp_end);
    int cnt = 1;
    int x_end = intvs[0][1];
    for(auto& e: intvs){
        int start = e[0];
        
        if(start>= x_end){
            cnt++;
            x_end = e[1];
        }
    }
    return cnt;
}

```


435 无重叠区间
```cpp
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    int n = intervals.size();
    if(n <= 1)  return 0;

    auto myCmp = [&](const auto& a,const auto& b) {
        return a[1] < b[1];
    };
    sort(intervals.begin(),intervals.end(),myCmp);
    int cnt = 1;
    int end = intervals[0][1];  // 区间动态历史最小值
    for(const auto interval : intervals) {
        int start = interval[0];
        if(start >= end) {
            cnt++;
            end = interval[1];
        }
    }
    return n - cnt;
}

```

312 戳气球
动态规划，time n^3, space n^2

d[i][j] 表示戳破(i,j)之间的所有气球所获得的最高分, 不包括i，j。
d[i][j] = max(d[i][j], d[i][k]+d[k][j]+ vi*vk*vj);

```cpp
int maxCoins(vector<int>& nums){
    int n = nums.size();
    vector<vector<int>> rec(n+2, vector<int>(n+2));

    vector<int> v(n+2);
    v[0] = 1;
    v[n+1] = 1;
    for(int i=1; i<=n; i++){
        v[i] = nums[i-1];
    }

    for(int i=n-1; i>=0; i--){
        // v[i], v[k], v[j]
        for(int j=i+2; j<=n+1; j++){
            for(int k=i+1; k<j; k++){
                int sum = v[i] * v[j] * v[k];
                sum += rec[i][k] + rec[k][j];
                
                rec[i][j] = max( rec[i][j], sum);
            }
        }
    }
    return rec[0][n+1];
}

```

986 区间列表的交集
双指针解法 time m+n, space m+n

```cpp
// a,b all sorted
vector<vector<int>> intervalIntersection(vector<vector<int>>& A, vector<vector<int>>& B){
    int m = A.size();
    int n = B.size();
    int i=0;
    int j=0;
    vector<vector<int>> res;

    while(i<m && j<n){
        int l = max(A[i][0], B[j][0]);
        int r = min(A[i][1], B[j][1]);
        if(l<=r){
            res.push_back( {l, r});
        }
        
        if(A[i][1] < B[j][1]){ // small move_on
            i++;
        }
        else{
            j++;
        }
    }
    return res;
}

```

```cpp

```


```cpp
```



KMP算法

```cpp
模板
int x
for 0<=j<M:
    for 0<=c <256:
        if c==pat[j]:
            // state move_on
            d[j][c] = j+1
        else:
            // state reset
            // compute x and other to get reset_status
            d[j][c] = d[x][c]
    

class KMP{
private:
    vector<vector<int>> d;
    string pat;

public:
    KMP(string pat){
        this->pat = pat;
        // init d, need time O(m)
    }

    int search(string str){
        // match str
        // need N(n)

    }
}

// example
KMP kmp("ab");
int p1 = kmp.search("aaab"); // 3

```


```cpp
class KMP{
private:
    vector<vector<int>> d;
    string pat;

public:
    KMP(string pat){
        this->pat = pat;
        int m = pat.size();
        
        d = vector<vector<int>>(m, vector<int>(256, 0));
        d[0][ pat[0]] = 1;
        
        int x = 0;
        for(int j=1; j<M; j++){

            for(int c=0; c<256; c++){
                d[j][c] = d[x][c];
            }
            d[j][ pat[j]] = j+1;
            x = d[x][ pat[j]];  // update shadow x
        }
    }

    int search(string str){
        int m = pat.size();
        int n = str.size();
        
        int j=0;
        for(int i=0; i<n; i++){
            
            j = d[j][str[i]];
            // end_condition
            if( j==m)  return i-m+1;
        }
        return -1;
    }
}

```


459 重复的子字符串
1) 枚举，n^2
2) kmp, time n

```cpp
bool kmp(const string& query, const string& pat){
    int n = query.size();
    
    int m = pat.size();
    vector<int> fail(m, -1);
    for(int i=1; i<m; i++){  // pat
        
        int j = fail[i-1];
        while( j!=-1 && pat[j+1]!=pat[i]){
            j = fail[j];
        }
        if(pat[j+1]==pat[i]){
            fail[i] = j+1;
        }
    }

    int match = -1;
    for(int i=1; i<n-1; ++i){
        while( match!=-1 && pat[match+1]!=query[i]){
            match = fail[match];
        }
        if(pat[match+1]==query[i]){
            match++;
            if(match==m-1){
                return true;
            }
        }
    }
    return false;
}

bool repeatedSubstringPattern(string s){
    return kmp(s+s, s);
}

```

```cpp
bool repeatedSubstringPattern(string s){
    int n= s.size();
    for(int i=1; i*2<=n; i++){
        if(n%i==0){
            bool match = true;
            
            for(int j=i; j<n; j++){
                if(s[j] !=s[j-i]){
                    match = false;
                }
            }

            if(match) return true;
        }
    }
    return false;
}
```



855 调度考生的座位
time n, space n.

每当一个学生进入时，你需要最大化他和最近其他人的距离；如果有多个这样的座位，安排到他到索引最小的那个座位。


如果将每两个相邻的考生看做线段的两端点，新安排考生就是找最长的线段，然后让该考生在中间把这个线段「二分」，中点就是给他分配的座位。leave(p) 其实就是去除端点 p，使得相邻两个线段合并为一个。

但凡遇到在动态过程中取最值的要求，肯定要使用有序数据结构，
我们常用的数据结构就是二叉堆和平衡二叉搜索树了。二叉堆实现的优先级队列取最值的时间复杂度是 O(logN)，但是只能删除最大值。平衡二叉树也可以取最值，也可以修改、删除任意一个值，而且时间复杂度都是 O(logN)。
综上，二叉堆不能满足 leave 操作，应该使用平衡二叉树。



集合（Set）或者映射（Map），有的读者可能就想当然的认为是哈希集合（HashSet）或者哈希表（HashMap），这样理解是有点问题的。

因为哈希集合/映射底层是由哈希函数和数组实现的，特性是遍历无固定顺序，但是操作效率高，时间复杂度为 O(1)。
而集合/映射还可以依赖其他底层数据结构，常见的就是红黑树（一种平衡二叉搜索树），特性是自动维护其中元素的顺序，操作效率是 O(logN)。这种一般称为「有序集合/映射」。


```cpp
// 48ms
class ExamRoom{
private:
    int n;
    set<int> stds;

public:
    ExamRoom(int n){
        this->n = n;
    }

    int seat(){
        if(stds.size()==0){
            stds.insert(0);
            return 0;
        }
        
        int p = 0;
        int dist = *stds.begin();
        int pre = -1;

        for(const auto& s: stds){
            if(pre !=-1){ // from second time
                int mid_d = (s - pre)/2;
                if(mid_d> dist){
                    dist = mid_d;
                    p = pre + mid_d;  // result
                }
            }
            pre = s;
        }

        int last = *stds.rbegin();
        if(n-1 -last > dist){
            p = n-1;
        }

        stds.insert( p);
        return p;
    }

    void  leave(int p){
        stds.erase( p);
    }

};

```



5547 等差子数组

暴力，mnlogn。m次查询，子数字排列time不超过 nlogn

```cpp
vector<bool> checkArithmeticSubarrays(vector<int> &nums, vector<int> &l, vector<int> &r)
{
    vector<bool> ans;
    int queryLen = l.size();
    for (int i = 0; i < queryLen; i++)
    {
        int gap = r[i] - l[i] + 1;
        int tmp[gap];
        for (int j = l[i], k = 0; k < gap; j++, k++)
            tmp[k] = nums[j];
        sort(tmp, tmp + gap);
        bool flag;
        int diff = tmp[1] - tmp[0];
        flag = true;
        for (int j = 1; j < gap; j++)
            if (tmp[j] - tmp[j - 1] != diff)
                flag = false;
        ans.push_back(flag);
    }
    return ans;
}

```


```cpp


```

384 洗牌算法
time n

1) n的全排列是n！
第i位置选择的时候，被选择的区间在变小。
5*4*3*2*1

```cpp
class Solution{
private:
    vector<int> arr;
public:
    Solution(vector<int>& nums){
        arr = nums;
    }
    
    vector<int> reset(){
        return arr;
    }

    vector<int> shuffle(){
        int n = arr.size();
        vector<int> v_shuffle = arr;

        for(int i=0; i<n; i++){
            int j = rand_range(i, n);
            swap(v_shuffle[i], v_shuffle[j]);
        }
        return v_shuffle;
    }

    // [min, max)
    int rand_range(int min, int max){
        return (rand()% (max-min)) + min;
    }

};

```

870 优势洗牌
time nlogn，space n
用贪心来解。

1）以B数组为基准，从前往后查找；
先对A进行排序，然后在去A里面二分查找第一个大于B[i]的数，然后删除该数；如果找不到比B[i]大的数，就用最小的数去“赛马”。
2）参考标准解法，利用贪心思想，如果A中的最小值大于B的最小值，它们就可以对上；如果A的最小值不敌B的最小值，那么它就负责消灭B的最大值； 解的时候注意需要保存一个index和value的vector~


```cpp
vector<int> advantageCount(vector<int>& A, vector<int>& B){
    sort(A.begin(), A.end());
    
    vector<int> ans;
    for(auto& i: B){
        auto it = upper_bound(A.begin(), A.end(), i);
        if(it!= A.end()){
            ans.push_back( *it);
            A.erase(it);
        }
        else if(A.size() !=0){
            ans.push_back(A[0]);
            A.erase(A.begin());
        }
    }
    return ans;
}

```

颜色填充
dfs

```cpp
vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int new_col){
    int old_col = image[sr][sc];
    dfs(image, sr, sc, old_col, new_col);
    return image;
}

void dfs(vector<vector<int>>& image, int sr, int sc, int old_col, int new_col){
    if(sr<0 || sc<0 || sr>=image.size() || sc>=image[0].size()){
        return ;
    }

    if(image[sr][sc]==old_col && image[sr][sc]!=new_col){
        image[sr][sc] = new_col;
        
        dfs(image, sr+1, sc, old_col, new_col);
        dfs(image, sr-1, sc, old_col, new_col);
        dfs(image, sr, sc+1, old_col, new_col);
        dfs(image, sr, sc-1, old_col, new_col);

    }
}

```

974 和可被k整除的子数组

连续子数组问题，可以用前缀和来处理。


1）哈希表+ 逐一统计， time n, space min(n,k)


```cpp

int subarraysDivByK(vector<int>& A, int K){
    unordered_map<int, int> record = {{0, 1}};
    
    int sum = 0;
    int ans = 0;
    for(auto& a: A){
        sum += a;
    
        int modulus = (sum%K +K)%K;  // if sum <0        
        ++record[modulus];
    }

    for(auto& [x, cx]: record){
        // cout << cx << "," << (cx *(cx-1)/2)<< endl;
        ans += cx *(cx-1)/2;
    }
    return ans;
}


```

560 和为K的子数组

前缀和 + 哈希表，time n, space n
暴力是 time n^2, space 1

子数组的和为k，对应 pre[i]- pre[j-1] = k

pre[j-1] == pre[i]-k

以 i 结尾的和为 k 的连续子数组个数时只要统计有多少个前缀和为 pre[i]−k 的 pre[j] 即可。



```cpp
int subarraySum(vector<int>& nums, int k){
    unordered_map<int, int> mp;
    mp[0] = 1;

    int cnt = 0;
    int pre = 0;
    for(auto& x: nums){
        pre += x;
        
        if( mp.count(pre-k)){
            cnt += mp[pre-k];
        }
        mp[pre]++;
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

```cpp


```

```cpp


```


```cpp


```
