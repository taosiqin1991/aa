

状态压缩dp
```cpp

```

```cpp

```

1547 切棍子的最小成本

每次切割的成本都是当前要切割的棍子的长度
// cuts=[1,3,4,5]
// 

```cpp
// dp
int minCost(int n, vector<int>& cuts){
    cuts.push_back(0);
    cuts.push_back(n);
    sort(cuts.begin(), cuts.end());

    int m = cuts.size();
    vector<vector<int>> dp(m, vector<int>(m, 0x3f3f3f3f));
    for(int i=0; i<m-1; i++){
        dp[i][i] =0;
        dp[i][i+1] =0;
    }
    dp[m-1][m-1]=0;
    for(int i=2; i<m; i++){ // i in [2, m-1]
        for(int j=0; j+i<m; j++){  // j+i < m
            for(int k=j; k<j+i; k++){ // j<= k < i+j
                dp[j][j+i] = min(dp[j][j+i], dp[j][k] +dp[k][j+i] + cuts[j+i]-cuts[j]);;
            }
        }
    }
    return dp[0][m-1];
}
```

1524 和为奇数的子数组数目

前缀和

```cpp
// 150 ms
int numOfSubarrays(vector<int>& arr) {
    const int mod = 1e9+7;
    int odd = 0, even = 1;
    int sub_arr = 0;
    int sum = 0;
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        sum += arr[i];
        sub_arr = (sub_arr + (sum % 2 == 0 ? odd : even)) % mod; // 
        if (sum % 2 == 0) {
            even++;
        } else {
            odd++;
        }
    }
    return sub_arr;
}

// 160 ms
int numOfSubarrays(vector<int>& arr){
    int n = arr.size();
    long long mod=1e9+7;
    long long res=0;
    int odd=0;
    int even=0;
    
    for(auto v: arr){
        if(v%2==0) even++;
        else{
            int t= odd;
            odd =even+1;
            even =t;
        }
        res = (res+odd)%mod;
    }
    return res;
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