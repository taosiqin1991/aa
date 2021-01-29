
810  黑板异或游戏

如果初始异或和就是 0, 先手直接赢了。
胜负其实和数组长度有关，如果长度为偶数，先手必胜，否则必败。

小红小明都想赢，一个数组中如果某个元素出现了偶数次，他们肯定是优先拿这种元素，这样就变为奇数次了，异或和就不为0了.

[1,1,2,3]

```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution{
public:
// win
bool xorGame(vector<int>& nums) {
    int a=0;
    for(auto x: nums) a^=x;
    return a==0 || !(nums.size()&1);
}

};

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

