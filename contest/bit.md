
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

761 特殊的二进制序列

找山谷问题
对于栈 stack_0，遍历输入 S，每遇到一个'1'就入栈当前位置，每遇到一个'0'就出栈。

https://leetcode-cn.com/problems/special-binary-string/solution/c-0ms-63mb-fei-di-gui-lei-bi-yu-zhao-shan-gu-by-ch/

```cpp
string makeLargestSpecial(string s){

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

