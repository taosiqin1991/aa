



面试题 08.12 八皇后
dfs

位运算
每一个皇后对于后面的皇后，都相当于往左斜下方、右斜下方、正下方投下阴影。
这些阴影中，不能放置新的皇后。
左边的阴影区，每下移一行，就会左移一个位置，右边同理。
使用left,right,down来描述这三种“阴影”。它们的与是当前行的非法区域。

```cpp
// mysrio 4 ms
vector<vector<string>> res;
vector<vector<string>> solveNQueens(int n){
    vector<int> tmp(n);
    dfs(n, tmp, 0, 0, 0, 0);
    return res;
}

void dfs(int n, vector<int>& tmp, int l, int r, int d, int row){
    if(row==n){
        vector<string> result(n, string(n, '.'));
        for(int i=0; i<n; i++) result[i][tmp[i]]='Q';
        res.push_back( result);
        return ;
    }
    
    int invalid = l | r | d;
    for(int i=0; i<n; i++){
        if( ((1<<i)& invalid)==0 ){
            tmp[row] =i;  // ok
            dfs(n, tmp, 
            ((1<<i)| l)<<1,
            ((1<<i)| r)>>1,
            ((1<<i)| d), 
            row+1);
        }
    }
}

```

看上去time n^3
```cpp
// 4 ms
vector<vector<string>> res;  // all answer
vector<vector<string>> solveNQueens(int n){
    vector<string> board(n, string(n, '.'));
    dfs(board, 0);
    return res;
}

void dfs(vector<string>& board, int row){
    if( row == board.size()){
        res.push_back( board);
        return ;
    }

    int n = board[row].size();
    
    for(int col=0; col<n; col++){
        if( !is_valid( board, row, col)) continue;
        board[row][col] = 'Q';
        backtrack(board, row+1);
        board[row][col] = '.';
    }
}

// time n
bool is_valid(vector<string>& board, int row, int col){
    int n = board.size();
    // col conflict
    for(int i=0; i<n; i++){
        if( board[i][col]== 'Q'){
            return false;
        }
    }
    // right-up conflict
    for(int i=row-1, j=col+1; i>=0 && j<n; i--,j++){
        if( board[i][j]=='Q'){
            return false;
        }
    }
    // left-up conflict
    for(int i=row-1, j=col-1; i>=0 && j>=0; i--,j--){
        if( board[i][j]=='Q'){
            return false;
        }
    }
    return true;
}
```



看上去 time n^3
```cpp
// 8 ms
vector<vector<string>> res;
vector<vector<string>> solveNQueens(int n){
    vector<string> tmp;
    dfs(n, 0, tmp);
    return res;
}

void dfs(int n, int pos, vector<string>& tmp){
    if(pos>=n){
        res.push_back( tmp);
        return ;
    }
    
    vector<int> vis(n, 1);
    for(int i=0; i<tmp.size(); i++){
        int j=tmp[i].find("Q");  // tmp[i][j]
        vis[j]=0;
        
        if(j- (pos-i)>=0 ){
            vis[j- (pos-i)]=0;
        }
        if(j+ (pos-i)<n){
            vis[j +(pos-i)]=0;
        }
    }

    for(int i=0; i<n; i++){
        if(vis[i]==1){
            string cur_pos(n, '.');
            tmp.push_back( cur_pos.replace(i, 1, "Q"));
            dfs(n, pos+1, tmp);
            tmp.pop_back(); // 
        }
    }
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



```cpp

```



```cpp

```

