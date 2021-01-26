



```cpp

void dfs(int x, int y){
    if( get_end || can_not_go){
        do_omething;
        return ;
    }
    
    if(x_next_can_go){
        mark_something;
        dfs(x+1, y);
        cancel_mark;
    }
    else if( y_next_can_go){
        mark_samething;
        dfs(x, y+1);
        cancel_mark;

    }
    
}

void dfs_stack(int start, int n){
    stack<int> s;
    
    for(int i=0; i<n, i++){
        if( next_can_go && point_not_marked){
            mark_point_visited;
            s.push(i); // push stack
        }
    }

    while( !s.empty() ){
        // do something s.top();
        // do something
        s.pop();

        for(int i=1; i<=n; i++){
            if( s_top_next_can_go && point_not_marked){
                mark_point_visited;
                s.push(i); // push stack
            }
        }

    }
}


void bfs(int state){
    queue<int> que;
    que.push(state);

    while( !que.empty() ){
        
        if( x_next_can_go) que.push(state + x);

        if( y_next_can_go) que.push(state + y);
        
        // handle que.front();
        // do something
        que.pop();
        
    }

}


```


二叉树

```cpp

// template
void bst(TreeNode* root, int target){
    if(root->val == target){
        // do something
    }
    
    if(root->val < target){
        bst(root->right, target);
    }
    if(root->val > target){
        bst(root->left, target);
    }
}





```


```cpp

````




