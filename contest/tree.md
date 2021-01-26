


### 前序、中序、后序遍历

```cpp
// 先序遍历
void printPreorder1(TreeNode* head){
    if (head == nullptr){
        return;
    }
    cout << head->value << " ";
    printPreorder1(head->left);
    printPreorder1(head->right);
}

// 中序遍历
void printInorder1(TreeNode* head){
    if (head == nullptr){
        return;
    }
    printInorder1(head->left);
    cout << head->value << " ";
    printInorder1(head->right);
}

// 后序遍历
void printPostorder1(TreeNode* head){
    if (head == nullptr){
        return;
    }
    printPostorder1(head->left);
    printPostorder1(head->right);
    cout << head->value << " ";
}


```


```cpp
// 迭代版
void printPreorder2(TreeNode* head){
    cout << "Pre Order:" << endl;
    if (head != nullptr){
        stack<TreeNode*> *sta = new stack<TreeNode*>;
        sta->push(head);
        TreeNode* cur = head;
        while(!sta->empty()){
            cur = sta->top();
            sta->pop();
            cout << cur->value << " ";
            if (cur->right != nullptr){
                sta->push(cur->right);
            }
            if (cur->left != nullptr){
                sta->push(cur->left);     // 先压右边节点，再压左边节点，这与栈的特性有关
            }
        }
    }
    cout << endl;
}


void printInorder2(TreeNode* head){
     cout << "In Order:" << endl;
     if(head != nullptr){
         stack<TreeNode*>* sta = new stack<TreeNode*>;
         TreeNode* cur = head;
         while(!sta->empty() || cur != nullptr){
             if(cur != nullptr){
                sta->push(cur);
                cur = cur->left;
             }else{
                cur = sta->top();
                sta->pop();
                cout << cur->value << " ";
                cur = cur->right;
             }
         }
     }
     cout << endl;
}


void printPostorder2(TreeNode* head){
    cout << "Post Order:" << endl;
    if (head != nullptr){
        stack<TreeNode*>* sta1 = new stack<TreeNode*>;
        stack<TreeNode*>* sta2 = new stack<TreeNode*>;
        TreeNode* cur = head;
        sta1->push(cur);
        while(!sta1->empty()){
            cur = sta1->top();
            sta1->pop();      // 弹出的是最晚被压入栈的数据
            sta2->push(cur);
            if(cur->left != nullptr){
                sta1->push(cur->left);
            }
            if(cur->right != nullptr){
                sta1->push(cur->right);
            }
        }
        while(!sta2->empty()){
            cur = sta2->top();
            sta2->pop();
            cout << cur->value << " ";
        }
    }
    cout << endl;
}


```


### 二叉树 序列化与反序列化

二叉树的序列化本质上是对其值进行编码，更重要的是对其结构进行编码。可以遍历树来完成上述任务。众所周知，我们一般有两个策略：BFS / DFS。

BFS 可以按照层次的顺序从上到下遍历所有的节点
DFS 可以从一个根开始，一直延伸到某个叶，然后回到根，到达另一个分支。根据根节点、左节点和右节点之间的相对顺序，可以进一步将DFS策略区分为：
- 先序遍历
- 中序遍历
- 后序遍历



```py

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Codec:
    def serialize(self, root):
        res = []

        def preorder(root):
            if not root:
                res.append("#")
                return
            res.append( str(root.val))
            preorder(root.left)
            preorder(root.right)
        
        preorder(root)
        return ",".join(res)
    
    def deserialize(self, data):
        d = iter( data.split(","))
        
        def helper():
            tmp = next(d)
            if tmp=="#":
                return

            node = TreeNode(int(tmp))
            node.left = helper()
            node.left = helper()
            return node
        
        return helper()

```

```py
from collections import deque

class  Codec:
    def serialize(self, root):
        res_list = []
        if not root:
            return ""
        
        node_que = deque()
        node_que.append( root)
        while len(node_que)>0:
            node = node_que.popleft()
            
            if node==None:
                res_list.append("#")
            else:
                res_list.append( str(node.val))
                node_que.append( node.left)
                node_que.append( node.right)
        
        return " ".join(res_list)

    def deserialize(self, data):
        if not data:
            return None
        
        res_list = data.split(" ")
        num = len(res_list)
        if not num:
            return None
        
        root = TreeNode( res_list[0])
        node_que = deque()
        node_que.append( root)
        
        i=1
        while i<num:
            node = node_que.popleft()
            
            if res_list[i]=="#":
                node.left = None
            else:
                node.left = TreeNode( res_list[i])
                node_que.append( node.left)
            
            i += 1
            if res_list[i]=="#":
                node.right = None
            else:
                node.right = TreeNode(res_list[i])
                node_que.append(node.right)
            i+=1

        return root

```


```cpp
struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x){
        val = x;
        left = nullptr;
        right = nullptr;
    }
};

class Codec{
public:
    string serialize(TreeNode* root){
        queue<TreeNode*> node_que;
        string result;

        if(root==nullptr){
            return result;
        }
        node_que.push( root);
        while( !node_que.empty()){
            TreeNode* tmp = node_que.front();
            node_que.pop();

            if( tmp!=nullptr){
                result += to_string(tmp->val) + " ";
                node_que.push( tmp->left);
                node_que.push( tmp->right);
            }
            else{
                result += "# ";
            }
        }
        return result;
    }

    TreeNode* deserialize(string& data){
        if( data.empty()){
            return nullptr;
        }
        istringstream ss(data);
        string tmp;
        
        ss >> tmp;
        TreeNode* root = new TreeNode(stoi(tmp));
        queue<TreeNode*> node_que;
        node_que.push( root);

        while(ss >> tmp){
            TreeNode* node = node_que.front();
            node_que.pop();

            node->left = tmp=="#"? NULL: new TreeNode(stoi( tmp));
            ss >> tmp;

            node->right = tmp=="#"? NULL: new TreeNode(stoi( tmp));
            
            if(node->left){
                node_que.push( node->left);
            }
            if(node->right){
                node_que.push( node->right);
            }
        }
        return root;
    }

}

// Codec codec;
// codec.deserialize(codec.serialize(root));

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