

def bfs(graph, start, end):
    queue = []
    visited = set()

    queue.append( [start])
    visited.add( start)
    
    while queue:
        node = queue.pop( 0)
        visited.add( node)

        process(node)
        node = generate_ralted_nodes(node)
        queue.apend( nodes)



visited = set()
def dfs(node, visited):
    visited.add( node)
    
    for next_node in node:
        if not next_node in visited:
            dfs( next_node, visited)



def dfs(tree):
    if tree.root is None:
        return []

    visited = []
    stack = [tree.root]

    while stack:
        node = stack.pop(-1)
        visited.add( node)
        
        process( node)
        nodes = generate_related_nodes(node)
        stack.append( nodes)


# dfs permutatation
def full_permuatation(arr):
    visited = [0] * len(arr)
    tmp = arr[:] # shadow copy
    result = []

    def dfs(pos):
        if pos==len(arr):
            result.append( tmp[:])
            return 
        
        for idx in range( len(arr)):
            if visited[idx]==0:
                tmp[pos] = arr[idx]

                visited[idx] = 1
                dfs(pos+1)
                visited[idx] = 0
    dfs(0)
    return result



# dfs visited = set()
def full_permutation(arr):
    visited = set()
    tmp = arr[:]
    
    result = []
    
    def dfs(pos):
        if pos==len(arr):
            result.append( tmp[:])
            return 

        for idx in range( len(arr)):
            if idx not in visited:
                tmp[pos] = arr[idx]
                
                visited.add( idx)
                dfs(pos+1)
                visited.remove( idx)

    dfs(0)
    return result


# backtrack, swap
def full_permutation(arr):
    result = []

    def backtrack(pos, end):
        if pos==end:
            result.append( arr[:])
            return 
        
        for idx in range( pos, end):
            arr[idx], arr[pos] = arr[pos], arr[idx]
            backtrack( pos+1, end)

            arr[pos], arr[idx] = arr[idx], arr[pos]
    
    backtrack( 0, len(arr))
    return result


# 生成有效括号组合，n=3，总共2n个元素，
# 1) 2n个元素，每个元素两种情况选择，先生成再判断是否有效，dfs 时间复杂度 2^(2n)
# 2) n个左括号和n个右括号，可以先全排列，再判断是否有效。比解法1快些，但也很慢。
# 3）

def generate_parenthesis(n):
    vec = []
    
    def gen(left, right, n, result):
        if left==n and right==n:
            vec.append( result)
            return 
        
        if left<n:
            gen(left+1, right, n, result +"(")
            
        if left>right and right <n:
            gen(left, right+1, n, result+")")
        
    gen(0,0, n, "")
    return vec


def test_parenthesis():
    print( generate_parenthesis(3))


def test():
    arr = [1, 2, 4]
    print( )


if __name__=="__main__":
    test_parenthesis()