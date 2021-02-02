


692 前k 个 高频词
可以用堆排序来求解，或者用字典树TrieTree

1）排序好后是NlogN
2）堆排序是 Nlogk



```python
def topK_freq(self, words, k):
    count = collections.Counter( words)
    heap = [ (-freq,word) for word,freq in count.items()]
    
    heapq.heapify( heap)
    return [ heapq.heappop( heap)[1] for _ in range(k)]
```'




699 掉落的方块
seg_tree, time O(nlogn), space n
线段树能求解区间合并问题，多次询问的区间最长连续 上升序列问题，区间最大子段和问题等。

```cpp
class SegmentTree(object):
    def __init__(self, N, update_fn, query_fn):
        self.N = N
        self.H = 1
        while i<< self.H <N:
            self.M += 1

        self.update_fn = update_fn
        self.query_fn = query_fn
        self.tree = [0] *(2*n)
        self.lazy = [0] *N

    def _apply(self, x, val):
        self.tree[x] = self.update_fn(self.tree[x])
        if x < self.N:
            self.lazy[x] = self.update_fn(self.lazy[x], val)
        
    def _pull(self, x):
        while x>1:
            x/=2
            self.tree[x] = self.query_fn(self.tree[x*2], self.tree[x*2+1])
            self.tree[x] = self.update_fn(self.tree[x], self.lazy[x])
            
    def _push(self, x):
        for h in xrange(self.H, 0 , -1):
            y = x>>h
            if self.lazy[y]:
                self._apply( y*2, self.lazy[y])
                self._apply( y*2+1, self.lazy[y])
                self.lazy[y] = 0

    def update(self, L, R, h):
        L += self.N
        R += self.N
        L0, R0 = L, R
        while L <=R:
            if L & 1:
                self._apply(L, h)
                L += 1
            if R & 1==0:
                self._apply(R, h)
                R -=1
            L /=2;R /=2
        self._pull(L0)
        self._pull(R0)
        
    def query(self, L, R):
        L += self.N
        R += self.N
        self._push( L)
        self._push( R)
        ans = 0
        
        while L <=R:
            if L & 1:
                ans = self.query_fn( ans, self.tree[L])
                L += 1
            if R & 1==0:
                ans = self.query_fn( ans, self.tree[R])
                R -=1
            L /=2; R/=2
        return ans

class Solution(object):
    def falling_squares(self, pos):
        tree = Segment( len(index), max, max)
        best = 0
        ans = []

        for left,size in pos:
            L,R = index[left], index[left+ size-1]
            h = tree.query(L, R) + size
            tree.update(L, R, h)
            best = max( best, h)
            ans.append( best)

        return ans
            
```




745 前缀和后缀搜索
解决方案
1）成对的单词查找树
2）后缀修饰的单词查找树
两种time 都是 NK^2 + QK, space NK^2

```python
Trie = lambda: collections.defaultdict(Trie)
WEIGHT = False

class WordFilter(object):
    def __init__(self, words):
        self.trie = Trie()

        for weight,word in enumerate(words):
            cur = self.trie
            cur[WEIGHT] = weight
            for i,x in enumerate(word):
                # put all prefixs and suffixs

                tmp = cur
                for letter in word[i:]:
                    tmp = tmp[letter, None]
                    tmp[WEIGHT] = weight
                # advance letters
                cur = cur[x, word[~i]]
                cur[WEIGHT] = weight
                
    def search(self, prefix, suffix):
        cur = self.trie
        for a,b in map(None, prefix, suffix[::-1]):
            if (a,b) not in cur:
                return -1
            cur = cur[a, b]
        return cur[WEIGHT]

```


```python
```

```py
Trie = lambda: collections.defaultdict(Trie)
WEIGHT = False

class WordFilter(object):
    def __init__(self, words):
        self.trie = Trie()
        
        for weight,word in enumerate(words):
            word += "#"
            for i in range( len(word)):
                cur = self.trie
                cur[WEIGHT] = weight
                for j in range(i, 2*len(word)-1):
                    cur = cur[word[j % len(word)]]
                    cur[WEIGHT] = weight
    
    def f(self, prefix, suffix):
        cur = self.trie
        for letter in suffix + "#" + prefix:
            if letter not in cur:
                return -1
            cur = cur[letter]
        return cur[WEIGHT]

        
```


850 矩形面积

```python
global X
X = set()

class Node(object):
    def __init__(self, start, end, X):
        self.start = start
        self.end = end
        self.total = 0
        self.count = 0
        self._left = None
        self._right = None

    @property
    def mid(self):
        return self.start + (self.end-self.start)/2

    
    def left(self):
        self._left = self._left or Node(self.start, self.mid)
        return self._left
        
    def right(self):
        self._right = self._right or Node(self.mid, self.end)
        return self._right

    def update(self, i, j, val):
        if i>=j: return 0
        
        if self.start==i and self.end==j:
            self.count += val
        else:
            self.left.update(i, mid(self.mid, j), val)
            self.right.update( max(self.mid, i), j, val)
        
        if self.count >0:
            self.total = X[self.end] - X[self.start]
        else: 
            self.total = self.left.total + self.right.total
        
        return self.total

class Solution(object):
    def rectangle_area(self, rectangle):
        OPEN, CLOSE = 1, -1
        events = []

        for x1, y1, x2, y2 in rectangles:
            events.append( (y1, OPEN, x1, x2))
            events.append( (y2, CLOSE, x1, x2))
            X.append(x1)
            X.append(x2)
        events.sort()
        
        X = sorted(X)
        Xi = {x: i for i, x in enumerate(X)}
        
        active = Node(0, len(X)-1)
        ans = 0
        cur_x_sum = 0
        cur_y = events[0][0]
        
        for y, typ, x1, x2 in events:
            ans += cur_x_sum  * (y- cur_y)
            cur_x_sum = active.update( Xi[x1], Xi[x2], typ)
            cur_y = y

        return ans % (10**9 + 7)
    

```




