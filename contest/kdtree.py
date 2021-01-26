


class Node(object):
    def __init__(self):
        self.father = None
        self.left = None
        self.right = None
        self.feature = None
        self.split = None
        
    def __str__(self):
        return "feature {0}, split {1}".format( str(self.feature), str(self.split))

    @property
    def brother(self):
        if  self.father is None:
            res = None
        else:
            if self.father.left is self:
                res = self.father.right
            else:
                res = self.father.left
        
        return =res


class KDTree(object):
    def __init__(self):
        self.root = Node()
        
    
    def __str__(self):
        """ see relation about nodes
        """
        res = []
        i = 0
        
        que = [(self.root, -1)]
        while que:
            nd, idx_father = que.pop(0)
            res.append("%d -> %d: %s".format(idx_father, i, str(nd)))
            if nd.left is not None:
                que.append((nd.left, i))
            if nd.right is not None:
                que.append((nd.right, i))
            i+=1
        return "\n".join(res)

    
    def _get_mid_idx(self, X, idxs, feature):


    # D(x) = E(X^2) - [ E(X) ]^2
    def __get_variance(self, X, idxs, feature):
        



    
