import numpy as np
import time

#@profile
def as_arrays(X):
    for k in X:
        X[k] = np.array(X[k], dtype='float32')
    return X

def trunc_arrays(X):
    for k in X:
        if not k.endswith('.len'):
            X[k] = X[k][:X[k+'.len']]
    for k in X:
        if k.endswith('.len'):
            print k, X[k]
    return X

#@profile
def dict_append(X, x):
    if len(X) == 0:
        for k in x:
            X[k] = []
    for k in x:
        if type(x[k]) == list:
            assert False
            X[k] += x[k] 
        else:
            X[k].append(x[k])

def dict_concat(X, x):
    for k in x:
        if x[k].size == 0:
            continue
        if k not in X:
            shape = list(x[k].shape)
            shape[0] = 150000000
            X[k] = np.zeros((shape), dtype=x[k].dtype)
            X[k+'.len'] = 0
        assert X[k+'.len'] + x[k].shape[0] < 150000000
        X[k][X[k+'.len'] : X[k+'.len'] + x[k].shape[0]] = x[k]
        X[k+'.len'] += x[k].shape[0]  
        v = x[k] if type(x[k]) == list else [x[k]]
          
#class ArrDict:
#    def __init__(self, shape0):
#        self.shape0 = shape0
#        self.d = dict()
#        self.i = 0
#    
#    def append(self, x):
#        if len(self.d) == 0:
#            for k in x:
#                self.d[k] = np.zeros((self.shape0, x[k].shape[0]), dtype='float32')
#        for k in x:
#            self.
            
def dict_stack(X, Y):
    if len(X) == 0:
        for key in Y:
            X[key] = np.array([])
    for key in Y:
        X[key] = np.append(X[key], Y[key])

def load_dict(filename):
    zf = np.load(filename)
    d = dict()
    for f in zf.files:
        d[f] = zf[f]
    return d
    
class Timer:    
    def __enter__(self):
        self.start = time.time()
        print 'Begin...',
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        
class SoftAsserter:
    def __init__(self, name):
        self.name = name
        self.true_cnt = 0
        self.false_cnt = 0
        
    def soft_assert(self, predicate):
        if predicate:
            self.true_cnt += 1
        else:
            self.false_cnt += 1
            
    def report(self):
        print self.name, 'false_pct =', self.false_cnt,  '/ (', self.true_cnt, '+', self.false_cnt, ') =',\
             self.false_cnt * 1. / (self.true_cnt + self.false_cnt)