import numpy as np
import time

#@profile
def as_arrays(X):
    for k in X:
        X[k] = np.array(X[k], dtype='float32')
    return X

def trunc_arrays(X, keys=None):
    if keys is None:
        keys = X.keys()
    for k in keys:
        if k+'.len' in X:
            X[k] = X[k][:X[k+'.len']]
            assert k+'.starts' in X and k+'.stops' in X
            X[k+'.starts'] = np.array(X[k+'.starts'], dtype='int32')
            X[k+'.stops'] = np.array(X[k+'.stops'], dtype='int32')
            print k+'.len', '=', X[k+'.len']
    return X

def free_arrays(X):
    for k in X:
        del X[k]

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

g_buffers = dict()

def downcast_dtype(key, dtype):
    if dtype == np.float64:
        print 'Downcasting', key, 'float64 -> float32'
        dtype = np.float32
    elif dtype == np.int64:
        print 'Downcasting', key, 'int64 -> int32'
        dtype = np.int32
    return dtype

def dict_concat(X, x):
    max_len = 120000000
    for k in x:
        if k not in X:
            shape = list(x[k].shape)
            shape[0] = max_len
            if k not in g_buffers:
                dtype = downcast_dtype(k, x[k].dtype)
                print 'Allocating', shape, dtype, 'for', k
                g_buffers[k] = np.zeros((shape), dtype=x[k].dtype)
            X[k] = g_buffers[k]
            X[k+'.len'] = 0
            X[k+'.starts'] = []
            X[k+'.stops'] = []
        start = X[k+'.len']
        stop = X[k+'.len'] + len(x[k])
        assert stop <= max_len
        X[k][start : stop] = x[k]
        X[k+'.len'] = stop
        X[k+'.starts'].append(start)
        X[k+'.stops'].append(stop)
        
def dict_concat_old(X, x):
    max_len = 150000000
    for k in x:
        if k+'.len' not in X:
            X[k+'.len'] = 0
            X[k+'.starts'] = []
            X[k+'.stops'] = []
        if k not in X and len(x[k]) > 0:
            shape = list(x[k].shape)
            shape[0] = max_len
            X[k] = np.zeros((shape), dtype=x[k].dtype)
        start = X[k+'.len']
        stop = X[k+'.len'] + len(x[k])
        assert stop <= max_len
        if k in X and len(x[k]) > 0:
            X[k][start : stop] = x[k]
        X[k+'.len'] = stop
        X[k+'.starts'].append(start)
        X[k+'.stops'].append(stop)
        
def dict_concat_old(X, x):
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
#        print 'Begin...',
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