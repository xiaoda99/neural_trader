import os
import glob
import time
import datetime
import ctypes
import numpy as np
import math
import gzip
import cPickle
import pylab as plt
import matplotlib.gridspec as gridspec
from profilehooks import profile
import csv

try: 
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict
    
base_dir = '/home/xd/data/trading'

def get_day_dir(exchange, year, month, day):
    assert year == 2015
    if year == 2015: 
        return '%s/f_c%d/f_c%d%02dd/%s/%d%02d%02d' % (base_dir, year, year, month, exchange, year, month, day)
    else:
        return None
        
def find_main_contract_path(day_dir, commodity):
    max_size = 0
    main_contract_path = None
    for contract_path in glob.glob('%s/%s*' % (day_dir, commodity)):
        size = os.path.getsize(contract_path)
        if size > max_size:
            main_contract_path = contract_path
            max_size = size
#    print main_contract_path, max_size
    return main_contract_path

def get_paths(exchange, commodity, year, months):
    paths = []       
    for month in months:     
        for day in range(1, 32):
            day_dir = get_day_dir(exchange, year, month, day)
#            print day_dir
            if os.path.isdir(day_dir):
                mc_path = find_main_contract_path(day_dir, commodity)
                if mc_path is not None:
                    paths.append(mc_path)
    return paths
            
def load_ticks(exchange, commodity, year, months, use_cache=True):
    fname = '%s/ticks_%s%d%02d.pkl' % (base_dir, commodity, year % 1000, months[0])
    if use_cache and os.path.isfile(fname):
        ticks = OrderedDict()
        for month in months:
            fname = '%s/ticks_%s%d%02d.pkl' % (base_dir, commodity, year % 1000, month)
            print 'Loading', fname, '...'
            with open(fname) as f:
                dict_stack(ticks, cPickle.load(f))
            print 'Done.'
        return ticks
        
    paths = get_paths(exchange, commodity, year, months)
#    print paths
    ticks = OrderedDict()
    for path in paths:
        with open(path) as f:
            reader = csv.reader(f)
            prev = None
            prev_tick = None
            n_ticks = 0
            reader.next()
            ticks_today = OrderedDict()
            for data in reader:
                now = datetime.datetime.strptime(data[2], '%Y-%m-%d %H:%M:%S.%f')
                if prev is None and not (now.time().minute == 0 and now.time().second in [0, 1]):
                    continue
                if now.time().hour == 15 and (now.time().minute > 0 or now.time().second > 0):
                    break
                if 0 < now.time().microsecond < 500000:
                    now = now.replace(microsecond=0)
                if 500000 < now.time().microsecond < 1000000:
                    now = now.replace(microsecond=500000) 
                tick = make_tick2(data)
                if prev is not None:
                    if now < prev:
                        continue
                    if now == prev:
                        if prev.time().microsecond == 0:
                            now = now + datetime.timedelta(microseconds=500000)
                        else:
                            assert prev.time().microsecond == 500000
                            continue
                    dt = now - prev
                    if dt.microseconds not in [0, 500000]:
                        print now, 'dt.microseconds =', dt.microseconds
                        break
                    
                    if dt.seconds >= 60 * 14:
                        n_missed = ((dt.seconds % (60 * 15))* 1000000 + dt.microseconds) / 500000
                        n_missed = 0
                    else:
                        n_missed = (dt.seconds * 1000000 + dt.microseconds) / 500000 - 1
                    if n_missed > 100:
                        print now, 'n_missed =', n_missed, 'dt.seconds =', dt.seconds
                    for _ in range(n_missed):
                        prev_tick['time_in_ticks'] = n_ticks  # fix this bug in 20160126
                        dict_append(ticks, prev_tick)
#                        dict_append(ticks_today, tick)
                        n_ticks += 1
                if tick['ask_price'] == 0 or tick['bid_price'] == 0:
#                    print now, 'ask_price =', tick['ask_price'], 'bid_price =', tick['bid_price'], tick['ask_vol'], tick['bid_vol']
                    if not tick['last_price'] > 0:
                        print 'last_price =', tick['last_price']
                        assert False
                    if tick['ask_price'] == 0:
                        tick['ask_price'] = tick['last_price']
                    if tick['bid_price'] == 0:
                        tick['bid_price'] = tick['last_price']
                tick['price'] = (tick['ask_price'] + tick['bid_price']) * 1. / 2
                if tick['ask_vol'] == 0:
                    tick['ask_vol'] = tick['bid_vol']
                if tick['bid_vol'] == 0:
                    tick['bid_vol'] = tick['ask_vol']
                tick['time_in_ticks'] = n_ticks
                if tick['time_in_ticks'] == 0 and ticks.has_key('last_price') and abs(tick['last_price'] - ticks['last_price'][-1]) > 400:
                    print '*******************************************big jump!', path
                dict_append(ticks, tick)
#                dict_append(ticks_today, tick)
                n_ticks += 1
                prev = now
                prev_tick = tick
            print path, n_ticks
#            plt.plot(ticks_today['last_price'])
#            plt.show()
    return as_arrays(ticks)

def make_tick2(data):
    t = {}
    d = data  
    t['last_price'] = int(float(d[3]))
    t['pos'] = int(d[4])
    t['pos_inc'] = int(d[5])
    t['vol'] = int(float(d[7]))
    t['open_vol'] = int(d[8])
    t['close_vol'] = int(d[9])
    assert d[11] in ['B', 'S']
    t['direction'] = 1 if d[11] == 'B' else -1
    
    t['ask_price'] = int(round(float(d[-3])))
    t['bid_price'] = int(round(float(d[-4])))
    t['ask_vol'] = int(d[-1])
    t['bid_vol'] = int(d[-2])
#    t['price'] = (float(d[-3]) + float(d[-4])) * 1. / 2
    return t
                      
def make_tick(data):
    t = {}
    d = data  
    t['ask_price'] = d[2]
    t['bid_price'] = d[18]
    t['price'] = (t['ask_price'] + t['bid_price']) * 1. / 2
    t['vol'] = sum(d[41:41+7])
    
    t['ask_vol'] = d[10]
    t['bid_vol'] = d[26]
    return t

#@profile
def as_arrays(X):
    for k in X:
        X[k] = np.array(X[k], dtype='float32')
    return X

#@profile
def dict_append(X, x):
    if len(X) == 0:
        for key in x:
            X[key] = []
    for key in x:
        X[key].append(x[key])
  
def dict_stack(X, Y):
    if len(X) == 0:
        for key in Y:
            X[key] = np.array([])
    for key in Y:
        X[key] = np.append(X[key], Y[key])

futures = [
##('dc', 'pp', 1, 3.75),
('dc', 'l', 5, 3.75),
('zc', 'MA', 1, 6.25),
('zc', 'TA', 2, 6.25),

('zc', 'SR', 1, 6.25), # tang, 5000
('dc', 'm', 1, 6.25), # doupo, 2000
('zc', 'RM', 1, 6.25), # caipo, 2000
('dc', 'y', 2, 6.25), # douyou, 5000
('dc', 'p', 2, 6.25), # zonglv, 5000
('dc', 'c', 1, 3.75), # yumi, 2000
('dc', 'cs', 1, 3.75), # dianfen, 2000
('zc', 'CF', 5, 6.25 ),  # cotton, 12000

('sc', 'ag', 1, 9.25), # Ag, 3000
('sc', 'cu', 10, 7.75), # Cu, 30000
('sc', 'zn', 5, 7.75), # Zn, 15000
('sc', 'al', 5, 7.75), # Al, 12000
('sc', 'ni', 10, 7.75), # Ni, 60000
]

if __name__ == "__main__":
    for exchange, commodity, _, _ in futures:
        year = 2015  
        for month in range(1, 13):  
            print 'month =', month
            ticks = load_ticks(exchange, commodity, year, [month], use_cache=False)
            fname = '%s/ticks_%s%d%02d.pkl' % (base_dir, commodity, year % 1000, month)
            print 'Saving', fname, '...'
            with open(fname, 'w') as f:
                cPickle.dump(ticks, f)
    
#    ticks = load_ticks('dc', 'pp', 2015, range(12,13), use_cache=True)
       
