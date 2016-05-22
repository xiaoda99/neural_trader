import numpy as np
from time import sleep
import pylab as plt
from profilehooks import profile
import matplotlib.gridspec as gridspec
from collections import OrderedDict

from utils import dict_append, dict_concat, as_arrays, trunc_arrays, Timer

class Strategy(object):
    def __init__(self, name, history, tick_size, hours_per_day, show_freq=None, save_freq=10):
        self.__dict__.update(locals())
        del self.self
        
        self.history['last_price'] = self.history['last_price'].astype('int32') / tick_size 
#        self.buffers = self.history.values()
        self.buffer_len = self.history['last_price'].size
        self.indicators = []
        self.now = 0
        self.d = OrderedDict()
        self.vd = OrderedDict()
        
    def add_indicator(self, indicator):
        self.indicators.append(indicator)
       
    def show(self):
        plt.ion()
        plt.clf()
        
        gses = gridspec.GridSpec(1, 1)
        
        for ind, gs in zip(self.indicators, gses):
            ind.plot(gs)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.draw()
#        sleep(0.3)
         
#    @profile
    def step(self):
        rval = OrderedDict()
        rvec = OrderedDict()
        self.now += 1
            
        for ind in self.indicators:
            ind.step()
        if (self.now - 1) % self.save_freq == 0:
            rval['step'] = self.now - 1 
            rval['price'] = self.history['last_price'][self.now - 1]
            rval['pos'] = self.history['pos'][self.now - 1]
            rval['time_in_ticks'] = self.history['time_in_ticks'][self.now - 1]
            for ind in self.indicators:
                rval.update(ind.output())
                if hasattr(ind, 'output_sparse_vec'):
                    rvec.update(ind.output_sparse_vec())
        if self.show_freq is not None and self.price[1] >= self.indicators[0].m and self.price[1] % self.show_freq == 0:
            self.show()
        dict_append(self.d, rval)
        if len(rvec) > 0:
            dict_concat(self.vd, rvec)
        return
            
    def is_opening_tick(self):
        return self.history['time_in_ticks'][self.now - 1] == 0
            
    def run(self):
        while self.now < self.buffer_len:
            self.step()
            
        self.d = as_arrays(self.d)
        self.vd = trunc_arrays(self.vd)
        self.d.update(self.vd)
        print 'd.keys() =', self.d.keys()
        with Timer() as t:
            np.savez_compressed(self.name + '.npz', **self.d)
        print 'Savez took %f sec.' % t.interval