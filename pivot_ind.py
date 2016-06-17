import math
import time
import operator

import numpy as np
import scipy
from scipy import ndimage
import pylab as plt
from profilehooks import profile
import matplotlib.gridspec as gridspec
import gc

from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

try:
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict  #XD
    
from common_ind import EMA, SMA, MVAR, MDEV
from utils import *

square = lambda x : x**2
inverse_sub = lambda x, y : y - x
inverse_div = lambda x, y : y / (x + 1e-5)

def get(a):
    if a[1] is None or a[1] < a[3]:
        return None
    return a[0][a[1]]

def get_valid_part(a):
    if a[3] == 0:
        return a[0]
    return a[0][a[3]:-a[3]]

def step_elementwise(res, op, arg0, arg1=None):
    if arg0[1] is None:
        return res
    assert res[1] is None or res[1] == arg0[1] - 1
    assert arg0[1] >= res[3] 
    if arg1 is not None:
        if type(arg1) == list: 
            res[0][arg0[1]] = op(arg0[0][arg0[1]], arg1[0][arg0[1]])
        else:
            res[0][arg0[1]] = op(arg0[0][arg0[1]], arg1)
    else:
        res[0][arg0[1]] = op(arg0[0][arg0[1]])
    res[1] = arg0[1]
    return res
 
def step_delayed_sma(x, ma, delay):
    if x[1] is None or x[1] < delay:
        ma[1] = None
    elif x[1] == delay:
        ma[1] = 0
        ma[0][ma[1]] = x[0][:delay+1].sum() / (delay * 2 + 1)
    elif x[1] < delay * 2 + 1:
        ma[1] += 1
        assert ma[1] + delay == x[1]
        ma[0][ma[1]] = (ma[0][ma[1] - 1] * (delay * 2 + 1) + x[0][ma[1] + delay]) / (delay * 2 + 1)
    else:
        ma[1] += 1
        assert ma[1] + delay == x[1]
        ma[0][ma[1]] = (ma[0][ma[1] - 1] * (delay * 2 + 1) + x[0][ma[1] + delay] - x[0][ma[1] - 1 - delay]) / (delay * 2 + 1) 
    return ma

def step_delayed_moving_sum(x, ms, delay):
#    if x[1] is None or x[1] < delay:
#        ms[1] = None
#    elif x[1] == delay:
#        ms[1] = 0
#        ms[0][ms[1]] = x[0][:delay+1].sum()
#    elif x[1] < delay * 2 + 1:
#        ms[1] += 1
#        assert ms[1] + delay == x[1]
#        ms[0][ms[1]] = ms[0][ms[1] - 1] + x[0][ms[1] + delay]
    if x[1] is None or x[1] < x[3] + delay * 2:
        ms[1] = None
    elif x[1] == x[3] + delay * 2:
        ms[1] = x[3] + delay
        ms[0][ms[1]] = x[0][ms[1] - delay : ms[1] + delay + 1].sum()
    else:
        assert ms[1] + 1 + delay == x[1], str(x[1])
        ms[1] += 1
        ms[0][ms[1]] = ms[0][ms[1] - 1] + x[0][ms[1] + delay] - x[0][ms[1] - 1 - delay]
#        ms[0][ms[1]] = x[0][ms[1] - delay : ms[1] + delay + 1].sum()
        if ms[0][ms[1]] < 0:
            print 'old =', ms[0][ms[1] - 1], '+', x[0][ms[1] + delay], '-', x[0][ms[1] - 1 - delay], \
                'new =', ms[0][ms[1]], '@', ms[1]
            if abs(ms[0][ms[1]]) < 1e-7: 
                ms[0][ms[1]] = 0
            else:
                print 'wrong moving sum value:', ms[0][ms[1]]
    return ms

def step_delayed_ma(x, ms, ma, delay):
    step_delayed_moving_sum(x, ms, delay)
    step_elementwise(ma, operator.div, ms, delay * 2 + 1)
    return ma

def follow(last, delay):
    if last is None:
        return None
    ret = last - delay
    if ret < 0:
        ret = None
    return ret

def smooth(a, w):
    return np.convolve(a, w, mode='same')
    
def to_distribution(a):
    assert np.all(a >= 0)
    return a * 1. / a.sum()

def downsample(a, fact):
    print 'downsamle', a.size, 'by', fact, 'to', a.size / fact
    pad_size = math.ceil(float(a.size)/fact)*fact - a.size
    a_padded = np.append(a, np.zeros(pad_size)*np.NaN)
    return scipy.nanmean(a_padded.reshape(-1, fact), axis=1)

def calc_entropy(p):
    return (-p * np.log2(p+1e-10)).sum()

def round_to_even(val):
    return int(2.*round(val/2))

def round_to_odd(val):
    return int(2.*round((val+1)/2)-1)

def n2str(n):
#    assert n % 120 == 0, str(n)
    m = n / 120
#    if m < 60:
    if True:
        return str(m) + 'm'
    assert m % 60 == 0, str(m)
    h = m / 60
    return str(h) + 'h'

class ProfileBase(object):
    def __init__(self, s, n, m,
                 name=None,
                 std_saliency_thld=1.5,
                 output_thlds=[1.5, 2.5, 3],
                 price_range=None,
                 tick_size=1,
                 normalize_offset=True,
                 show_len=None
                 ):
        self.__dict__.update(locals())
        del self.self
        
        self.price = [self.s.history['last_price'], self.s.now - 1, 0, 0] # for PivotProfile
        
        self.buffer_len = self.s.buffer_len
        if self.price_range is None:
            self.price_range = [0, self.price[0].max() + 10]
        if self.name is None:
            self.name = 'piv' + n2str(self.n)
        if self.show_len is None:
            self.show_len = self.m
        
        self.dms = [np.zeros(self.buffer_len), None, n, self.n]
        self.dma = [np.zeros(self.buffer_len), None, 0, self.n]
        self.bias = [np.zeros(self.buffer_len), None, 0, self.n]
        if isinstance(self, PivotProfile):
            self.squared_bias = [np.zeros(self.buffer_len), None, 0, self.n]
            self.squared_bias_dms = [np.zeros(self.buffer_len), None, n, self.n * 2]
            self.squared_bias_dma = [np.zeros(self.buffer_len), None, 0, self.n * 2]
            self.denorm = [np.zeros(self.buffer_len), None, 0, self.n * 2]
            self.denorm2 = [np.zeros(self.buffer_len), None, 0, self.n * 2]
            self.saliency = [np.zeros(self.buffer_len), None, 0, self.n * 2]
            self.buffers = [self.dms, self.dma, self.bias, self.squared_bias, self.squared_bias_dms, self.squared_bias_dma, self.denorm, self.denorm2, self.saliency]
        else:
            assert isinstance(self, SRProfile)
            self.saliency = self.bias
            self.buffers = [self.dms, self.dma, self.bias]
#        for b in self.buffers:
#            b[0][:] = np.inf
        self.precomputed = False
        self.denorm_EMA = EMA(self.m) 
        
        self.saliency_MDEV = MDEV(self.m * 4, dev_type='standard', warmup_steps=self.m/4)
        self.saliency_mstdev = np.zeros(self.buffer_len)
        self.bias_MDEV = MDEV(self.m * 4, dev_type='absolute', warmup_steps=self.m/4, filter_thld=4)
        self.v = np.zeros(self.buffer_len)
        
        self.saliency_vec = np.zeros(self.price_range[1], dtype='float32')
        self.access_time_profile = np.zeros(self.price_range[1], dtype='int32')
        self.init_profiles()
        
        self.sparse_lrn_configs = [
                             {'square' : False, 'w' : 0.5},
                             {'square' : False, 'w' : 1.},
                             {'square' : True, 'w' : 0.5},
                             {'square' : True, 'w' : 1.},
                             ]
        
    def init_profiles(self):
        raise NotImplementedError()
    
    def precompute(self):
        self.w = np.ones(self.n * 2 + 1) / float(self.n * 2 + 1)
        self.dma[0][:] = np.convolve(self.price[0], self.w, mode='same')
        self.bias[0][:] = self.price[0] - self.dma[0]
        if isinstance(self, PivotProfile):
            self.squared_bias[0][:] = self.bias[0]**2
            self.squared_bias_dma[0][:] = np.convolve(self.squared_bias[0], self.w, mode='same')
            self.denorm[0][:] = np.sqrt(self.squared_bias_dma[0])
            for i in range(self.denorm[3], self.buffer_len):
                self.denorm2[0][i] = (lambda x : max(x, self.denorm_EMA.step(x)))(self.denorm[0][i])
            self.saliency[0] = self.bias[0] / (self.denorm2[0] + 1e-5)
        else:
            pass
        self.precomputed = True
       
#    @profile
    def _precomputed_step(self):
        last = self.price[1]
        for buf in self.buffers:
            buf[1] = follow(last, buf[2])
            last = buf[1]
    
    def _step(self):
        step_delayed_ma(self.price, self.dms, self.dma, self.n)
        step_elementwise(self.bias, inverse_sub, self.dma, self.price)
        if isinstance(self, PivotProfile):
            step_elementwise(self.squared_bias, square, self.bias)
                
            step_delayed_ma(self.squared_bias, self.squared_bias_dms, self.squared_bias_dma, self.n)
            
            step_elementwise(self.denorm, math.sqrt, self.squared_bias_dma)
            step_elementwise(self.denorm2, lambda x : max(x, self.denorm_EMA.step(x)), self.denorm)
            step_elementwise(self.saliency, inverse_div, self.denorm2, self.bias)
        else:
            pass
        
    def get_saliency(self):
        sal = get(self.saliency)
        if sal is None:
            return None, None
        sal_mstdev = self.saliency_MDEV.step(sal)
        if sal_mstdev is None:
            return None, None
        self.saliency_mstdev[self.saliency[1]] = sal_mstdev
        std_sal = sal / sal_mstdev
        sal = sal * (abs(std_sal) >= self.std_saliency_thld)
        price = self.price[0][self.saliency[1]]
        return price, sal
    
    def accum_saliency(self):
        raise NotImplementedError()
        
    def update_access_time_profile(self):
        price = self.price[0][self.s.now - 1]
        prev_price = self.price[0][self.s.now - 2] if self.s.now - 2 >= 0 else price
        # deal with price gap between ticks
        if price > prev_price:
            self.access_time_profile[prev_price + 1 : price + 1] = self.s.now - 1
        if price < prev_price:
            self.access_time_profile[price : prev_price] = self.s.now - 1
        else:
            self.access_time_profile[price] = self.s.now - 1
            
    def step(self):
        self.price = [self.s.history['last_price'], self.s.now - 1, 0, 0]
        if self.precomputed:
            self._precomputed_step()
        else:
            self._step()
            
        self.update_access_time_profile()
        
        bias = get(self.bias)
        if bias is None:
            return
        self.volatility = self.bias_MDEV.step(bias)
        if self.volatility is not None:
            self.v[self.bias[1]] = self.volatility
        
        price, sal = self.get_saliency()
        if sal is None:
            return
        self.accum_saliency(price, sal)
        
    def normaliza_profile(self, profile, profile_range):
        p = profile[profile_range[0] : profile_range[1]]
        p2 = p**2
        w = self.volatility * 10
        if w > profile_range[1] - profile_range[0]:
            print 'exception in normalize_profile:', w, '>', profile_range[1] - profile_range[0]
        w = min(w, profile_range[1] - profile_range[0])
        v = np.ones(w) / w 
        denorm = np.sqrt(np.convolve(p2, v, mode='same'))
        min_denorm = np.sqrt(p2.mean()) * 1.5
        denorm[denorm < min_denorm] = min_denorm
        profile[profile_range[0] : profile_range[1]] = p / denorm
        return profile
        
    def sparse_lrn(self, strengths, square=False, w=0.5):
        w = int(round(strengths.shape[0] * w))
        if w < 1:
            print 'length-1 profile!'
            w = 1
        v = np.ones(w) / w
        s = strengths**2 if square else strengths
        denorm = np.convolve(s, v, mode='same')
        if square:
            denorm = np.sqrt(denorm)
        return strengths / denorm 
        
    def smooth_profile(self, profile, profile_range, normalize=True):
        smoothing_win = self.volatility * 0.5  # choose 0.5 for no good reason
        smoothed = ndimage.filters.gaussian_filter1d(profile, smoothing_win / 2., mode='constant')
#        smoothed = profile
        smoothed[:profile_range[0]] = 0.
        smoothed[profile_range[1]:] = 0.
        
        if normalize:
#            smoothed = self.normaliza_profile(profile, profile_range)
            denorm = smoothed[profile_range[0]:profile_range[1]].mean()
##            denorm = np.sqrt(np.square(smoothed[profile_range[0]:profile_range[1]]).mean())
            if denorm > 0:
                smoothed /= denorm
            else:
                assert denorm == 0
                smoothed *= 0. 
        return smoothed
    
    @staticmethod
    def sparsify_profile(pivot_profile, access_time_profile, thld, normalize=False):
        levels = np.where(pivot_profile >= thld)[0]
        strengths = pivot_profile[levels]
        atimes = access_time_profile[levels]
#        if normalize and strengths.shape[0] > 0:
#            strengths_list = []
#            for kwargs in self.sparse_lrn_configs:
#                strengths_list.append(self.sparse_lrn(strengths, **kwargs))
#            strengths = np.array(strengths_list).T
        return levels, strengths, atimes
    
    def output_vec(self):
        rval = OrderedDict()
        if self.s.now < self.m:
            rval[self.name] = np.zeros_like(self.saliency_vec)
            return rval
        rval[self.name] = self.smoothed_pivot_profile
        return rval
    
    def output_sparse_vec(self):
        raise NotImplementedError()
    
    def output_default(self, price, profile_range, rval):
        rval[self.name + '.volatility'] = 0.
        rval[self.name + '.level_start'] = profile_range[0]
        rval[self.name + '.level_stop'] = profile_range[1]
        rval[self.name + '.price_strength'] = 0.
        return rval
        
    def _output(self, price, profiles, profile_range, thld, rval):
        name = self.name + '_th' + str(thld)
        
#        S, R = self.get_nearby_levels_sparse(price, profile, profile_range, thld)
        s_profile, r_profile = profiles[0], profiles[1]
        S, S_strength = self.get_nearby_support(price, s_profile, profile_range, thld)
        R, R_strength = self.get_nearby_resistance(price, r_profile, profile_range, thld)
        R_offset = R - price
#        R_strength = r_profile[R] 
        R_elapsed = max(1., (self.s.now - 1 - self.access_time_profile[R]) * 1. / self.m)
        S_offset = price - S
#        S_strength = s_profile[S]
        S_elapsed = max(1., (self.s.now - 1 - self.access_time_profile[S]) * 1. / self.m)
        if self.normalize_offset:
            R_offset = R_offset * 1. / self.volatility
            S_offset = S_offset * 1. / self.volatility
    #        assert 0. <= rval['R_elapsed'] <= 1., str(rval['R_elapsed'])
    #        assert 0. <= rval['S_elapsed'] <= 1., '%f, now = %d, t = %d' % (rval['S_elapsed'], self.s.now, self.access_time_profile[i])
        
        if R_offset == 0 and S_offset != 0: # just on the highest
            RS_ratio = -10.
        elif R_offset != 0 and S_offset == 0: # just on the lowest
            RS_ratio = 10.
        elif R_offset == 0 and S_offset == 0: # just on the S/R line
            RS_ratio = 0. 
        else:
            RS_ratio = np.log(R_offset * 1. / (S_offset))
            assert abs(RS_ratio) <= 10., str(RS_ratio) + '= log(-' + str(R_offset) + '/' + str(S_offset) + ')'
        rval[name + '.R'] = R
        rval[name + '.R_offset'] = R_offset
        rval[name + '.R_strength'] = R_strength 
        rval[name + '.R_elapsed'] = R_elapsed
        rval[name + '.S'] = S
        rval[name + '.S_offset'] = S_offset
        rval[name + '.S_strength'] = S_strength
        rval[name + '.S_elapsed'] = S_elapsed
        rval[name + '.R/S_ratio'] = RS_ratio 
        return rval
    
    def get_profiles(self, profile_range):
        raise NotImplementedError()
    
    def output(self, profile_range=None):
        rval = OrderedDict()
        price = get(self.price)
        if profile_range is None:
            m = min(self.m, self.s.now)
            profile_range = [self.price[0][self.s.now - m : self.s.now].min(), 
                        self.price[0][self.s.now - m : self.s.now].max() + 1]
        
        if self.s.now < self.m:
            self.output_default(price, profile_range, rval)
            return rval
        
        rval[self.name + '.volatility'] = self.volatility
        rval[self.name + '.level_start'] = profile_range[0]
        rval[self.name + '.level_stop'] = profile_range[1] 
        profiles = self.get_profiles(profile_range)
        rval[self.name + '.price_strength'] = self.dense_profiles[0][price]
        
#        for thld in self.output_thlds: 
#            self._output(price, profiles, profile_range, thld, rval)
        return rval
    
    def plot(self, gs):
        unit_len = self.show_len * 1. / 5.
        if self.s.now - self.show_len < 0:
            return 
            
        price = self.price[0][self.s.now - self.show_len : self.s.now]
        profile_range = [price.min(), price.max() + 1]
        floor, ceil = profile_range[0] - 1, profile_range[1] + 1
            
        d = self.output(3, profile_range)
        
        ax = plt.subplot(gs)
        plt.plot(price)
        day_begin = np.where(self.s.history['time_in_ticks'][self.s.now - self.show_len : self.s.now] == 0)[0]
        for x in day_begin:
            plt.axvline(x, color='r', linestyle=':')
        y = self.smoothed_pivot_profile[floor : ceil]
        plt.barh(np.arange(floor, ceil) - 0.5, y * unit_len, 1.0, label=self.name,
                 alpha=0.2, color='r', edgecolor='none')
        
        last_price = int(get(self.price))
        support = last_price + int(round((d['S_offset']) * self.volatility))
        resistance = last_price + int(round((d['R_offset']) * self.volatility))
        highlighted = [support, resistance]
        plt.barh(np.array(highlighted) - 0.5, self.smoothed_pivot_profile[highlighted] * unit_len, 1.0,
                 alpha=1.0, color='r', edgecolor='none')
        ax.set_xticks(np.arange(0, self.show_len * 1.22, unit_len))
        ax.xaxis.grid(b=True, linestyle='--')
        ax.yaxis.grid(b=False)
        plt.legend(loc='upper right')
        return ax
    
class PivotProfile(ProfileBase):
    def init_profiles(self):
        self.saliency_vec_EMA = EMA(self.m)
        
        self.split_stats = []
#        self.dlevels = np.array([], dtype='int32')
#        self.asserters = [SoftAsserter('Asserter0'), SoftAsserter('Asserter1')]
        
    def accum_saliency(self, price, sal):
        self.saliency_vec *= 0
        self.saliency_vec[price] = abs(sal)
        self.pivot_profile = self.saliency_vec_EMA.step(self.saliency_vec)
        
    def get_profiles(self, profile_range):
        pivot_profile = self.smooth_profile(self.pivot_profile, profile_range, normalize=True)
        thld = min(self.output_thlds)
        sparse_profile = self.sparsify_profile(pivot_profile, self.access_time_profile, thld)
        self.dense_profiles = [pivot_profile,]
        self.profiles = [sparse_profile,]
        return self.profiles 
    
    def output_sparse_vec(self):
        rval = OrderedDict()
#        assert len(self.profiles) == 1
        if self.s.now < self.m:
            rval[self.name] = np.zeros((0, 4), dtype='float32')
            return rval
        levels, strengths, atimes = self.profiles[0]
        steps = np.ones_like(levels) * (self.s.now - 1)
        rval[self.name] = np.vstack([steps, levels, strengths, atimes]).astype('float32').T
        return rval
    
class SRProfile(ProfileBase):
    def init_profiles(self):
        self.s_saliency_vec_EMA = EMA(self.m)
        self.s_visit_time_profile = np.zeros(self.price_range[1], dtype='int32')
        self.s_visit_time_profile[:] = -np.inf
        self.r_saliency_vec_EMA = EMA(self.m)
    
    def _step(self):
        step_delayed_ma(self.price, self.dms, self.dma, self.n)
        step_elementwise(self.bias, inverse_sub, self.dma, self.price)
        step_elementwise(self.squared_bias, square, self.bias)
            
        step_delayed_ma(self.squared_bias, self.squared_bias_dms, self.squared_bias_dma, self.n)
        
        step_elementwise(self.denorm, math.sqrt, self.squared_bias_dma)
        step_elementwise(self.denorm2, lambda x : max(x, self.denorm_EMA.step(x)), self.denorm)
        step_elementwise(self.saliency, inverse_div, self.denorm2, self.bias)
    
    def accum_saliency(self, price, sal):
        self.saliency_vec *= 0
        self.saliency_vec[price] = abs(sal)
        if sal > 0:
            self.resistance_profile = self.r_saliency_vec_EMA.step(self.saliency_vec)
        elif sal < 0:
            self.support_profile = self.s_saliency_vec_EMA.step(self.saliency_vec)
        
    def get_profiles(self, profile_range):
        if not hasattr(self, 'support_profile'):
            self.support_profile = self.saliency_vec * 0.
        if not hasattr(self, 'resistance_profile'):
            self.resistance_profile = self.saliency_vec * 0.
            
        s_profile = self.smooth_profile(self.support_profile, profile_range, normalize=True)
        sparse_s_profile = self.sparsify_profile(s_profile)
        r_profile = self.smooth_profile(self.resistance_profile, profile_range, normalize=True)
        sparse_r_profile = self.sparsify_profile(r_profile)
        self.dense_profiles = [s_profile, r_profile]
        self.profiles = [sparse_s_profile, sparse_r_profile]
        return sparse_s_profile, sparse_r_profile
    
#    def filter_levels(self, levels, strengths):
#        i = strengths.argmax()
#        return levels[[i]], strengths[[i]]
    
    def output_sparse_vec(self):
        rval = OrderedDict()
        price = get(self.price)
#        assert len(self.profiles) == 2
        if self.s.now < self.m:
            rval[self.name + '.levels'] = []
            rval[self.name + '.strengths'] = []
            return rval
        s_levels, s_strengths = self.profiles[0]
        r_levels, r_strengths = self.profiles[1] 
        levels = np.concatenate([s_levels[s_levels < price], r_levels[r_levels > price]]) 
        strengths = np.concatenate([s_strengths[s_levels < price], r_strengths[r_levels > price]])
        if price in s_levels or price in r_levels:
            level = price
            strength = np.concatenate([s_strengths[s_levels == price], r_strengths[r_levels == price]]).max()
            levels = np.append(levels, level) 
            strengths = np.append(strengths, strength)
        
        rval[self.name + '.levels'] = [(self.s.now - 1, l) for l in levels]
        rval[self.name + '.strengths'] = list(strengths)
        return rval
    
    def output_sparse_vec2(self):
        rval = OrderedDict()
#        assert len(self.profiles) == 2
        if self.s.now < self.m:
            rval[self.name + '.S_levels'] = []
            rval[self.name + '.S_strengths'] = []
            rval[self.name + '.R_levels'] = []
            rval[self.name + '.R_strengths'] = []
            return rval
        s_levels, s_strengths = self.profiles[0]
        r_levels, r_strengths = self.profiles[1] 
        rval[self.name + '.S_levels'] = [(self.s.now - 1, l) for l in s_levels]
        rval[self.name + '.S_strengths'] = list(s_strengths)
        rval[self.name + '.R_levels'] = [(self.s.now - 1, l) for l in r_levels]
        rval[self.name + '.R_strengths'] = list(r_strengths)
        return rval
    
from build_tick_dataset import load_ticks
from strategy import Strategy
   
def test_precompute():
    ticks = load_ticks('dc', 'pp', 2015, range(9,10), use_cache=True)
    
    s = Strategy(ticks, 3.75, show_freq=None)
    ind = PivotProfile(s, 5*60*2, int(1 * 3.75 * 60 * 60 * 2))
    t0 = time.time()
    ind.precompute()
    t1 = time.time()
    print 't_precompute =', t1 - t0, 'sec'
    
    s.add_indicator(ind)
    t0 = time.time()
    s.run()
    t1 = time.time()
    print 't =', t1 - t0, 'sec'
    
    s2 = Strategy(ticks, 3.75, show_freq=None)
    ind2 = PivotProfile(s2, 5*60*2, int(1 * 3.75 * 60 * 60 * 2))
    s2.add_indicator(ind2)
    t0 = time.time()
    s2.run()
    t1 = time.time()
    print 't =', t1 - t0, 'sec'
    
    sal = ind.saliency[0][ind.saliency[3]:-ind.saliency[3]]
    sal2 = ind2.saliency[0][ind2.saliency[3]:-ind2.saliency[3]]
    print np.abs(sal - sal2).mean(), np.abs(sal - sal2).max(), np.abs(sal).mean()

def piecewise_plot(d, n_pieces, n_plotted=None, vec_inds=[], inds=[], conds=[], cond_colors=['cyan', 'yellow'], xtick=10*60*2, ytick=None, stride=10, save_name=None):
    total_len = d['step'].shape[0]
    piece_len = int(math.ceil(total_len * 1. / n_pieces))
    if n_plotted is None:
        n_plotted = n_pieces
    for i, start in enumerate(range(0, total_len, piece_len)):
        if i == n_plotted:
            return
        stop = min(start + piece_len, total_len)
        save_name_i = save_name.replace('.png', '.part' + str(i) + '.png') if save_name is not None else None
#        plot(d, vec_inds=vec_inds, inds=inds, conds=conds, cond_colors=cond_colors, xtick=xtick, ytick=ytick, stride=stride, save_name=save_name_i,
#             start=start, stop=stop)
        plot2(d, vec_inds=vec_inds, inds=inds, xtick=xtick, ytick=ytick, stride=stride, save_name=save_name_i,
             start=start, stop=stop)
 
scale_plot_configs = [
           [((1.5, 2.), 'gold', 0.05),
            ((2., 2.5), 'gold', 0.1), 
           ((2.5, 3.), 'salmon', 0.2, ), 
           ((3., 3.5), 'red', 0.3, ),
           ((3.5, np.inf), 'brown', 0.5),
           ],
            
           [((1.5, 2.), 'cyan', 0.05),
            ((2., 2.5), 'cyan', 0.1), 
           ((2.5, 3.), 'dodgerblue', 0.2, ), 
           ((3., 3.5), 'blue', 0.3, ),
           ((3.5, np.inf), 'navy', 0.5),
           ],
                      
           [((1.5, 2.), 'lightgreen', 0.05),
            ((2., 2.5), 'lightgreen', 0.1), 
           ((2.5, 3.), 'limegreen', 0.2, ), 
           ((3., 3.5), 'green', 0.3, ),
           ((3.5, np.inf), 'darkgreen', 0.5),
           ],
        ]
                       
plot_configs = [
           [((1.5, 2.), 'cyan', 0.05),
            ((2., 2.5), 'turquoise', 0.1), 
           ((2.5, 3.), 'lime', 0.1, ), 
           ((3., 3.5), 'green', 0.1, ),
           ((3.5, np.inf), 'darkgreen', 0.15),
           ],
            
           [((1.5, 2.), 'yellow', 0.05),
            ((2., 2.5), 'gold', 0.1),  
           ((2.5, 3.), 'darkorange', 0.15, ),
           ((3., 3.5), 'red', 0.15, ),
           ((3.5, np.inf), 'purple', 0.2),
           ],
                
#           [((1.5, 2.), 'cyan', 0.05),
#            ((2., 2.5), 'turquoise', 0.05), 
#           ((2.5, 3.), 'lime', 0.1, ), 
#           ((3., 3.5), 'limegreen', 0.1, ),
#           ((3.5, 5.), 'green', 0.15), 
#           ((5., np.inf), 'darkolivegreen', 0.15)],
#            
#           [((1.5, 2.), 'white', 0.00),
#            ((2., 2.5), 'gold', 0.05),  
#           ((2.5, 3.), 'darkorange', 0.15, ),
#           ((3., 3.5), 'red', 0.15, ),
#           ((3.5, 5.), 'purple', 0.2), 
#           ((5., np.inf), 'indigo', 0.2)],
           ]
  
def plot2(d, vec_inds=[], inds=[], xtick=10*60*2, ytick=None, stride=10, save_name=None, start=None, stop=None):
    if ytick is None:
        ytick = d['piv30m.volatility'].mean()
    if start is None:
        start = 0
    if stop is None:
        stop = d['step'].shape[0]
#    print start, stop
    step = d['step']
    price = d['price']
    if price.shape[0] > step.shape[0]:
        price, _, _ = get_price(price, save_freq=stride)
        assert price.shape[0] == step.shape[0]
    step = step[start:stop]
    price = price[start:stop]
    
    nplots = len(vec_inds)
    height_ratios=[2] * nplots
    dpi = 1000
    width = min(30000, step.shape[0]) / dpi
    height = sum(height_ratios) 
    figsize=(width, height)
    print 'figsize =', figsize
    fig = plt.figure(figsize=figsize)
    
    gs = gridspec.GridSpec(nplots, 1, height_ratios=height_ratios)
    
    for i in range(nplots):
        if i == 0:
            ax0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharex=ax0)
        
        ax.set_xlim([step[0], step[-1]])
        span = price.max() - price.min()
        ax.set_ylim([price.min() - span * 0.1, price.max() + span * 0.1])
        
        for ind in inds:
            plt.plot(step, d[ind][start:stop], alpha=1, linewidth=.1, zorder=3, label=ind)
        plt.plot(step, price, color='k', linewidth=.1, zorder=3)
        
        ax.set_xticks(np.arange(step[0], step[-1], xtick))
        ax.set_yticks(np.arange(price.min(), price.max(), ytick))
        ax.xaxis.grid(b=True, color='gray', linestyle='-', linewidth=.05, alpha=.5, zorder=0)
        ax.yaxis.grid(b=True, color='gray', linestyle='-', linewidth=.05, alpha=.5, zorder=0)
                
        opening = (d['time_in_ticks'] < np.roll(d['time_in_ticks'], 1))
        opening = opening[start:stop]
        for x in step[opening]:
            plt.axvline(x, color='k', linewidth=.1, alpha=1, zorder=2)
            
        assert 'merged' in vec_inds[i], vec_inds[i]
        ind = vec_inds[i]
        locations = d[ind][:,:2].astype('int32')
        strengths = d[ind][:,2]
        scales = d[ind][:,-1].astype('int32')
        n_scales = scales.max() + 1
        assert n_scales == 3  # 5m+10m, 10m+30m, 30m+1h
        for scale in range(n_scales):
            for (low, high), color, alpha in scale_plot_configs[scale]:
                x, y = locations[(scales == scale) & 
                                 (locations[:,0] >= step[0]) & (locations[:,0] < step[-1]) & 
                                 (strengths >= low) & (strengths < high)].T
                plt.scatter(x, y, marker='_', s=0.1, linewidth=0.1, color=color, alpha=alpha)
        
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    if save_name is not None:
        with Timer() as t:
            plt.savefig(save_name, format='png', dpi=dpi, bbox_inches='tight')
        print 'plt.savefig %s took %f sec.' % (save_name, t.interval)
    else:
        plt.show()
        
    fig.clf()
    plt.close()
    gc.collect()
          
def plot(d, vec_inds=[], inds=[], extra_inds=['pos'], conds=[], cond_colors=['cyan', 'yellow'], xtick=10*60*2, ytick=None, stride=10, save_name=None, start=None, stop=None):
    if ytick is None:
        ytick = d['piv30m.volatility'].mean()
    if start is None:
        start = 0
    if stop is None:
        stop = d['step'].shape[0]
#    print start, stop
    step = d['step']
    price = d['price']
    if price.shape[0] > step.shape[0]:
        price, _, _ = get_price(price, save_freq=stride)
        assert price.shape[0] == step.shape[0]
    step = step[start:stop]
    price = price[start:stop]
    
    assert len(extra_inds) == 1
    nplots = 1 + 1
    height_ratios=[2] + [1] * 1
    dpi = 1000
    width = min(30000, step.shape[0]) / dpi
    height = sum(height_ratios) 
    figsize=(width, height)
#    print 'figsize =', figsize
    fig = plt.figure(figsize=figsize)
    
    gs = gridspec.GridSpec(nplots, 1, height_ratios=height_ratios)
    i = 0
    ax0 = plt.subplot(gs[i])
    i += 1
    
    ax0.set_xlim([step[0], step[-1]])
    span = price.max() - price.min()
    ax0.set_ylim([price.min() - span * 0.1, price.max() + span * 0.1])
    
    for ind in inds:
        plt.plot(step, d[ind][start:stop], alpha=1, linewidth=.1, zorder=3, label=ind)
    plt.plot(step, price, color='k', linewidth=.1, zorder=3)
    
    ax0.set_xticks(np.arange(step[0], step[-1], xtick))
    ax0.set_yticks(np.arange(price.min(), price.max(), ytick))
    ax0.xaxis.grid(b=True, color='gray', linestyle='-', linewidth=.05, alpha=.5, zorder=0)
    ax0.yaxis.grid(b=True, color='gray', linestyle='-', linewidth=.05, alpha=.5, zorder=0)
#    ax0.yaxis.grid(b=False)
            
    opening = (d['time_in_ticks'] < np.roll(d['time_in_ticks'], 1))
    opening = opening[start:stop]
    for x in step[opening]:
        plt.axvline(x, color='k', linewidth=.1, alpha=1, zorder=2)
        
    if len(vec_inds) == 1 and vec_inds[0].endswith('merged'):
        print 'plot merged'
        ind = vec_inds[0]
        locations = d[ind][:,:2].astype('int32')
        strengths = d[ind][:,2]
        scales = d[ind][:,-1].astype('int32')
        n_scales = scales.max() + 1
        assert n_scales == 3  # 5m+10m, 10m+30m, 30m+1h
        for scale in range(n_scales):
            for (low, high), color, alpha in scale_plot_configs[scale]:
                x, y = locations[(scales == scale) & 
                                 (locations[:,0] >= step[0]) & (locations[:,0] < step[-1]) & 
                                 (strengths >= low) & (strengths < high)].T
                plt.scatter(x, y, marker='_', s=0.1, linewidth=0.1, color=color, alpha=alpha)
    else:
        for j, ind in enumerate(vec_inds):
            l = d[ind][:,:2]
            s = d[ind][:,2]
            plot_config = plot_configs[j]
#            colors = colors[1:2]
#            plot_config = plot_config[1:2]
            for (low, high), color, alpha in plot_config:
                x, y = l[(l[:,0] >= step[0]) & (l[:,0] < step[-1]) & (s >= low) & (s < high)].T
                plt.scatter(x, y, marker='_', s=0.1, linewidth=0.1, color=color, alpha=alpha)
        
    for cond, color in zip(conds, cond_colors):    
        cond = cond[start:stop]
        with Timer() as t:
    #        xsignals = np.where(cond)[0] 
            cond_steps = step[cond]
            for x in cond_steps:
                plt.axvline(x, color=color, linewidth=.1, alpha=1., zorder=0)
#        print 'plt.axvline took %f sec.' % t.interval
    plt.legend(loc='upper right', fontsize='xx-small')
        
    for ind in extra_inds:
        ax = plt.subplot(gs[i], sharex=ax0)
        i += 1
        ax.set_xlim([step[0], step[-1]])
        plt.plot(step, d[ind][start:stop], linewidth=.1, label=ind)
        
        ax.set_xticks(np.arange(step[0], step[-1], xtick))
        ax.xaxis.grid(b=True, color='gray', linestyle='-', linewidth=.05, alpha=.5, zorder=0)
        ax.yaxis.grid(b=False)
        plt.legend(loc='upper right', fontsize='xx-small')
        
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    if save_name is not None:
        with Timer() as t:
            plt.savefig(save_name, format='png', dpi=dpi, bbox_inches='tight')
        print 'plt.savefig %s took %f sec.' % (save_name, t.interval)
    else:
        plt.show()
        
    fig.clf()
    plt.close()
    gc.collect()

def load_profiles(d, name):
    steps = d[name][:,0].astype('int32')
    levels = d[name][:,1].astype('int32')
    strengths = d[name][:,2] 
    atimes = d[name][:,3].astype('int32')
    starts = d[name + '.starts']
    stops = d[name + '.stops']
    profiles = [(steps[start:stop], levels[start:stop], strengths[start:stop], atimes[start:stop]) 
                for start, stop in zip(starts, stops)]
    return profiles

strength_thld = 2.5
def filter_profiles(profiles, strength_thld):
    profiles = [(l[s >= strength_thld], s[s >= strength_thld], t[s >= strength_thld]) for l, s, t in profiles]
    return profiles

def densify_profile(dense_profile, levels, strengths):
    dense_profile *= 0
    dense_profile[levels] = strengths
    return dense_profile
    
def merge_profiles(profiles, volatilities, n_profiles, thld=1.5, average_adjacent_profiles=True):
    assert len(profiles) == n_profiles, len(profiles)
    assert len(volatilities) == n_profiles, len(volatilities)
    if not hasattr(merge_profiles, 'dense_profiles'):
        merge_profiles.dense_profiles = [np.zeros(8000, dtype='float32') for _ in range(n_profiles)]
    if not hasattr(merge_profiles, 'dense_atime_profiles'):
        merge_profiles.dense_atime_profiles = [np.zeros(8000, dtype='int32') for _ in range(n_profiles)]
    dense_profiles = [densify_profile(dense_profile, profile[1], profile[2]) 
                      for dense_profile, profile in 
                      zip(merge_profiles.dense_profiles, profiles)] 
    dense_atime_profiles = [densify_profile(dense_profile, profile[1], profile[3]) 
                      for dense_profile, profile in 
                      zip(merge_profiles.dense_atime_profiles, profiles)] 
    if average_adjacent_profiles:
        dense_profiles = [(dense_profiles[i] + dense_profiles[i+1]) / 2. for i in range(len(dense_profiles) - 1)]
        volatilities = [(volatilities[i] + volatilities[i+1]) / 2. for i in range(len(volatilities) - 1)]
        
    dense_profiles = np.vstack(dense_profiles)
    dense_profile = dense_profiles.max(axis=0)
    scale_profile = dense_profiles.argmax(axis=0)
    vol_profile = np.array(volatilities)[scale_profile]    
    dense_atime_profiles = np.vstack(dense_atime_profiles)
    dense_atime_profile = dense_atime_profiles.max(axis=0)
    
    # sparsify again, adding scales and volatilities
    levels = np.where(dense_profile >= thld)[0]
    strengths = dense_profile[levels]
    atimes = dense_atime_profile[levels]
    scales = scale_profile[levels]
    vols = vol_profile[levels]
    
    # FIXME: this way of dealing with steps is ugly
    step = 0
    for steps, _, _, _ in profiles:
        if len(steps) > 0:
            step = steps[0]
            break
    steps = np.ones(levels.shape, dtype='int32') * step
    
    profile = (steps, levels, strengths, atimes, vols, scales)
    return profile
  
def concat_profiles(profiles):
    l = [np.vstack((steps, levels, strengths, atimes, vols, scales)).T 
         for steps, levels, strengths, atimes, vols, scales in profiles]
    return np.vstack(l) 

def merge2(d, n_scale, ns_in_minutes):
#    d['piv_merged'] = d0['piv_merged']
#    d.update(d0)
    names = ['piv' + n2str(n*60*2*n_scale) for n in ns_in_minutes]
    all_profiles = [load_profiles(d, name) for name in names]
    all_volatilities = [d[name + '.volatility'] for name in names]
    
    for profiles, volatilities, merged_name in [(all_profiles[:4], all_volatilities[:4], 'piv_merged2'),
                                                (all_profiles[1:], all_volatilities[1:], 'piv_merged3')]:
        for pivs_and_vols in zip(*(profiles+volatilities)):
            profile = merge_profiles(pivs_and_vols[:4], pivs_and_vols[4:], 4)
            rvec = {merged_name : np.vstack(profile).T.astype('float32')}
            dict_concat(d, rvec)
        trunc_arrays(d, keys=[merged_name,])
    return d

def merge2_old(d0, d, n_scale, ns_in_minutes):
    '''Consume too much memory.'''
    d['piv_merged'] = d0['piv_merged']
    names = ['piv' + n2str(n*60*2*n_scale) for n in ns_in_minutes]
    all_profiles = [load_profiles(d, name) for name in names]
    all_volatilities = [d[name + '.volatility'] for name in names]
    
    for profiles, volatilities, merged_name in [(all_profiles[:4], all_volatilities[:4], 'piv_merged2'),
                                                (all_profiles[1:], all_volatilities[1:], 'piv_merged3')]:
        merged_profiles = [merge_profiles(pivs_and_vols[:4], pivs_and_vols[4:], 4) for pivs_and_vols
                           in zip(*(profiles+volatilities))]
        d[merged_name] = concat_profiles(merged_profiles)
    return d
   
min_strength = 1.
max_gap = .5

def get_dlevels(levels):
    dlevels_below = levels - np.roll(levels, 1)
    dlevels_below[0] = 1000
    dlevels_above = np.roll(levels, -1) - levels
    dlevels_above[-1] = 1000
    return dlevels_below, dlevels_above

def locate_nearest_pivot(profile, lowest, highest, price, volatility):
    levels, strengths, atimes = profile
    if len(levels) == 0:
        lower_bound, upper_bound = lowest, highest
        return lower_bound, upper_bound

    temporal_nearest = levels[atimes == atimes.max()]
    nearest = temporal_nearest[np.abs(temporal_nearest - price).argmin()]
    dlevels_below, dlevels_above = get_dlevels(levels)
    if nearest < price:
        upper_bound = nearest
        lower_bound = levels[(levels <= nearest) & (dlevels_below > max_gap * volatility)].max()
    elif nearest > price:
        lower_bound = nearest
        upper_bound = levels[(levels >= nearest) & (dlevels_above > max_gap * volatility)].min()
    else:  # nearest == price
        lower_bound = levels[(levels <= nearest) & (dlevels_below > max_gap * volatility)].max()
        upper_bound = levels[(levels >= nearest) & (dlevels_above > max_gap * volatility)].min()
    return lower_bound, upper_bound

def output_nearest_pivot(profile, lowest, highest, lower_bound, upper_bound, profile0, volatility):
    levels, strengths, atimes = profile
    levels0, strengths0, atimes0 = profile0
    if lower_bound == lowest and upper_bound == highest:
        center = (lower_bound + upper_bound) * 1. / 2
        max_strength = 0.
        atime = 0 
        downward_dominance = [0., 0., 0.]
        upward_dominance = [0., 0., 0.]
    else:
        inpivot_idx = (levels >= lower_bound) & (levels <= upper_bound)
        max_strength = strengths[inpivot_idx].max()
        center = int(levels[inpivot_idx & (strengths == max_strength)].mean().round())
        atime = atimes.max()
        
        downward_dominance = []
        upward_dominance = []
        for distance in []:
            upper_range = (upper_bound + 1, upper_bound + 1 + distance)
            lower_range = (lower_bound - 1, lower_bound - 1 - distance)
            upper_strengths = strengths0[(levels0 >= upper_range[0]) & (levels0 < upper_range[1])]
            upper_max_strength = upper_strengths.max() if len(upper_strengths) > 0 else min_strength 
            lower_strengths = strengths0[(levels0 >= lower_range[0]) & (levels0 < lower_range[1])]
            lower_max_strength = lower_strengths.max() if len(lower_strengths) > 0 else min_strength
            downward_dominance.append(max_strength / lower_max_strength)
            upward_dominance.append(max_strength / upper_max_strength)  
    return max_strength, center, atime, downward_dominance, upward_dominance

def output_support_resistance(d, profile, lowest, highest, price, price_strength, lower_bound, upper_bound, max_strength, volatility, 
                        max_inpivot_time=1*60*2, decay_factor=.5, accessed_max_strength=True):
    levels, strengths, atimes = profile
    if len(levels) == 0:
        support, resistance = lowest, highest
        support_strength, resistance_strength = 0., 0.
        return support, support_strength, resistance, resistance_strength
    if accessed_max_strength:
        max_strength = strengths[(levels >= lower_bound) & (levels <= upper_bound) & 
                                    (atimes.max() - atimes <= max_inpivot_time)].max()
    effective_strength = max(strength_thld, max_strength * decay_factor)
    effective_strength = min(3.5, effective_strength)
    
    dlevels_below, dlevels_above = get_dlevels(levels)
    strengths_below = np.roll(strengths, 1)
    strengths_below[0] = 0.
    strengths_below = strengths_below * (dlevels_below <= max_gap * volatility)
    strengths_above = np.roll(strengths, -1)
    strengths_below[-1] = 0.
    strengths_above = strengths_above * (dlevels_above <= max_gap * volatility)
    
    supports = levels[(strengths >= effective_strength) & (strengths_above < effective_strength)]
    supports_strengths = strengths[(strengths >= effective_strength) & (strengths_above < effective_strength)]
    resistances = levels[(strengths >= effective_strength) & (strengths_below < effective_strength)] 
    resistances_strengths = strengths[(strengths >= effective_strength) & (strengths_below < effective_strength)]
    if price_strength >= effective_strength: # in pivot, support above resistance (reversed)
        argmin = supports[supports >= price].argmin()
        support = supports[supports >= price][argmin]
        support_strength = supports_strengths[supports >= price][argmin]
        argmax = resistances[resistances <= price].argmax()
        resistance = resistances[resistances <= price][argmax]
        resistance_strength = resistances_strengths[resistances <= price][argmax]
    else: # out of pivot, support below resistance (normal)
        if len(supports[supports < price]) > 0:
            argmax = supports[supports < price].argmax()
            support = supports[supports < price][argmax]
            support_strength = supports_strengths[supports < price][argmax]
        else:
            support = lowest
            support_strength = 0.
        if len(resistances[resistances > price]) > 0:
            argmin = resistances[resistances > price].argmin()
            resistance = resistances[resistances > price][argmin]
            resistance_strength = resistances_strengths[resistances > price][argmin]
        else:
            resistance = highest
            resistance_strength = 0.
#    d['piv5m.support'] = support
#    d['piv5m.support_strength'] = support_strength
#    d['piv5m.resistance'] = resistance
#    d['piv5m.resistance_strength'] = resistance_strength
    return support, support_strength, resistance, resistance_strength
        
def output_support_resistance_old(profile, lowest, highest, price, lower_bound, upper_bound):
    levels, strengths, atimes = profile
#    if lowest not in levels:
#        levels = np.concatenate([[lowest], levels])
#    if highest not in levels:
#        levels = np.concatenate([levels, [highest]])
    if lower_bound <= price <= upper_bound and lower_bound != lowest and upper_bound != highest:
        # search avoid the pivot the price is now in
        lower_start = lower_bound - 1
        upper_start = upper_bound + 1
    else:
        lower_start = price - 1
        upper_start = price + 1
    lower_levels = levels[levels <= lower_start]
    support = lower_levels.max() if len(lower_levels) > 0 else lowest
    upper_levels = levels[levels >= upper_start]
    resistance = upper_levels.min() if len(upper_levels) > 0 else highest    
    return support, resistance

def diff_levels(profile):
    levels, strengths, atimes = profile[:,0], profile[:,1], profile[:,2] 
    diffs = np.diff(levels)
    diffs = diffs[diffs > 1]
    diff_sum = diffs.sum()
    ndiff = len(diffs)
    return diff_sum, ndiff

def get_price(p, save_freq=10):
#    pad = (save_freq - p.shape[0] % save_freq) % save_freq
    pad = int(math.ceil(p.shape[0] * 1. / save_freq) * save_freq) - p.shape[0] 
    p = np.append(p, np.ones(pad) * p[-1])
    p = p.reshape(-1, save_freq)
    price = p[:, 0]
    price_low = p.min(axis=1)
    price_high = p.max(axis=1)
    return price, price_low, price_high
    
def zip2arr(l):
    return [np.array(a) for a in zip(*l)]
    
def load_pivot_inds(profiles):
    global pivot_lower_bound, pivot_upper_bound, pivot_strength, pivot_center, pivot_atime, support, resistance
    l = [locate_nearest_pivot(profile, l, h, p, v) for profile, l, h, p, v in zip(profiles, lowest, highest, price, volatility)]
    pivot_lower_bound, pivot_upper_bound = zip2arr(l)
    assert 'pivot_lower_bound' in globals()
    l = [output_nearest_pivot(profile, l, h, lb, ub, profile, v) for profile, l, h, lb, ub, v in 
         zip(profiles, lowest, highest, pivot_lower_bound, pivot_upper_bound, volatility)]
    pivot_strength, pivot_center, pivot_atime, _, _ = zip2arr(l)

##    l = [output_support_resistance(profile, l, h, p, lb, ub) for profile, l, h, p, lb, ub in zip(profiles, lowest, highest, price, pivot_lower_bound, pivot_upper_bound)]

futures = [
#('dc', 'pp', 1, 3.75, 1),
#('dc', 'l', 5, 3.75, 2),
#('zc', 'MA', 1, 6.25, 2),
#('sc', 'bu', 2, (7.75+5.75)/2, 2),  # liqing, 2000 #

#('zc', 'SR', 1, 6.25, 2), # tang, 5000
#('dc', 'm', 1, 6.25, 2), # doupo, 2000 #
#('dc', 'p', 2, 6.25, 2), # zonglv, 5000
#('dc', 'c', 1, 3.75, 2), # yumi, 2000
#('zc', 'CF', 5, 6.25, 2),  # cotton, 12000
#
#('sc', 'ag', 1, 9.25, 3), # Ag, 3000
#('sc', 'zn', 5, 7.75, 2), # Zn, 15000
#
#('sc', 'rb', 1, (7.75+5.75)/2, 2), # luowen, 2000 #
#('dc', 'i', 0.5, 6.25, 2), # tiekuang, 300 
#('dc', 'j', 0.5, 6.25, 2), # #jiaotan, 800
#
('sc', 'ni', 10, 7.75, 2), # Ni, 60000 
]

plot_futures = [
#('dc', 'pp', 1, 3.75, 1),
#('dc', 'l', 5, 3.75, 2),
#('zc', 'MA', 1, 6.25, 2),
#('sc', 'bu', 2, (7.75+5.75)/2, 2),  # liqing, 2000 #
#
#('zc', 'SR', 1, 6.25, 2), # tang, 5000
#('dc', 'm', 1, 6.25, 2), # doupo, 2000 #
#('dc', 'p', 2, 6.25, 2), # zonglv, 5000
#('dc', 'c', 1, 3.75, 2), # yumi, 2000
#('zc', 'CF', 5, 6.25, 2),  # cotton, 12000

#('sc', 'zn', 5, 7.75, 2), # Zn, 15000

#('sc', 'rb', 1, (7.75+5.75)/2, 2), # luowen, 2000 #
#('dc', 'i', 0.5, 6.25, 2), # tiekuang, 300 
#('dc', 'j', 0.5, 6.25, 2), # #jiaotan, 800

#('sc', 'ag', 1, 9.25, 3), # Ag, 3000
('sc', 'ni', 10, 7.75, 2), # Ni, 60000 
]

base_dir = 'data/'

ns_in_minutes = [5, 10, 30, 60, 120]
#@profile
def test_output(year=2016, month_range=range(1, 6), exp_name=''):
    for exchange, commodity, tick_size, hours_per_day, n_scale in futures:
        ticks = load_ticks(exchange, commodity, year, month_range)
        name = base_dir+commodity+str(year%100)+str(month_range[0]).zfill(2)+'-'\
            +str(year%100)+str(month_range[-1]).zfill(2)+'_pivot'+exp_name+str(n_scale)
        
        s = Strategy(name, ticks, tick_size, hours_per_day, save_freq=10)#, show_freq=10)
    #    ind_xs = SRProfile(s, 0.5*60*2, int(30*60*2))
        ind1 = PivotProfile(s, 5*60*2*n_scale, int(1 * hours_per_day * 60 * 60 * 2 * 1.5))
        ind2 = PivotProfile(s, 10*60*2*n_scale, int(2 * hours_per_day * 60 * 60 * 2 * 1.5))
        ind3 = PivotProfile(s, 30*60*2*n_scale, int(5 * hours_per_day * 60 * 60 * 2 * 1.5))
        ind4 = PivotProfile(s, 60*60*2*n_scale, int(10 * hours_per_day * 60 * 60 * 2 * 1.5))
        ind5 = PivotProfile(s, 120*60*2*n_scale, int(20 * hours_per_day * 60 * 60 * 2 * 1.5))
        for ind in [
                    ind1,
                    ind2, 
                    ind3, 
                    ind4,
                    ind5
                    ]:
            ind.precompute()
            s.add_indicator(ind)
        print 'Running', name
        with Timer() as t:
            s.run()
        print 'Run took %f sec.' % t.interval
    return s

def test_merge_plot(year=2015, month_range=range(1, 13), exp_name0='', exp_name='', plot_name=''):
    for exchange, commodity, tick_size, hours_per_day, n_scale in plot_futures:
        name0 = base_dir+commodity+str(year%100)+str(month_range[0]).zfill(2)+'-'\
            +str(year%100)+str(month_range[-1]).zfill(2)+'_pivot'+exp_name0
        name = base_dir+commodity+str(year%100)+str(month_range[0]).zfill(2)+'-'\
            +str(year%100)+str(month_range[-1]).zfill(2)+'_pivot'+exp_name+str(n_scale)
#        print 'Loading', name+'.npz' 
        d = load_dict(name+'.npz')
        if 'piv_merged2' not in d:
            with Timer() as t:
                merge2(d, n_scale, ns_in_minutes)
            print 'merge took %f sec.' % t.interval
    #        return d
            if commodity != 'ni':
                d.update(load_dict(name0+'.npz'))
            
            with Timer() as t:
                np.savez_compressed(name+'.npz', **d)
            print 'Savez took %f sec.' % t.interval
        
        if commodity == 'ni':
            d['piv_merged'] = d['piv_merged2']
            d['piv30m.volatility'] = d['piv60m.volatility']
            np.savez_compressed(name+'.npz', **d)
                
        with Timer() as t:
            save_name = name + plot_name + '.png'
            piecewise_plot(d, len(month_range), vec_inds=['piv_merged', 'piv_merged2', 'piv_merged3'], save_name=save_name)
        print 'piecewise_plot took %f sec.' % t.interval
        del d
        gc.collect() 
        
def test_plot(year=2015, month_range=range(1, 13), exp_name='', plot_name=''):
    for _, commodity, _, _ in plot_futures:
        name = base_dir+commodity+str(year%100)+str(month_range[0]).zfill(2)+'-'+str(year%100)+str(month_range[-1]).zfill(2)+'_pivot'+exp_name
        d = load_dict(name+'.npz')
        with Timer() as t:
            merge(d)
        print 'merge took %f sec.' % t.interval
        
        with Timer() as t:
            np.savez_compressed(name+'.npz', **d)
        print 'Savez took %f sec.' % t.interval
        
        with Timer() as t:
            save_name = name + plot_name + '.png'
            piecewise_plot(d, len(month_range), vec_inds=['piv_merged'], inds=['pos'], save_name=save_name)
        print 'piecewise_plot took %f sec.' % t.interval
        del d
                
if __name__ == '__main__':
    year = 2016
    month_range = range(1, 6)
#    test_output(year=year, month_range=month_range, exp_name='_nscale')
    test_merge_plot(year=year, month_range=month_range, exp_name='_nscale', plot_name='')
    