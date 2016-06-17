import math
import time
import operator

import numpy as np
import scipy
from scipy import ndimage
import pylab as plt
from profilehooks import profile
import matplotlib.gridspec as gridspec

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
    if m < 60:
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
#        self.access_time_profile[:] = -np.inf
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
        
#    @profile
    def step(self):
        self.price = [self.s.history['last_price'], self.s.now - 1, 0, 0]
        if self.precomputed:
            self._precomputed_step()
        else:
            self._step()
        
        self.access_time_profile[get(self.price)] = self.s.now - 1
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
    
    def sparsify_profile(self, profile, normalize=False):
        thld = min(self.output_thlds)
#        profile = self.smoothed_pivot_profile
        levels = np.where(profile >= thld)[0]
        strengths = profile[levels]
        if normalize and strengths.shape[0] > 0:
            strengths_list = []
            for kwargs in self.sparse_lrn_configs:
                strengths_list.append(self.sparse_lrn(strengths, **kwargs))
            strengths = np.array(strengths_list).T
        return levels, strengths
    
    def output_vec(self):
        rval = OrderedDict()
        if self.s.now < self.m:
            rval[self.name] = np.zeros_like(self.saliency_vec)
            return rval
        rval[self.name] = self.smoothed_pivot_profile
        return rval
    
    def output_sparse_vec(self):
        raise NotImplementedError()
    
    def get_nearby_levels(self, price, profile, profile_range, thld):
        lowest = profile_range[0]
        highest = profile_range[1] - 1
        # resistance
        i = price
        while i < highest and (profile[i] < thld or profile[i] < profile[i - 1] or profile[i] < profile[i + 1]):
            i += 1
        R = i #if i < highest else price
        
        # support
        i = price
        while i > lowest and (profile[i] < thld or profile[i] < profile[i - 1] or profile[i] < profile[i + 1]):
            i -= 1
        S = i #if i > lowest else price
        return S, R
    
    def get_nearby_levels_sparse(self, price, profile, profile_range, thld):
        lowest = profile_range[0]
        highest = profile_range[1] - 1
        levels, strengths = profile
        levels = levels[strengths >= thld]
        strengths = strengths[strengths >= thld]
        levels = np.concatenate((levels, np.array([lowest, highest])))
        strengths = np.concatenate((strengths, np.array([profile[lowest], profile[highest]])))
        
        nearest = levels[np.abs(levels - price).argmin()]
        if nearest > price:
            left_levels = levels[levels < price]
        elif nearest < price:  
            left_levels = levels[levels > price]
        else:
            left_levels = levels[levels != price]
        second_nearest = left_levels[np.abs(left_levels - price).argmin()]
        S = min(nearest, second_nearest)
        R = max(nearest, second_nearest)
        return S, R    
       
    def get_nearby_support(self, price, profile, profile_range, thld):
        levels, strengths = profile
        levels, strengths = levels[(levels <= price) & (strengths >= thld)], \
                        strengths[(levels <= price) & (strengths >= thld)]
#        if hasattr(self, 'filter_levels') and len(levels) > 0:
#            levels, strengths = self.filter_levels(levels, strengths)
#        lowest = min(profile_range[0], max(0, price - self.volatility * 10))
        lowest = profile_range[0]
        precond = False
        if lowest not in levels:
            levels = np.append(levels, lowest)
            strengths = np.append(strengths, self.dense_profiles[0][lowest])
#            if self.dense_profiles[0][lowest] >= thld:
#                print '111', self.dense_profiles[0][lowest], '>=', thld
#                precond = True
        
        i = np.abs(levels - price).argmin()
        support = levels[i]
        strength = strengths[i]
#        if support == lowest:
#            print '222', strength
        return support, strength
       
    def get_nearby_resistance(self, price, profile, profile_range, thld):
        levels, strengths = profile
        levels, strengths = levels[(levels >= price) & (strengths >= thld)], \
                        strengths[(levels >= price) & (strengths >= thld)]
#        if hasattr(self, 'filter_levels') and len(levels) > 0:
#            levels, strengths = self.filter_levels(levels, strengths)
#        highest = max(profile_range[1], min(profile.shape[0], price + self.volatility * 10))
        highest = profile_range[1] - 1
        if highest not in levels:
            levels = np.append(levels, highest)
            strengths = np.append(strengths, self.dense_profiles[1][highest])
        
        i = np.abs(levels - price).argmin()
        resistance = levels[i]
        strength = strengths[i]
        return resistance, strength
    
    def output_default(self, price, profile_range, rval):
        # Make this indicator as "ineffective" as possible to avoid disturbing other good indicators. 
        # This turns out to be not easy! 
        rval[self.name + '.volatility'] = 0.
        rval[self.name + '.level_start'] = profile_range[0]
        rval[self.name + '.level_stop'] = profile_range[1] 
        return rval
        lowest = profile_range[0]
        highest = profile_range[1] - 1
        for thld in self.output_thlds:
            name = self.name + '_th' + str(thld)
            rval[name + '.R'] = highest
            rval[name + '.R_offset'] = 10.
            rval[name + '.R_strength'] = 0.
            rval[name + '.R_elapsed'] = 1.
            rval[name + '.S'] = lowest
            rval[name + '.S_offset'] = 10.
            rval[name + '.S_strength'] = 0.
            rval[name + '.S_elapsed'] = 1.
            rval[name + '.R/S_ratio'] = 0.
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
      
    def _output_old(self, price, profile, profile_range, thld, rval):
        raise NotImplementedError()
        name = self.name + '_th' + str(thld)
        
        if type(profile) == np.ndarray: 
            S, R = self.get_nearby_levels(price, profile, profile_range, thld)
        else:
            assert type(profile) in [list, tuple] and len(profile) == 2 # sparse profile
            S, R = self.get_nearby_levels_sparse(price, profile, profile_range, thld)
            
        R_offset = (R - price) * 1. / self.volatility
        R_strength = profile[R] 
        R_elapsed = max(1., (self.s.now - 1 - self.access_time_profile[R]) * 1. / self.m)
        S_offset = (price - S) * 1. / self.volatility
        S_strength = profile[S]
        S_elapsed = max(1., (self.s.now - 1 - self.access_time_profile[S]) * 1. / self.m)
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
        return
    
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
        profile = self.smooth_profile(self.pivot_profile, profile_range, normalize=True)
        sparse_profile = self.sparsify_profile(profile)
#        self.sync_nearby_visit_time(sparse_profile)
        self.dense_profiles = [profile, profile]
        self.profiles = [sparse_profile, sparse_profile]
        return sparse_profile, sparse_profile
    
    def output_sparse_vec(self):
        rval = OrderedDict()
#        assert len(self.profiles) == 1
        if self.s.now < self.m:
            rval[self.name + '.levels'] = np.array([], dtype='int32')
            rval[self.name + '.strengths'] = np.array([], dtype='float32')
            return rval
        levels, strengths = self.profiles[0]
        rval[self.name + '.levels'] = np.vstack((np.ones_like(levels) * (self.s.now - 1), levels)).astype('int32').T
        rval[self.name + '.strengths'] = strengths.astype('float32')
        return rval
    
    def sync_nearby_visit_time(self, profile):
        levels, strengths = profile
        if len(levels) == 0:
            return
#        dlevels = levels - np.roll(levels, 1)
#        dlevels[dlevels < 0] = -1
#        self.dlevels = np.concatenate([self.dlevels, dlevels])
        
        split_idx = np.where(levels - np.roll(levels, 1) != 1)[0]
        self.split_stats.append((len(levels), len(split_idx)))
        split_idx = np.append(split_idx, levels.shape[0])
        for i in range(split_idx.shape[0] - 1):
            level_cluster = levels[split_idx[i] : split_idx[i + 1]]
            self.access_time_profile[level_cluster] = self.access_time_profile[level_cluster].max() 
        
    def get_recent_levels(self, profile, profile_range, thld):
        levels, strengths = profile
        levels = levels[strengths >= thld]
        lowest = profile_range[0]
        if lowest not in levels:
            levels = np.append(levels, lowest)
        highest = profile_range[1] - 1
        if highest not in levels:
            levels = np.append(levels, highest)
#        if len(levels) == 0:
#            return None, None
        vtimes = self.access_time_profile[levels]
        latest = vtimes.max()
        latest_levels = levels[vtimes == latest]
        if len(vtimes[vtimes < latest]) == 0:
            assert False
#            return latest_levels, None
        second_latest = vtimes[vtimes < latest].max()
        second_latest_levels = levels[vtimes == second_latest]
        return latest_levels, second_latest_levels
    
#    def output_default(self, price, profile_range, rval):
#        super(PivotProfile, self).output_default(price, profile_range, rval)
#        for thld in self.output_thlds:
#            name = self.name + '_th' + str(thld)
#            rval[name + '.R_latest'] = False
#            rval[name + '.R_2ndlatest'] = False
#            rval[name + '.S_latest'] = True  # trick to satisfy asserter0
#            rval[name + '.S_2ndlatest'] = False
#            rval[name + '.latestlevel'] = rval[name + '.S']
#            rval[name + '.2ndlatestlevel'] = rval[name + '.S'] 
#        return rval
#        
#    def _output(self, price, profiles, profile_range, thld, rval):
#        super(PivotProfile, self)._output(price, profiles, profile_range, thld, rval)
#        name = self.name + '_th' + str(thld)
#        R = rval[name + '.R']
#        S = rval[name + '.S']
#        rval[name + '.R_latest'] = False
#        rval[name + '.R_2ndlatest'] = False
#        rval[name + '.S_latest'] = False
#        rval[name + '.S_2ndlatest'] = False
#        latest_levels, second_latest_levels = self.get_recent_levels(profiles[0], profile_range, thld)
#        rval[name + '.latestlevel'] = latest_levels.mean()
#        rval[name + '.2ndlatestlevel'] = second_latest_levels.mean()
#        if latest_levels is None:
#            assert False
##            return rval
##        self.asserters[0].soft_assert(S in latest_levels or R in latest_levels)
#        if S in latest_levels:
#            rval[name + '.S_latest'] = True
#        if R in latest_levels:
#            rval[name + '.R_latest'] = True
#        if second_latest_levels is None:
#            assert False
##            return rval
#        if S in second_latest_levels:
#            rval[name + '.S_2ndlatest'] = True
#        if R in second_latest_levels:
#            rval[name + '.R_2ndlatest'] = True
##        if S != R and S not in second_latest_levels and R not in second_latest_levels:  
##            # the two level groups are on the same side
##            self.asserters[1].soft_assert((latest_levels.mean() - price) * (second_latest_levels.mean() - price) >= 0)
#        return rval
 
#    @staticmethod 
#    def output_closest_pivot(name, profile, profile_range, thld, price):
#        min_strength = 1.
#        
#        levels0, strengths0 = profile
#        levels, strengths = levels0[strengths0 >= thld], strengths0[strengths0 >= thld]
#        if len(levels) == 0:
#            lowest, highest = profile_range[0], profile_range[1] - 1
#            levels = np.array([lowest, highest])
#            closest = levels[np.abs(levels - price).argmin()]
#            upper_bound = lower_bound = center = closest
#            max_strength = min_strength
#        else:
#            closest = levels[np.abs(levels - price).argmin()]
#            downward_diff = levels - np.roll(levels, 1)
#            downward_diff[0] = np.Inf
#            upward_diff = np.roll(levels, -1) - levels
#            downward_diff[-1] = np.Inf
#            if closest < price:
#                upper_bound = closest
#                lower_bound = levels[(levels <= closest) & (downward_diff > max_gap)].max()
#            elif closest > price:
#                lower_bound = closest
#                upper_bound = levels[(levels >= closest) & (upward_diff > max_gap)].min()
#            else:  # closest == price
#                lower_bound = levels[(levels <= closest) & (downward_diff > max_gap)].max()
#                upper_bound = levels[(levels >= closest) & (upward_diff > max_gap)].min()
#            max_strength = strengths[(levels >= lower_bound) & (levels <= upper_bound)].max()
#            center = int(levels[(levels >= lower_bound) & (levels <= upper_bound) & (strengths == max_strength)].mean().round())
#            
#        for distance in distances:
#            upper_range = (upper_bound + 1, upper_bound + 1 + distance)
#            lower_range = (lower_bound - 1, lower_bound - 1 - distance)
#            upper_strengths = strengths0[(levels >= upper_range[0]) & (levels < upper_range[1])]
#            upper_max_strength = upper_strengths.max() if len(upper_strengths) > 0 else min_strength 
#            lower_strengths = strengths0[(levels >= lower_range[0]) & (levels < lower_range[1])]
#            lower_max_strength = lower_strengths.max() if len(lower_strengths) > 0 else min_strength  
#            
#        return upper_bound, lower_bound, center, max_strength
#    
#    @staticmethod
#    def output_support_resistance(name, profile, thld, price):
#        levels, strengths = profile
#        levels, strengths = levels[(levels >= price) & (strengths >= thld)], \
#                        strengths[(levels >= price) & (strengths >= thld)]
#        lowest = profile_range[0]
#        if lowest not in levels:
#            levels = np.append(levels, lowest)
#            strengths = np.append(strengths, self.dense_profiles[0][lowest])
#        highest = profile_range[1] - 1
#        if highest not in levels:
#            levels = np.append(levels, highest)
#            strengths = np.append(strengths, self.dense_profiles[1][highest])
#        
#        i = np.abs(levels - price).argmin()
#        resistance = levels[i]
#        strength = strengths[i]
#        return resistance, strength
#        return support, resistance, support_strength, resistance_strength
    
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

def piecewise_plot(d, n_pieces, vec_inds=[], inds=[], conds=[], cond_colors=['cyan', 'yellow'], xtick=10*60*2, ytick=None, stride=10, save_name=None):
    total_len = d['step'].shape[0]
    piece_len = int(math.ceil(total_len * 1. / n_pieces))
    for i, start in enumerate(range(0, total_len, piece_len)):
#        if i > 0:
#            return
        stop = min(start + piece_len, total_len)
        save_name_i = save_name.replace('.png', '.part' + str(i) + '.png') if save_name is not None else None
        plot(d, vec_inds=vec_inds, inds=inds, conds=conds, cond_colors=cond_colors, xtick=xtick, ytick=ytick, stride=stride, save_name=save_name_i,
             start=start, stop=stop)
    
def plot(d, vec_inds=[], inds=[], conds=[], cond_colors=['cyan', 'yellow'], xtick=10*60*2, ytick=None, stride=10, save_name=None, start=None, stop=None):
    if ytick is None:
        ytick = d['piv30m.volatility'].mean()
    if start is None:
        start = 0
    if stop is None:
        stop = d['step'].shape[0]
#    print start, stop
    step = d['step'][start:stop]
    price = d['price'][start:stop]
    
    nplots = 1 + 1
    height_ratios=[2] + [1] * 1
    dpi = 1000
    width = min(30000, step.shape[0]) / dpi
    height = sum(height_ratios) 
    figsize=(width, height)
    print 'figsize =', figsize
    plt.figure(figsize=figsize)
    
    gs = gridspec.GridSpec(nplots, 1, height_ratios=height_ratios)
    i = 0
    ax0 = plt.subplot(gs[i])
    i += 1
    
    ax0.set_xlim([step[0], step[-1]])
    span = price.max() - price.min()
    ax0.set_ylim([price.min() - span * 0.1, price.max() + span * 0.1])
    for ind in inds:
        if ind.lower().startswith('macd') or ind.endswith('latestlevel') or ind.endswith('2ndlatestlevel') or ind.endswith('R') or ind.endswith('S') :
            plt.plot(step, d[ind][start:stop], alpha=1, linewidth=.1, zorder=3, label=ind)
    plt.plot(step, price, color='k', linewidth=.1, zorder=3)
    
    ax0.set_xticks(np.arange(step[0], step[-1], xtick))
    ax0.set_yticks(np.arange(price.min(), price.max(), ytick))
    ax0.xaxis.grid(b=True, color='gray', linestyle='-', linewidth=.05, alpha=.5, zorder=0)
    ax0.yaxis.grid(b=True, color='gray', linestyle='-', linewidth=.05, alpha=.5, zorder=0)
#    ax0.yaxis.grid(b=False)
            
    opening = d['time_in_ticks'] < np.roll(d['time_in_ticks'], 1)
    opening = opening[start:stop]
    for x in step[opening]:
        plt.axvline(x, color='k', linewidth=.1, alpha=1, zorder=2)
        
    plot_configs = [
                   [((0., 1.), 'cyan', 0.1), 
                   ((1., 1.5), 'lime', 0.1, ), 
                   ((1.5, np.inf), 'green', 0.1)],
                    
                   [((0., 1.), 'yellow', 0.1), 
                   ((1., 1.5), 'red', 0.1, ), 
                   ((1.5, np.inf), 'purple', 0.1)],
                    
#                   [((1.5, 2.), 'cyan', 0.05),
#                    ((2., 2.5), 'turquoise', 0.05), 
#                   ((2.5, 3.), 'lime', 0.1, ), 
#                   ((3., 3.5), 'limegreen', 0.1, ),
#                   ((3.5, 5.), 'green', 0.15), 
#                   ((5., np.inf), 'darkolivegreen', 0.15)],
#                    
#                   [((1.5, 2.), 'yellow', 0.05),
#                    ((2., 2.5), 'gold', 0.05),  
#                   ((2.5, 3.), 'darkorange', 0.15, ),
#                   ((3., 3.5), 'red', 0.15, ),
#                   ((3.5, 5.), 'purple', 0.2), 
#                   ((5., np.inf), 'indigo', 0.2)],
                   ]
    
    with Timer() as t:
        for j, ind in enumerate(vec_inds):
            l = d[ind + '.levels']
            s = d[ind + '.strengths'][:,3]
            plot_config = plot_configs[j]
#            colors = colors[1:2]
#            plot_config = plot_config[1:2]
            for (low, high), color, alpha in plot_config:
                x, y = l[(l[:,0] >= step[0]) & (l[:,0] < step[-1]) & (s >= low) & (s < high)].T
                plt.scatter(x, y, marker='_', s=0.1, linewidth=0.1, color=color, alpha=alpha)
    print 'plt.scatter took %f sec.' % t.interval
        
    for cond, color in zip(conds, cond_colors):    
        cond = cond[start:stop]
        with Timer() as t:
    #        xsignals = np.where(cond)[0] 
            cond_steps = step[cond]
            for x in cond_steps:
                plt.axvline(x, color=color, linewidth=.1, alpha=0.4, zorder=0)
        print 'plt.axvline took %f sec.' % t.interval
    plt.legend(loc='upper right', fontsize='xx-small')
        
    inds = [ind for ind in inds if not (ind.lower().startswith('macd') or ind.endswith('R') or ind.endswith('S'))]
    for ind in inds:
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
        print 'plt.savefig took %f sec.' % t.interval
    else:
        plt.show()

def load_profiles(d, name, thld=None):
    levels = d[name + '.levels'][:,1]
    strengths = d[name + '.strengths']
    starts = d[name + '.levels.starts']
    stops = d[name + '.levels.stops']
    profiles = [(levels[start:stop], strengths[start:stop]) for (start, stop) in zip(starts, stops)]
    return profiles

max_gap = 2
def diff_levels(profile):
    levels, strengths = profile
    diffs = np.diff(levels)
    diffs = diffs[diffs > max_gap]
    diff_sum = diffs.sum()
    ndiff = len(diffs)
    return diff_sum, ndiff
#from build_tick_dataset import futures
futures = [
('dc', 'pp', 1, 3.75),
#('dc', 'l', 5, 3.75),
#('zc', 'MA', 1, 6.25),
#('zc', 'TA', 2, 6.25),
#
#('zc', 'SR', 1, 6.25), # tang, 5000
#('dc', 'm', 1, 6.25), # doupo, 2000
###('zc', 'RM', 1, 6.25), # caipo, 2000
###('dc', 'y', 2, 6.25), # douyou, 5000
#('dc', 'p', 2, 6.25), # zonglv, 5000
###('dc', 'c', 1, 3.75), # yumi, 2000
###('dc', 'cs', 1, 3.75), # dianfen, 2000
#('zc', 'CF', 5, 6.25 ),  # cotton, 12000
#
#('sc', 'ag', 1, 9.25), # Ag, 3000
##('sc', 'cu', 10, 7.75), # Cu, 30000
#('sc', 'zn', 5, 7.75), # Zn, 15000
##('sc', 'al', 5, 7.75), # Al, 12000
#('sc', 'ni', 10, 7.75), # Ni, 60000
]

base_dir = 'data/pp/'

def test_output(year=2015, month_range=range(1, 13), name=''):
    for exchange, commodity, tick_size, hours_per_day in futures:
        ticks = load_ticks(exchange, commodity, year, month_range)   
        name = base_dir+commodity+str(year%100)+str(month_range[0]).zfill(2)+'-'+str(year%100)+str(month_range[-1]).zfill(2)+'_pivot'+name
        s = Strategy(name, ticks, tick_size, hours_per_day, save_freq=10)#, show_freq=10)
    #    ind_xs = SRProfile(s, 0.5*60*2, int(30*60*2))
        ind_s = PivotProfile(s, 5*60*2, int(1 * hours_per_day * 60 * 60 * 2 * 1.5))
        ind_m = PivotProfile(s, 30*60*2, int(5 * hours_per_day * 60 * 60 * 2 * 1.5))
        for ind in [
                    ind_m,
                    ind_s, 
    #                ind_xs,
                    ]:
            ind.precompute()
            s.add_indicator(ind)
        s.run()
    return s

def test_plot(year=2015, month_range=range(1, 13), name='', plot_name=''):
    for _, commodity, _, _ in futures:
        name = base_dir+commodity+str(year%100)+str(month_range[0]).zfill(2)+'-'+str(year%100)+str(month_range[-1]).zfill(2)+'_pivot'+name
        d = load_dict(name+'.npz')
        save_name = name + plot_name + '.png'
        piecewise_plot(d, len(month_range), vec_inds=['piv30m', 'piv5m'], inds=['pos'], save_name=save_name)
        del d
        
if __name__ == '__main__':
    year = 2015
    month_range = range(1, 13)
    name = '+range'
    test_output(year=year, month_range=month_range, name=name)
#    test_plot(month_range=month_range, name=name, plot_name='_3')
    
#d = load_dict('pp1506-1509_pivot_stride10.npz')
#roff = d['piv5m_th2.5.R_offset']
#soff = d['piv5m_th2.5.S_offset']
#roff2 = d['piv30m_th2.5.R_offset']
#soff2 = d['piv30m_th2.5.S_offset']
#rlat = d['piv5m_th2.5.R_latest'].astype('bool')
#slat = d['piv5m_th2.5.S_latest'].astype('bool')
#r2lat = d['piv5m_th2.5.R_2ndlatest'].astype('bool')
#s2lat = d['piv5m_th2.5.S_2ndlatest'].astype('bool')
#r = d['piv5m_th2.5.R']
#s = d['piv5m_th2.5.S']
#rstr = d['piv5m_th2.5.R_strength']
#sstr = d['piv5m_th2.5.S_strength']
#rstr1 = d['piv0.5m_th2.5.R_strength']
#sstr1 = d['piv0.5m_th2.5.S_strength']
#cond_bounce = (roff > 0) & (soff > 0) & (rstr >= 2.5) & (sstr >= 2.5)
#cond_bounceup = cond_bounce & slat & r2lat & (roff / soff >= 4) & (roff2 >= 1) & (roff >= 2) & (sstr1 >= 2.5)
#cond_bouncedown = cond_bounce & rlat & s2lat & (soff / roff >= 4) & (soff2 >= 1) & (soff >= 2) & (rstr1 >= 2.5)
#piecewise_plot(d, 4, vec_inds=['piv5m', 'piv0.5m'], cond=cond, save_name='pp1506-1509_bounceup.png')
#dc = load_dict('pp1506-1509_common_stride10.npz')
#y = d['ec5m.orig']
#y = dc['ec5m.orig']
#y2 = dc['ec10m.orig']
#(y[cond] > 0).mean()
#(y2[cond] > 0).mean()
#y[cond].mean()
#y2[cond].mean()
