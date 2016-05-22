import time
import numpy as np
import pylab as plt
import matplotlib.gridspec as gridspec
from profilehooks import profile

from build_tick_dataset import *

from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

class EMA(object):
    def __init__(self, n):
        self.__dict__.update(locals())
        del self.self
        self.alpha = 2./(self.n + 1.)
        self.sum = 0.
        self.sum_cnt = 0
           
    def reset(self):
        self.sum = 0.
        self.sum_cnt = 0
        
    def step(self, x):
        if self.n == 1:
            return x
    
        if self.sum_cnt < self.n:
            self.sum += x
            self.sum_cnt += 1
            self.ema = self.sum * 1. / self.sum_cnt
#            if type(self.ema) != type(x):
#                print '1', self.sum_cnt, type(self.ema), type(x)
        else:
            self.ema = self.ema * (1. - self.alpha) + x * self.alpha
#            if type(self.ema) != type(x):
#                print '2', self.sum_cnt, type(self.ema), type(x)
        return self.ema
    
class SaliencyEMA():
    def __init__(self, 
                 simulator,
                 n = 60 * 60 * 2,  # 1 hour
                 stdev_coef = 1.5,
                 days = 5,
                 trading_hours_per_day = 3.75,
                 tick_size = 1,
                 smooth_n = 3
                 ):
        self.__dict__.update(locals())
        del self.self
        
        self.price = self.simulator.history['last_price'].astype('int')
        self.vol = self.simulator.history['vol']
        self.m = int(days * trading_hours_per_day * 60 * 60 * 2)
        self.w = np.ones(self.n + 1) / float(self.n + 1)
        self.y = self.price - np.convolve(self.price, self.w, mode='same')
        denorm = np.sqrt(np.convolve(self.y**2, self.w, mode='same'))
        denorm_mean = denorm.mean()
        denorm2 = (denorm > denorm_mean) * denorm + (denorm <= denorm_mean) * denorm_mean
#        self.ny = self.y
#        self.ny = (denorm > 0) * self.y / denorm
        self.ny = (denorm2 > 0) * self.y / denorm2
        self.ny_mean_EMA = EMA(self.m * 1. / self.n)
        self.ny_stdev_EMA = EMA(self.m * 1. / self.n)
        
        self.start_tick = self.n * 4
        self.saliency = np.zeros(self.price.max() + self.simulator.stopprofit + 10)
        self.saliency2 = np.zeros(self.price.max() + self.simulator.stopprofit + 10)
        self.saliency_EMA = EMA(self.m)
        self.saliency2_EMA = EMA(self.m)
        self.mean_saliency_EMA = EMA(self.m * 1. / self.n)
        
        assert self.smooth_n % 2 == 1
        self.smooth_w = np.ones((self.smooth_n - 1) * self.tick_size + 1) / self.smooth_n 
    
    def normalize(self, saliency_ema, level, name=None): 
        mean_saliency = saliency_ema.mean() * self.tick_size
        if self.saliency_EMA.sum_cnt == self.m and saliency_ema[level] > mean_saliency * 5. and \
                saliency_ema[level] > saliency_ema[level - self.tick_size] * 5.and \
                saliency_ema[level] > saliency_ema[level + self.tick_size] * 5.:
            if name is not None:
                print name
#            print 'smooth out abnormal saliency at', level, \
#                'mean_saliency =', mean_saliency, 'saliency =', saliency_ema[level]
            saliency_ema[level] = mean_saliency
            
    def smooth(self, saliency_ema):
        return np.convolve(saliency_ema, self.smooth_w, mode='same')
    
    def smooth_all(self):
        self.smoothed_saliency_ema = self.smooth(self.saliency_ema)
#        self.smoothed_saliency2_ema = self.smooth(self.saliency2_ema)
     
    def get_max_of_max(self, sr_level):
        low = sr_level
        high = sr_level
        while self.smoothed_saliency_ema[low] >= self.mean_saliency_ema * self.simulator.openable_normalized_saliency:
            low -= 1
        while self.smoothed_saliency_ema[high] >= self.mean_saliency_ema * self.simulator.openable_normalized_saliency:
            high += 1
        max_saliency = self.smoothed_saliency_ema[low : high].max()
        sr_level = self.smoothed_saliency_ema[low : high].argmax() + low
        return sr_level, max_saliency
    
    def get_nearby_sr_level(self, distance):
        distance *= self.tick_size
        self.smooth_all()
        last_price = self.price[self.simulator.now - 1]
        nearby_saliency = self.smoothed_saliency_ema[last_price - distance : last_price + distance + 1]
        max_saliency = nearby_saliency.max()
        sr_level = nearby_saliency.argmax() + last_price - distance
        if max_saliency >= self.mean_saliency_ema * self.simulator.openable_normalized_saliency:
            sr_level, max_saliency = self.get_max_of_max(sr_level)
        return sr_level, max_saliency 
           
    def step(self):
        if self.simulator.now < self.start_tick:
            return None 
        
        self.saliency *= 0.
        self.saliency2 *= 0.
        if (self.simulator.now - self.start_tick) % self.n == 0: 
            mean = np.mean(self.ny[self.simulator.now - self.start_tick : self.simulator.now])
            self.mean = self.ny_mean_EMA.step(mean)
            stdev = np.sqrt(np.var(self.ny[self.simulator.now - self.start_tick : self.simulator.now]))
            self.stdev = self.ny_stdev_EMA.step(stdev)
                
#        mean = self.ny_mean_EMA.ema
#        stdev = self.ny_stdev_EMA.ema
        ny = self.ny[self.simulator.now - self.n]
        ny = ny * (abs(ny - self.mean) > self.stdev * self.stdev_coef)
        
        level = self.price[self.simulator.now - self.n]
        self.saliency[level] = abs(ny)
        self.saliency_ema = self.saliency_EMA.step(self.saliency)
        
        if self.simulator.history['time_in_ticks'][self.simulator.now - 1] == 0 and \
                abs(self.price[self.simulator.now - 1] - self.price[self.simulator.now - 2]) > 400:
            self.saliency_EMA.reset()
            self.saliency_ema *= 0.
            print 'Reset saliency on month change.' 
        
        if (self.simulator.now - self.start_tick) % self.n == 0:  
            floor = self.price[self.simulator.now - self.start_tick : self.simulator.now].min()
            ceil = self.price[self.simulator.now - self.start_tick : self.simulator.now].max()
            mean_saliency = self.saliency_ema[floor : ceil + 1].mean() 
            self.mean_saliency_ema = self.mean_saliency_EMA.step(mean_saliency)
        
#        self.saliency2[level] = self.saliency[level] * self.vol[self.simulator.now - self.n]
#        self.saliency2_ema = self.saliency2_EMA.step(self.saliency2)
        
        if self.simulator.now % 360 == 0:
            self.normalize(self.saliency_ema, level)
#            self.normalize(self.saliency2_ema, level, 'saliency2_ema')
        return self.saliency_ema #, self.saliency2_ema
    
class IntensityEMA():
    def __init__(self, 
                 simulator,
                 days = 5,
                 trading_hours_per_day = 3.75,
                 tick_size = 1,
                 smooth_n = 3
                 ):
        self.__dict__.update(locals())
        del self.self
        
        self.price = self.simulator.history['last_price'].astype('int')
        self.vol = self.simulator.history['vol']
        self.m = int(days * trading_hours_per_day * 60 * 60 * 2)
        
        self.intensity = np.zeros(self.price.max() + 1)
        self.intensity_ema = np.zeros(self.price.max() + 1)
        self.intensity_sum = 0
        self.stay = np.zeros(self.price.max() + 1)
        self.stay_ema = np.zeros(self.price.max() + 1)
        self.stay_sum = 0
        self.sum_cnt = 0
        self.intensity_ema2 = np.zeros(self.price.max() + 1)
        self.alpha = 2. / (self.m + 1)
        
        assert self.smooth_n % 2 == 1
        self.smooth_w = np.ones((self.smooth_n - 1) * self.tick_size + 1) / self.smooth_n 
            
    def smooth(self, ema):
        return np.convolve(ema, self.smooth_w, mode='same')
    
    def smooth_all(self):
        self.smoothed_intensity_ema = self.smooth(self.intensity_ema)
        self.smoothed_intensity_ema2 = self.smooth(self.intensity_ema2)
        
    def step(self):
        self.intensity *= 0.
        self.stay *= 0.
        t = self.simulator.now - 1
        level = self.price[t]
        self.intensity[level] = self.vol[t]
        self.stay[level] = 1.
        if self.sum_cnt < self.m:
            self.intensity_sum += self.intensity
            self.stay_sum += self.stay
            self.sum_cnt += 1
            self.intensity_ema = self.intensity_sum * 1. / self.sum_cnt
            self.stay_ema = self.stay_sum * 1. / self.sum_cnt
        else:
            self.intensity_ema = self.intensity_ema * (1 - self.alpha) + self.intensity * self.alpha
            self.stay_ema = self.stay_ema * (1 - self.alpha) + self.stay * self.alpha
        self.intensity_ema2 *= 0.
        self.intensity_ema2[self.stay_ema > 0] = (self.intensity_ema / self.stay_ema)[self.stay_ema > 0]
        
#        if self.simulator.now % 100 == 0:
#            self.normalize(self.saliency_ema, level)
#            self.normalize(self.saliency2_ema, level, 'saliency2_ema')
        return self.intensity_ema, self.intensity_ema2
    
OPENED = 2
PLACED = 1
CLOSED = 0
class Simulator():
    def __init__(self, history, step_size=1, 
                 show_freq = 15 * 60 * 2, 
                 zoomin_show_freq = 30 * 60 * 2,
                 zoomin_show_len = 5 * 3.75 * 60 * 60 * 2,
                 saliency_scale = 7*1e7,
                 openable_normalized_saliency=2.,
                 closable_normalized_saliency=2.,
                 intensity_scale = 1*1e6,
                 intensity2_scale = 3*1e2,
                 tick_size=1,
                 trading_hours_per_day = 3.75,
                 stoploss=6,
                 stopprofit=30,
                 sleeptime_on_action = 0
                 ):
        self.__dict__.update(locals())
        del self.self
        self.stoploss *= self.tick_size
        self.stopprofit *= self.tick_size
        self.ticks_per_day = int(self.trading_hours_per_day * 60 * 60 * 2)
        
        self.now = 0
        self.indicators = []
        
        self.sr_level = None
        self.last_sr_level = None
        self.stopprofit_level = None
        self.stoploss_level = None
        self.stopprofit_price = None
        self.stoploss_price = None
        self.pending_open_level = None
        self.open_level = None
        self.open_price = None
        self.state = CLOSED
        self.gains = []
        self.total_gain = 0
        self.equity_curve = np.zeros_like(self.history['last_price'])
        self.figure_cnt = 0
        
    def add_indicator(self, indicator):
        self.indicators.append(indicator)
         
    @profile   
    def step(self, step_size=1):
        self.now += step_size
        for indicator in self.indicators:
            indicator.step()
    
        if self.now >= self.indicators[0].m and self.now % self.show_freq == 0:
            self.show()
            
#    @profile
    def do(self):
        if self.now - self.indicators[0].m < self.indicators[0].n:
            return
        
        if self.state == OPENED:
            if not self.try_early_quit():
                if self.try_stop_loss():
                    self.try_place(after_stop_loss=True)
                else:
                    self.try_stop_profit()
        elif self.state == PLACED:
            self.try_open()
        else:
            assert self.state == CLOSED
            self.try_place()
            
        self.equity_curve[self.now - 1] = self.total_gain
            
        if self.now % self.zoomin_show_freq == 0: # and self.state != CLOSED:
            self.zoomin_show()
        
    def try_early_quit(self):
        last_price = self.history['last_price'][self.now - 1]
        if self.now == self.history['last_price'].shape[0] or \
                self.history['time_in_ticks'][self.now] == 0 and \
                abs(self.history['last_price'][self.now] - last_price) > 400: # change month
            print self.figure_cnt, 'early quit at', last_price
            if (last_price - self.open_price) * (self.stoploss_level - self.open_price) > 0:
                self.stoploss_price = last_price
                self.close(False)
            else:
                self.stopprofit_price = last_price
                self.close(True)
            return True
        return False
    
    def try_stop_loss(self):
        last_price = self.history['last_price'][self.now - 1]
        if (last_price - self.open_price) * (self.stoploss_level - self.open_price) > 0 and \
                abs(last_price - self.open_price) >= abs(self.stoploss_level - self.open_price):
            self.stoploss_price = last_price
            self.last_sr_level = self.sr_level
            
#            if abs(self.stoploss_price - self.stoploss_level) > abs(self.open_level - self.stoploss_level):
            if abs(self.stoploss_price - self.open_price) > abs(self.stoploss_level - self.open_level) * 2.:
                print 'time_in_ticks =', int(self.history['time_in_ticks'][self.now - 1]), '/', self.ticks_per_day, 
                print 'loss =', abs(self.stoploss_price - self.open_price),
                print 'open_exceed =', abs(self.open_price - self.open_level), 'stoploss_exceed =', abs(self.stoploss_price - self.stoploss_level)
            
            self.close(False)
            return True
        return False
    
    def try_stop_profit(self):
        last_price = self.history['last_price'][self.now - 1]
        sr_level, saliency = self.indicators[0].get_nearby_sr_level(self.stoploss/2)
        if abs(last_price - self.sr_level) < self.stopprofit:
            if abs(last_price - self.sr_level) >= self.stoploss and \
                    abs(sr_level - self.sr_level) > self.stoploss and \
                    saliency >= self.indicators[0].mean_saliency_ema * self.closable_normalized_saliency and \
                    saliency > self.indicators[0].smoothed_saliency_ema[self.sr_level]:
#                self.stopprofit_level = sr_level
                self.stopprofit_price = last_price
                self.close(True)
                return True
            else:
                return False
        else:
            if saliency >= self.indicators[0].mean_saliency_ema * 3.:
#            if True:
#                self.stopprofit_level = sr_level
                self.stopprofit_price = last_price
                self.close(True)
                return True
            else:
                return False
          
    def close(self, is_stop_profit):
        last_price = self.history['last_price'][self.now - 1]
        self.state = CLOSED
#        print self.figure_cnt, 'close at', last_price, 'is_stop_profit =', is_stop_profit
#        if not is_stop_profit:
#            print 'last_sr_level =', self.last_sr_level
        self.zoomin_show(self.sleeptime_on_action)
        
        gain = abs(last_price - self.open_price)
        if not is_stop_profit:
            gain *= -1.
#        if gain < 0 and abs(gain) > self.stoploss * 1.5 * 2:
#            print gain, '->', -self.stoploss * 1.5 * 2
#            gain = -self.stoploss * 1.5 * 2
#        else:
#            print gain
        if gain % self.tick_size != 0:
            print gain, self.tick_size
            assert False
        gain /= self.tick_size
        self.gains.append(gain)
        self.total_gain += gain 
        
        self.sr_level = None
        self.open_level = None
        self.stoploss_level = None
        self.stopprofit_level = None
        self.open_price = None
        self.stoploss_price = None
        self.stopprofit_price = None
          
    def try_place(self, after_stop_loss=False): 
#        if self.history['time_in_ticks'][self.now - 1] < 3 * 60 * 2 or \
#                self.history['time_in_ticks'][self.now - 1] > self.ticks_per_day - 15 * 60 * 2:
#            return False
        if after_stop_loss:
            assert self.last_sr_level is not None
            last_price = self.last_sr_level
            self.last_sr_level = None
#            print self.figure_cnt, 'Try to place after stop loss at', last_price
        else:
            last_price = self.history['last_price'][self.now - 1]
        sr_level, saliency = self.indicators[0].get_nearby_sr_level(self.stoploss)
        if saliency >= self.indicators[0].mean_saliency_ema * self.openable_normalized_saliency and abs(last_price - sr_level) < self.stoploss:
            self.sr_level = sr_level
            self.pending_open_level = [self.sr_level - self.stoploss, self.sr_level + self.stoploss]
            self.state = PLACED
#            if after_stop_loss:
#                print self.figure_cnt, 'Place after stop loss at', last_price
            self.zoomin_show(self.sleeptime_on_action)
            return True
        return False
             
    def try_open(self): 
        last_price = self.history['last_price'][self.now - 1]
        if abs(last_price - self.sr_level) >= self.stoploss:
            self.pending_open_level = None
            if last_price > self.sr_level:
                self.open_level = self.sr_level + self.stoploss
                self.stoploss_level = self.sr_level - self.stoploss / 2. 
            else:
                self.open_level = self.sr_level - self.stoploss
                self.stoploss_level = self.sr_level + self.stoploss / 2. 
            self.open_price = last_price
            self.state = OPENED
            self.zoomin_show(self.sleeptime_on_action)
            
            if abs(self.open_price - self.open_level) > abs(self.open_level - self.stoploss_level):
                print 'time_in_ticks =', int(self.history['time_in_ticks'][self.now - 1]), '/', self.ticks_per_day, 
                print 'open_exceed =', abs(self.open_price - self.open_level)
            return True
        return False
                     
    def showable(self):
        return self.now - self.indicators[0].m >= self.indicators[0].n and self.now % self.show_freq == 0
      
    def zoomin_show(self, sleeptime=0.0):
        self.figure_cnt += 1
        return
        self.indicators[0].smooth_all()
        
#        plt.ion()
        plt.clf()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
#        gs = gridspec.GridSpec(1, 1)
        
        y = self.indicators[0].smoothed_saliency_ema
        y_mean = self.indicators[0].mean_saliency_ema
        show_len = self.zoomin_show_len
        
        scale = self.saliency_scale
        scale = scale * show_len * 1. / self.indicators[0].m
            
        ax0 = plt.subplot(gs[0])
        price = self.history['last_price'][self.now - show_len : self.now]
        plt.plot(price)
        if self.sr_level is None:
            floor = price.min() - 1
            ceil = price.max() + 1
        else:
            floor = min(self.sr_level - self.stopprofit, price.min()) - 1
            ceil = max(self.sr_level + self.stopprofit, price.max()) + 1
        
        day_begin = np.where(self.history['time_in_ticks'][self.now - show_len : self.now] == 0)[0]
        for x in day_begin:
            plt.axvline(x, color='k', linestyle=':')
        
        if self.sr_level is not None:
            plt.plot(np.ones_like(price) * self.sr_level, color='k')
            if self.state == PLACED:
                assert self.pending_open_level is not None
                plt.plot(np.ones_like(price) * (self.pending_open_level[0]), '--', color='k', label='pending open')
                plt.plot(np.ones_like(price) * (self.pending_open_level[1]), '--', color='k', label='pending open')
            else:
                assert self.state == OPENED or self.state == CLOSED
                assert self.open_level is not None
                assert self.stoploss_level is not None
                assert self.open_level != self.sr_level
                assert self.stoploss_level != self.sr_level
                plt.plot(np.ones_like(price) * self.open_level, '--', color='k', label='open')
                plt.plot(np.ones_like(price) * self.stoploss_level, '--', color='r', label='stoploss0')
                if self.open_level > self.sr_level:
                    stopprofit_level0 = self.sr_level + self.stopprofit
                else:
                    stopprofit_level0 = self.sr_level - self.stopprofit
                plt.plot(np.ones_like(price) * stopprofit_level0, '--',
                         color='g', label='stopprofit0')
                if self.state == CLOSED:
                    if self.stopprofit_price is not None:
                        plt.plot(np.ones_like(price) * self.stopprofit_price, color='g', label='stopprofit')
                    if self.stoploss_price is not None:
                        plt.plot(np.ones_like(price) * self.stoploss_price, color='r', label='stoploss')
            
        y = y[floor : ceil + 1] * scale
#        y[y > show_len * 1.2] = show_len * 1.2
        plt.barh(np.arange(floor, ceil + 1) - 0.5, y, np.ones_like(y),
                 alpha=0.2, color='r', edgecolor='none')
        if y_mean is not None:
            y_mean = int(y_mean * 2. * scale)
            ax0.set_xticks(np.arange(0, show_len, y_mean))
#            ax0.set_yticks(np.arange(floor, ceil + 1, 1000)) # remove horizontal grid
            for xmaj in ax0.xaxis.get_majorticklocs():
                ax0.axvline(x=xmaj,ls='-', color='r')
            
#        plt.grid(color='r', linestyle='-')
#        plt.legend(loc='upper right')
        
        ax = plt.subplot(gs[1], sharex=ax0)
        pos = self.history['pos'][self.now - show_len : self.now]
        plt.plot(pos, label='pos')
        plt.grid()
        plt.legend(loc='upper right')
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.savefig('figures/%d.png' % self.figure_cnt, bbox_inches='tight')
        self.figure_cnt += 1
#        plt.draw()
#        plt.show()
        time.sleep(sleeptime)
        
    def _plot_histogram(self, gs, y, scale, y_mean=None, show_len=None, label=None, sharex=None):
        if show_len is None:
            show_len = self.indicators[0].m
        else:
            scale = scale * show_len * 1. / self.indicators[0].m
            
        ax = plt.subplot(gs, sharex=sharex)
        price = self.history['last_price'][self.now - show_len : self.now]
        plt.plot(price)
        floor = price.min()
        ceil= price.max()
#        floor = self.history['last_price'].min()
#        ceil= self.history['last_price'].max()
        y = y[floor : ceil + 1] * scale
#        y[y > show_len * 1.2] = show_len * 1.2
        plt.barh(np.arange(floor, ceil + 1), y, 1.0, label=label,
                 alpha=0.2, color='r', edgecolor='none')
        if y_mean is not None:
            y_mean = int(y_mean * 2. * scale)
            ax.set_xticks(np.arange(0, show_len, y_mean))
        plt.grid()
        plt.legend(loc='upper right')
        return ax
    
    def show(self):
        for indicator in self.indicators:
            indicator.smooth_all()
        
        plt.ion()
        plt.clf()
        gs = gridspec.GridSpec(2, 2, height_ratios=[1,1])
#        gs = gridspec.GridSpec(1, 1)
        
        y = self.indicators[0].smoothed_saliency_ema
        y_mean = self.indicators[0].mean_saliency_ema
        ax0 = self._plot_histogram(gs[0], y, self.saliency_scale, y_mean=y_mean, label='saliency')
        
#        y = self.indicators[1].smoothed_intensity_ema
#        ax = self._plot_histogram(gs[1], y, self.intensity_scale, label='intensity', sharex=ax0)
        
#        y = self.indicators[2].smoothed_saliency_ema
#        ax = self._plot_histogram(gs[2], y, self.saliency_scale, label='stdev=1.5', sharex=ax0)
        
#        y = self.indicators[1].smoothed_intensity_ema2
#        ax = self._plot_histogram(gs[3], y, self.intensity2_scale, label='mean_intensity', sharex=ax0)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.draw()
        
    def finished(self):
        return self.now >= self.history['last_price'].shape[0] 
    
    def plot_equity_curve(self):
        plt.clf()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])
        ax0 = plt.subplot(gs[0])
        price = self.history['last_price']
        plt.plot(price, label='price')
        plt.legend()
        plt.grid()
        ax = plt.subplot(gs[1], sharex=ax0)
        plt.plot(self.equity_curve, label='equity_curve')
#        plt.legend()
        plt.grid()
        plt.savefig('figures/equity_curve22.png', bbox_inches='tight')
        
    def run(self):
        while not self.finished():
            self.step()
#            self.do()
#        print self.gains
#        self.gains = np.array(self.gains)
#        mean_gain = self.gains[self.gains > 0].mean()
#        mean_loss = -self.gains[self.gains < 0].mean()
#        print 'total_gain =', self.gains.sum(), 'total_trades =', self.gains.size, \
#            'gain/trade =', self.gains.mean(), 'winning rate =', (self.gains > 0).mean(), \
#            'mean_gain / mean_loss =', mean_gain, '/', mean_loss, '=', mean_gain / mean_loss 
#        self.plot_equity_curve()
#        plt.show()
#        raw_input()
                
if __name__ == '__main__':
#    ticks = load_ticks('zc', 'SR', 2015, [9,], use_cache=True); tick_size = 1; trading_hours_per_day = 6.25
#    ticks = load_ticks('zc', 'MA', 2015, [7,12], use_cache=True); tick_size = 1; trading_hours_per_day = 6.25
#    ticks = load_ticks('zc', 'TA', 2015, [11,], use_cache=True); tick_size = 2; trading_hours_per_day = 6.25
#    for month in range(7, 9):
    if True:
        ticks = load_ticks('dc', 'pp', 2015, range(9, 11), use_cache=True); tick_size = 1; trading_hours_per_day = 3.75
        s = Simulator(ticks, tick_size=tick_size, trading_hours_per_day=trading_hours_per_day)
        for indicator in [
                SaliencyEMA(s, tick_size=tick_size, trading_hours_per_day=trading_hours_per_day),
                ]:
            s.add_indicator(indicator)
        s.run()

    
