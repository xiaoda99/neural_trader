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

strength_thld = 2.5
def filter_profiles(profiles, strength_thld):
    profiles = [(l[s >= strength_thld], s[s >= strength_thld], t[s >= strength_thld]) for l, s, t in profiles]
    return profiles
    
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

def load_globals():
#    global dc, d, price, price_low, price_high, price_strength, now_step, macd, macd2, \
#        past_low, past_high, past_low_step, past_high_step, \
#        future_low, future_high, future_low_before_high, future_high_before_low, \
#        future_low2, future_high2, future_low_before_high2, future_high_before_low2, \
#        future_low4, future_high4, future_low_before_high4, future_high_before_low4, \
#        lowest, highest, volatility
    dc = load_dict('data/pp/pp1501-1512_common.npz')
    d = load_dict('data/pp/pp1501-1512_pivot.npz')
    macd = dc['macd5m']
    denorm = np.sqrt((macd**2).mean())
    macd = macd / denorm
    dc['macd5m'] = macd
    macd2 = dc['macd10m']
    denorm = np.sqrt((macd2**2).mean())
    macd2 = macd2 / denorm
    dc['macd10m'] = macd2
    return d, dc
    price, price_low, price_high = get_price(dc['price'])
    now_step = dc['step']
    past_low = dc['past5m.low']
    past_high = dc['past5m.high']
    past_low_step = dc['past5m.when_low']
    past_high_step = dc['past5m.when_high']
    future_low = dc['in10m.low']
    future_high = dc['in10m.high']
    future_low_before_high = dc['in10m.low_before_high']
    future_high_before_low = dc['in10m.high_before_low']
    future_low2 = dc['in30m.low']
    future_high2 = dc['in30m.high']
    future_low_before_high2 = dc['in30m.low_before_high']
    future_high_before_low2 = dc['in30m.high_before_low']
    future_low4 = dc['in60m.low']
    future_high4 = dc['in60m.high']
    future_low_before_high4 = dc['in60m.low_before_high']
    future_high_before_low4 = dc['in60m.high_before_low']
    lowest = d['piv5m.level_start']
    highest = d['piv5m.level_stop'] - 1
    volatility = d['piv5m.volatility']
    price_strength = d['piv5m.price_strength']
    
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

def _get_target(stoploss_level, stopprofit_level, low, high, low_before_high, high_before_low):
    assert stoploss_level != stopprofit_level
    if stoploss_level < stopprofit_level:
        if high >= stopprofit_level and low_before_high > stoploss_level:
            return 1
        elif low <= stoploss_level and high_before_low < stopprofit_level:
            return 0
        else:
            return 0.5
    else:
        if low <= stopprofit_level and high_before_low < stoploss_level:
            return 1
        elif high >= stoploss_level and low_before_high > stopprofit_level:
            return 0
        else:
            return 0.5
        
def get_target(future_low, future_high, future_low_before_high, future_high_before_low, 
               future_low2, future_high2, future_low_before_high2, future_high_before_low2,
               signal, stoploss_level, stopprofit_level, entry_price, n_pieces=12, count_by_month=False, transaction_cost=1.):
    assert np.all(stoploss_level[signal] < stopprofit_level[signal]) or np.all(stoploss_level[signal] > stopprofit_level[signal]) 
    if np.all(stoploss_level[signal] < stopprofit_level[signal]):
        stoploss = entry_price - stoploss_level 
        stopprofit = stopprofit_level - entry_price
        win = signal & (future_high >= stopprofit_level) & (future_low_before_high > stoploss_level)
        loss = signal & (future_low <= stoploss_level) & (future_high_before_low < stopprofit_level)
        unknown = signal & np.negative(win) & np.negative(loss) 
        win2 = signal & (future_high2 >= stopprofit_level) & (future_low_before_high2 > stoploss_level)
        loss2 = signal & (future_low2 <= stoploss_level) & (future_high_before_low2 < stopprofit_level)
    else:
        stoploss = stoploss_level - entry_price  
        stopprofit = entry_price - stopprofit_level
        win = signal & (future_low <= stopprofit_level) & (future_high_before_low < stoploss_level)
        loss = signal & (future_high >= stoploss_level) & (future_low_before_high > stopprofit_level)
        unknown = signal & np.negative(win) & np.negative(loss) 
        win2 = signal & (future_low2 <= stopprofit_level) & (future_high_before_low2 < stoploss_level)
        loss2 = signal & (future_high2 >= stoploss_level) & (future_low_before_high2 > stopprofit_level)
    win = win | (unknown & win2) 
    loss = loss | (unknown & loss2) 
    unknown = signal & np.negative(win) & np.negative(loss) 
    
    total_netprofit = (stopprofit * win - stoploss * loss).sum()
    total_transaction_cost = (win + loss).sum() * transaction_cost 
    mean_netprofit = total_netprofit / (win + loss).sum()
    
    total_len = signal.shape[0]
    piece_len = int(math.ceil(total_len * 1. / n_pieces))
    if count_by_month:
        for i, start in enumerate(range(0, total_len, piece_len)):
            stop = min(start + piece_len, total_len)
            print 'month', i
            signal_i = signal[start : stop]
            win_i = win[start : stop]
            unknown_i = unknown[start : stop]
            print 'num_signals =', signal_i.sum(), 'num_wins =', win_i.sum()
            print 'win_rate, unknown_rate =', win_i.mean() / signal_i.mean(), unknown_i.mean() / signal_i.mean()
#    print '\n'
    print 'num_signals =', signal.sum(), 'num_wins =', win.sum()
    print 'win_rate, loss_rate =', win.mean() / signal.mean(), loss.mean() / signal.mean(), loss.mean() / win.mean()
    print 'mean_profit / mean_loss =', ((stopprofit * win).sum() / win.sum()) / ((stoploss * loss).sum() / loss.sum())
    print 'netprofit: ', total_netprofit, '-', total_transaction_cost, '=', total_netprofit - total_transaction_cost, \
        mean_netprofit, '-', transaction_cost, '=', mean_netprofit - transaction_cost
        
    return win, loss, unknown
         
def get_by_step(ind, step, stride=10):
    return ind[(step / stride).round().astype('int32')]

def get_prev(ind):
    prev_ind = np.roll(ind, 1)
    prev_ind[0] = prev_ind[1]
    return prev_ind

profit_loss_ratio = 3.

#def bounceup2(d, dc, n):
#    past_low = dc['past'+n2str(n)+'.low']
#    past_low_step = dc['past'+n2str(n)+'.when_low']
#    past_high2 = dc['past'+n2str(n*2)+'.high']
#    now_step = dc['step']
#    macd = dc['macd'+n2str(n)]
#    macd2 = dc['macd1'+n2str(n*2)]
#    price, price_low, price_high = get_price(dc['price'])
#    support = d['piv'+n2str(n)+'.support']
#    strength = d['piv'+n2str(n)+'.strength']
#    lower_bound = d['piv'+n2str(n)+'.lower_bound']
#    upper_bound = d['piv'+n2str(n)+'.upper_bound']
#    stoploss_price = past_low - 1
#    entry_price = support + 1
#    stoploss = entry_price - stoploss_price
#    
#    stopprofit = stoploss * profit_loss_ratio
#    stopprofit_price = np.round(entry_price + stopprofit).astype('int32')
#    signal = (pivot_strength >= min_strength) & \
#        (past_low >= lower_bound - 1) & (past_low <= upper_bound) & \
#        (now_step - past_low_step <= min_dt) & \
#        (get_by_step(macd, past_low_step) <= -min_macd) & \
#        (get_by_step(macd, past_low_step) > -max_macd) & \
#        (get_by_step(macd2, past_low_step) <= -min_macd2) & \
#        (get_by_step(macd2, past_low_step) > -max_macd2) & \
#        (entry_price + stoploss * profit_loss_ratio2 < resistance) & \
#        (get_prev(price_low) <= support) & (price > support)
#    
#    abnomal_pct = (stoploss_level[signal] >= stopprofit_level[signal]).mean()
#    if abnomal_pct > 0:
#        print 'abnomal_pct =', abnomal_pct
#        stopprofit_level[signal] = np.maximum(stoploss_level[signal] + 1, stopprofit_level[signal])
#        assert (stoploss_level[signal] >= stopprofit_level[signal]).mean() == 0
#    return signal, stoploss_level, stopprofit_level, entry_price
    
    
def bounceup(pivot_strength, pivot_lower_bound, pivot_upper_bound, past_low, past_low_step, past_high, past_high_step, macd, macd2,
        price, price_low, price_high, now_step, support, resistance, min_strength=2.5, min_dt=0.5*60*2, 
        min_macd=1., max_macd=np.inf, min_macd2=0., max_macd2=np.inf,
        profit_loss_ratio=profit_loss_ratio, profit_loss_ratio2=None):
    if profit_loss_ratio2 is None:
        profit_loss_ratio2 = profit_loss_ratio
    stoploss_level = past_low - 1
    entry_price = support + 1
    stoploss = entry_price - stoploss_level
    stopprofit = stoploss * profit_loss_ratio
    stopprofit_level = np.round(entry_price + stopprofit).astype('int32')
    signal = (pivot_strength >= min_strength) & \
        (past_low >= pivot_lower_bound - 1) & (past_low <= pivot_upper_bound) & \
        (now_step - past_low_step <= min_dt) & \
        (get_by_step(macd, past_low_step) <= -min_macd) & \
        (get_by_step(macd, past_low_step) > -max_macd) & \
        (get_by_step(macd2, past_low_step) <= -min_macd2) & \
        (get_by_step(macd2, past_low_step) > -max_macd2) & \
        (entry_price + stoploss * profit_loss_ratio2 < resistance) & \
        (get_prev(price_low) <= support) & (price > support)
    
    abnomal_pct = (stoploss_level[signal] >= stopprofit_level[signal]).mean()
    if abnomal_pct > 0:
        print 'abnomal_pct =', abnomal_pct
        stopprofit_level[signal] = np.maximum(stoploss_level[signal] + 1, stopprofit_level[signal])
        assert (stoploss_level[signal] >= stopprofit_level[signal]).mean() == 0
    return signal, stoploss_level, stopprofit_level, entry_price

def bouncedown(pivot_strength, pivot_lower_bound, pivot_upper_bound, past_low, past_low_step, past_high, past_high_step, macd, macd2,
        price, price_low, price_high, now_step, support, resistance, min_strength=2.5, min_dt=0.5*60*2, 
        min_macd=1., max_macd=np.inf, min_macd2=0., max_macd2=np.inf, 
        profit_loss_ratio=profit_loss_ratio, profit_loss_ratio2=None):
    if profit_loss_ratio2 is None:
        profit_loss_ratio2 = profit_loss_ratio
    stoploss_level = past_high + 1
    entry_price = resistance - 1
    stoploss = stoploss_level - entry_price
    stopprofit = stoploss * profit_loss_ratio
    stopprofit_level = np.round(entry_price - stopprofit).astype('int32')
    signal = (pivot_strength >= min_strength) & \
        (past_high <= pivot_upper_bound + 1) & (past_high >= pivot_lower_bound) & \
        (now_step - past_high_step <= min_dt) & \
        (get_by_step(macd, past_high_step) >= min_macd) & \
        (get_by_step(macd, past_low_step) < max_macd) & \
        (get_by_step(macd2, past_low_step) >= min_macd2) & \
        (get_by_step(macd2, past_low_step) < max_macd2) & \
        (entry_price - stoploss * profit_loss_ratio2 > support) & \
        (get_prev(price_high) >= resistance) & (price < resistance)
    
    abnomal_pct = (stoploss_level[signal] <= stopprofit_level[signal]).mean()
    if abnomal_pct > 0:
        print 'abnomal_pct =', abnomal_pct
        stopprofit_level[signal] = np.minimum(stoploss_level[signal] - 1, stopprofit_level[signal])
        assert (stoploss_level[signal] <= stopprofit_level[signal]).mean() == 0
    return signal, stoploss_level, stopprofit_level, entry_price

def breakup(pivot_strength, pivot_lower_bound, pivot_upper_bound, volatility, macd, macd2,
        price, price_low, price_high, now_step, support, resistance, min_strength=2.5, min_macd=1.5, min_macd2=0., 
        min_stoploss_factor=0, profit_loss_ratio=profit_loss_ratio, profit_loss_ratio2=None):
    if profit_loss_ratio2 is None:
        profit_loss_ratio2 = profit_loss_ratio
    stoploss = np.maximum(np.ceil(volatility * min_stoploss_factor), pivot_upper_bound + 1 - pivot_lower_bound)
    stoploss_level = pivot_upper_bound
    entry_price = stoploss_level + stoploss
    stopprofit = stoploss * profit_loss_ratio
    stopprofit_level = np.round(entry_price + stopprofit).astype('int32')
    signal = (pivot_strength >= min_strength) & \
        (get_by_step(macd, now_step) >= min_macd) & \
        (get_by_step(macd2, now_step) >= min_macd2) & \
        (entry_price + stoploss * profit_loss_ratio2 < resistance) & \
        (get_prev(price_low) < entry_price) & (price >= entry_price)
    return signal, stoploss_level, stopprofit_level, entry_price

def breakdown(pivot_strength, pivot_lower_bound, pivot_upper_bound, volatility, macd, macd2,
        price, price_low, price_high, now_step, support, resistance, min_strength=2.5, min_macd=1.5, min_macd2=1.,
        min_stoploss_factor=0, profit_loss_ratio=profit_loss_ratio, profit_loss_ratio2=None):
    if profit_loss_ratio2 is None:
        profit_loss_ratio2 = profit_loss_ratio
    stoploss = np.maximum(np.ceil(volatility * min_stoploss_factor), pivot_upper_bound + 1 - pivot_lower_bound)
    stoploss_level = pivot_lower_bound
    entry_price = stoploss_level - stoploss
    stopprofit = stoploss * profit_loss_ratio
    stopprofit_level = np.round(entry_price - stopprofit).astype('int32')
    signal = (pivot_strength >= min_strength) & \
        (get_by_step(macd, now_step) <= -min_macd) & \
        (get_by_step(macd2, now_step) <= -min_macd2) & \
        (entry_price - stoploss * profit_loss_ratio2 > support) & \
        (get_prev(price_high) > entry_price) & (price <= entry_price)
    return signal, stoploss_level, stopprofit_level, entry_price

#load_globals()
#with Timer() as t:
#    profiles0 = load_profiles(d, 'piv5m')
#    profiles = filter_profiles(profiles0, strength_thld)
#print 'Load and filter profiles took %f sec.' % t.interval
#
#with Timer() as t:
#    load_pivot_inds(profiles)
#print 'Load pivot indicators took %f sec.' % t.interval
#
#accessed_max_strength = False
#
#l = [output_support_resistance(profile, l, h, p, ps, lb, ub, s, v, 
#        accessed_max_strength=accessed_max_strength) 
#        for profile, l, h, p, ps, lb, ub, s, v in 
#        zip(profiles, lowest, highest, price, price_strength, pivot_lower_bound, pivot_upper_bound, pivot_strength, volatility)]
#support, resistance = zip2arr(l)
#
#candidate = breakdown(pivot_strength, pivot_lower_bound, pivot_upper_bound, volatility, macd, macd2,
#                   price, price_low, price_high, now_step, support=support, resistance=resistance)
#win, loss, unknown = get_target(future_low, future_high, future_low_before_high, future_high_before_low, 
#                future_low2, future_high2, future_low_before_high2, future_high_before_low2, *candidate, count_by_month=True)
#
#signal, stoploss_level, stopprofit_level, entry_price = candidate
#d['piv5m.stoploss_level'] = stoploss_level
#d['piv5m.stopprofit_level'] = stopprofit_level

#accessed_max_strength = False
#decay_factor = 0.5
#min_macd = 1.
#min_macd2 = 1.
#min_strength = 2.5
#min_dt = 0.5*60*2
#min_stoploss_factor=.5

#for profit_loss_ratio2 in [2., 3.]:
#    for min_macd in [0, 0.5, 1.]:
#        for min_dt in [1*60*2, 0.5*60*2]:
##            print 'accessed_max_strength =', accessed_max_strength, 'decay_factor =', decay_factor
#            print '\nprofit_loss_ratio2 =', profit_loss_ratio2, 'min_macd =', min_macd, 'min_dt =', min_dt
#            l = [output_support_resistance(profile, l, h, p, ps, lb, ub, s, v, 
#                    decay_factor=decay_factor, accessed_max_strength=accessed_max_strength) 
#                    for profile, l, h, p, ps, lb, ub, s, v in 
#                    zip(profiles, lowest, highest, price, price_strength, pivot_lower_bound, pivot_upper_bound, pivot_strength, volatility)]
#            support, resistance = zip2arr(l)
#            #            d['piv5m.support'] = support
#            #            d['piv5m.resistance'] = resistance
#            
#            candidate = bouncedown(pivot_strength, pivot_lower_bound, pivot_upper_bound, past_low, past_low_step, past_high, past_high_step, macd, 
#                               price, price_low, price_high, now_step, support=support, resistance=resistance, 
#                               min_strength=min_strength, min_macd=min_macd, min_dt=min_dt, profit_loss_ratio2=profit_loss_ratio2)
#            win, loss, unknown = get_target(future_low, future_high, future_low_before_high, future_high_before_low, 
#                            future_low2, future_high2, future_low_before_high2, future_high_before_low2, *candidate, count_by_month=False)

#candidate = bounceup(pivot_strength, pivot_lower_bound, pivot_upper_bound, past_low, past_low_step, past_high, past_high_step, macd, 
#                   price, price_low, price_high, now_step, support=support, resistance=resistance, min_strength=strength_thld, min_dt=1*60*2, min_macd=0.)
#win, loss, unknown = get_target(future_low, future_high, future_low_before_high, future_high_before_low, 
#                            future_low2, future_high2, future_low_before_high2, future_high_before_low2, *candidate, count_by_month=True)
#   


#from build_tick_dataset import futures

futures = [
#('dc', 'pp', 1, 3.75),#
#('dc', 'l', 5, 3.75),
('zc', 'MA', 1, 6.25),
('sc', 'bu', 2, (7.75+5.75)/2),  # liqing, 2000 #

('zc', 'SR', 1, 6.25), # tang, 5000
#('dc', 'm', 1, 6.25), # doupo, 2000 #
('dc', 'p', 2, 6.25), # zonglv, 5000
('dc', 'c', 1, 3.75), # yumi, 2000
('zc', 'CF', 5, 6.25 ),  # cotton, 12000

('sc', 'ag', 1, 9.25), # Ag, 3000
('sc', 'zn', 5, 7.75), # Zn, 15000
('sc', 'ni', 10, 7.75), # Ni, 60000

('sc', 'rb', 1, (7.75+5.75)/2), # luowen, 2000 #
('dc', 'i', 0.5, 6.25), # tiekuang, 300 
('dc', 'j', 0.5, 6.25), # #jiaotan, 800 
]

plot_futures = [
#('dc', 'pp', 1, 3.75),#
('dc', 'm', 1, 6.25), # doupo, 2000 # 
]

base_dir = 'data/'
    