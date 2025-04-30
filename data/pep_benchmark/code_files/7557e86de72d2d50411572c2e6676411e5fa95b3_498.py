from math import expm1, log


class Filter:
    def __init__(self, filter_length):
        self._buffer = [0]*filter_length
    
    def filter(self, val):
        for i in range(len(self._buffer) - 1):
            self._buffer[i] = self._buffer[i + 1]
        self._buffer[-1] = val
        
        avg = 0.0
        for i in range(len(self._buffer)):
            avg += self._buffer[i]
        return avg/len(self._buffer)


class WeightedFilter:
    def __init__(self, coefficients):
        self._coefficients = coefficients
        self._buffer = [0]*len(coefficients)
    
    def filter(self, val):
        for i in range(len(self._buffer) - 1):
            self._buffer[i] = self._buffer[i + 1]
        self._buffer[-1] = val
        return self._weighted_avg()
    
    def _weighted_avg(self):
        avg = 0.0
        normalizing_factor = 0.0
        for i in range(len(self._buffer)):
            avg += self._buffer[i]*self._coefficients[i]
            normalizing_factor += self._coefficients[i]
        return avg/normalizing_factor


class ExponentialFilter():
    """A simple exponential filter:
    s_n = alpha*val + (1-alpha)*s_n-1
    more info on wikipedia"""
    def __init__(self, smoothing_factor):
        assert 0 < smoothing_factor < 1, 'smoothing factor must be in (0, ' \
                                          '1]. it is {}'.format(
            smoothing_factor)
        self._sf = smoothing_factor
        self._smoothed = None
    
    def filter(self, val):
        if self._smoothed is None:
            self._smoothed = val
        else:
            self._smoothed +=  self._sf*(val - self._smoothed)
        return self._smoothed
    
    def reset(self):
        self._smoothed = None


class DoubleExponentialFilter():
    """Based on the double exponential filter on wikipedia."""
    def __init__(self, smoothing_factor, trend_smoothing_factor):
        assert 0 < smoothing_factor < 1, 'smoothing factor must be in (0, ' \
                                          '1]. it is {}'.format(
            smoothing_factor)
        self._sf = smoothing_factor
        assert 0 < trend_smoothing_factor < 1, 'trend smoothing factor must ' \
                                                'be in (0, 1]. it is {}'.format(
            trend_smoothing_factor)
        self._tsf = trend_smoothing_factor
        self._smoothed = None
        self._trend = None
    
    def filter(self, val):
        if self._smoothed is None:
            self._smoothed = val
        else:
            predicted = self.predict_next()
            self._smoothed = predicted + self._sf*(val - predicted)
            if self._trend is None:
                self._trend = self._smoothed - predicted
            else:
                self._trend += self._tsf*(self._smoothed - predicted)
        return self._smoothed
    
    def predict_next(self):
        if self._trend is None:
            return self._smoothed
        return self._smoothed + self._trend
    
    def reset(self):
        self._smoothed = None
        self._trend = None


class IrregularExponentialFilter():
    """Irregular Exponential Filter (info can be found in the paper titled
    Algorithms for Unevenly Spaced Time Series - Moving Averages and Other
    Rolling Operators by A. Eckner, https://eckner.com/papers/Algorithms%20for%20Unevenly%20Spaced%20Time%20Series.pdf)"""
    def __init__(self, normal_interval, smoothing_factor):
        assert 0 < smoothing_factor < 1, 'smoothing factor must be in (0, ' \
                                          '1]. it is {}'.format(
            smoothing_factor)
        self._tau = normal_interval/log(1/(1-smoothing_factor))
        self._smoothed = None
        self._last_val = None
    
    def filter(self, val, interval):
        if self._smoothed is None:
            self._smoothed = val
        else:
            alpha = -expm1(-interval/self._tau)
            y = self._tau*alpha/interval
            self._smoothed += alpha*(self._last_val - self._smoothed)
            self._smoothed += (1-y)*(val-self._last_val)
        self._last_val = val
        return self._smoothed
    
    def reset(self):
        self._smoothed = None
        self._last_val = None


class IrregularDoubleExponentialFilter():
    """A special filter developed for use in this project. It combines the
    features of the Double Exponential Filter (info can be found on wikipedia)
    and the Irregular Exponential Filter (info can be found in the paper titled
    Algorithms for Unevenly Spaced Time Series - Moving Averages and Other
    Rolling Operators by A. Eckner, https://eckner.com/papers/Algorithms%20for%20Unevenly%20Spaced%20Time%20Series.pdf)
    
    I know I haven't written enough about it in the report so any questions
    about it can be answered by the developer at sketchn98@gmail.com"""
    def __init__(self, normal_interval, smoothing_factor, trend_smoothing_factor):
        assert 0 < smoothing_factor < 1, 'smoothing factor must be in (0, ' \
                                          '1]. it is {}'.format(
            smoothing_factor)
        self._tau_s = normal_interval/log(1/(1-smoothing_factor))
        assert 0 < trend_smoothing_factor < 1, 'trend smoothing factor must ' \
                                                'be in (0, 1]. it is {}'.format(
            trend_smoothing_factor)
        self._tau_t = normal_interval/log(1/(1-trend_smoothing_factor))
        self._smoothed = None
        self._trend = None
        self._last_val = None
        self._last_rate = None
    
    def filter(self, val, interval):
        if self._smoothed is None:
            self._smoothed = val
        else:
            predicted = self.predict_next(interval)
            last_smoothed = self._smoothed
            
            alpha = -expm1(-interval/self._tau_s)
            y = self._tau_s*alpha/interval
            self._smoothed = predicted + (1-y)*(val - self._last_val)
            self._smoothed += alpha*(self._last_val - predicted)
            
            rate = (self._smoothed - last_smoothed)/interval
            if self._trend is None:
                self._trend = rate
            else:
                beta = -expm1(-interval/self._tau_t)
                z = self._tau_t*beta/interval
                self._trend += beta*(self._last_rate - self._trend)
                self._trend += (1-z)*(rate - self._last_rate)
            self._last_rate = rate
        self._last_val = val
        return self._smoothed
    
    def predict_next(self, interval):
        if self._trend is None:
            return self._smoothed
        return self._smoothed + self._trend*interval
    
    def reset(self):
        self._smoothed = None
        self._trend = None
        self._last_val = None
        self._last_rate = None
