import numpy as np

class Convolution():

        
    def convolute(self, window, window_len, data):

        if window not in ['flat', 'hanning', 'hamming', 'blackman', 'bartlett']:
            raise ValueError ("window must be one of the following: ['flat', 'hanning', 'hamming', 'blackman', 'bartlett']")

        if window_len % 2 != 0:
            window_len +=1

        #adding padding at begining and end of the signal to minimize data loss
        preprocessed_signal = np.r_[data[window_len-1 : 0: -1], data, data[-2 : -window_len-1 : -1]]

        if window == "flat":
            w = np.r_[[0], np.ones(window_len-2), [0]]
        else:
            w = eval('np.'+window+'(window_len)')

        #performing convolution
        out_ = np.convolve(w/w.sum(), preprocessed_signal, mode='valid')
       
        #adjusting length of the convoluted signal
        start = int(window_len/2)
        end = -int(window_len/2 - 1)
        out_signal = out_[start : end]

        return out_signal
