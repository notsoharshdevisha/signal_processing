import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from Convolution import Convolution

#generating signal
t = np.arange(0,10, 0.01)
orig_signal = 2 *np.sin(2 * np.pi * 1.5 * t) + 3 * np.cos(2 * np.pi * 1.75 * t)
err = np.random.rand(len(t))
signal = orig_signal + err * 1.5
fig, ax = plt.subplots(nrows = 2)
ax[0].plot(t, orig_signal, linewidth=1)
ax[0].set_title('Original Signal')
ax[0].set_xlabel("Time Index")
ax[0].set_ylabel('Amplitude')
ax[0].grid()
ax[0].set_xlim(0, 10)
ax[1].plot(t, signal)
ax[1].set_title('Signal with induced error')
ax[1].set_xlabel("Time Index")
ax[1].set_ylabel('Amplitude')
ax[1].grid()
ax[1].set_xlim(1, 10)
fig.tight_layout()
plt.show()


#Different window functions 
window_len = len(t)
hanning = np.hanning(window_len)
hamming = np.hamming(window_len)
blackman = np.blackman(window_len)
flat = np.r_[[0], np.ones(window_len-2), [0]]
bartlett = np.bartlett(window_len)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
ax.plot(t, hanning, label = 'hanning')
ax.plot(t, hamming, label = 'hamming')
ax.plot(t, blackman, label = 'balckman')
ax.plot(t, flat, label = 'flat')
ax.plot(t, bartlett, label = 'bartlett')
ax.legend(loc = 1)
ax.grid()
ax.set_title('Windows')
plt.show()

#perfroming convolution
window_len = 15
conv_obj = Convolution()
out_signal = conv_obj.convolute('flat', window_len, signal)
fig, ax = plt.subplots()
ax.plot(t, out_signal, label='out')
ax.plot(t, orig_signal, label='original')
ax.set_title('Using Convolution')
ax.legend()
plt.show()

preprocessed_signal = np.r_[signal[window_len-1:0:-1], signal, signal[-2:-window_len-1:-1]]
new_t = np.r_[t[window_len-1:0:-1], t, t[-2:-window_len-1:-1]]
new_t = new_t[:, np.newaxis]
knnregressor = KNeighborsRegressor(n_neighbors=window_len)
knnregressor.fit(new_t, preprocessed_signal)
ai_signal = knnregressor.predict(t[:, np.newaxis])
fig, ax = plt.subplots()
ax.plot(t, ai_signal, label='knnsignal')
ax.plot(t, orig_signal, label='original')
ax.set_title('Using KNeighboursRegressor')
ax.legend()
plt.show()

#comparing results
fig, ax = plt.subplots(nrows=2)
ax[0].plot(t, out_signal, label='out')
ax[0].plot(t, orig_signal, label='original')
ax[0].set_title('Using Convolution')
ax[0].legend()
ax[1].plot(t, ai_signal, label='knnsignal')
ax[1].plot(t, orig_signal, label='original')
ax[1].set_title('Using KNeighboursRegressor')
ax[1].legend()
fig.tight_layout()
plt.show()

signal = pd.Series(signal)
fig, ax = plt.subplots(nrows=2)
ax[0].fill_between(t, signal - signal.rolling(window_len).std(), signal + signal.rolling(window_len).std(), alpha=0.5, facecolor='lightseagreen')
ax[0].plot(t, out_signal, color='mediumseagreen')
ax[0].set_title('Using Convolution')
ax[1].fill_between(t, signal - signal.rolling(window_len).std(), signal + signal.rolling(window_len).std(), alpha=0.5, facecolor='lightseagreen')
ax[1].plot(t, ai_signal, color = 'mediumseagreen')
ax[1].set_title('Using KNeighboursRegressor')
fig.tight_layout()
plt.show()











