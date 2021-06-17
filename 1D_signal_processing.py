import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from Convolution import Convolution
#import seaborn as sns
#sns.set()

#generating signal
t = np.arange(0,10, 0.01)
orig_signal = 2 *np.sin(2 * np.pi * 1.5 * t) + 3 * np.cos(2 * np.pi * 1.75 * t)
err = np.random.rand(len(t))
signal = orig_signal + err * 1.5
fig, ax = plt.subplots(nrows = 2)
ax[0].plot(t, orig_signal, linewidth=1)
ax[0].set_title('Original Signal', fontsize=7)
ax[0].set_xlabel("Time Index", fontsize=6)
ax[0].set_ylabel('Amplitude', fontsize=6)
ax[0].set_xlim(0, 10)
ax[0].grid(linewidth=0.5, alpha=0.5)
ax[1].plot(t, signal)
ax[1].set_title('Signal with induced error', fontsize=7)
ax[1].set_xlabel("Time Index", fontsize=5)
ax[1].set_ylabel('Amplitude', fontsize=5)
ax[1].set_xlim(0, 10)
ax[1].grid(linewidth=0.5, alpha=0.5)
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
ax.legend(loc = 1, prop={'size':6})
ax.set_title('Windows', fontsize=10)
ax.grid(linewidth=0.5, alpha=0.5) 
plt.show()

#perfroming convolution

#with convolution module
window_len = 15
conv_obj = Convolution()
out_signal = conv_obj.convolute('flat', window_len, signal)

#using KNeighbors
kregressor = KNeighborsRegressor(n_neighbors=15)
kregressor.fit(t[:,np.newaxis], signal)
ai_signal = kregressor.predict(t[:,np.newaxis])

signal = pd.Series(signal)
fig, ax = plt.subplots(nrows=2)
fig.suptitle('Window Length:{}'.format(window_len), fontsize=9)
ax[0].fill_between(t, signal - signal.rolling(window_len).std(), signal + signal.rolling(window_len).std(), alpha=0.5, facecolor='lightseagreen'                             , label='Original Signal')
ax[0].plot(t, out_signal, color='mediumseagreen', label='Convoluted Signal')
ax[0].set_xlabel('Time Index', fontsize=5)
ax[0].set_ylabel('Amplitude', fontsize=5)
ax[0].set_title('Using Convolution', fontsize=7)
ax[0].set_xlim(t.min(), t.max())
ax[0].legend(loc=1, prop={'size':5})
ax[0].grid(linewidth=0.5, alpha=0.5) 
ax[1].fill_between(t, signal - signal.rolling(window_len).std(), signal + signal.rolling(window_len).std(), alpha=0.5, facecolor='lightseagreen'                             , label='Original Signal')
ax[1].plot(t, ai_signal, color = 'mediumseagreen', label='Knn')
ax[1].set_xlabel('Time Index', fontsize=5)
ax[1].set_ylabel('Amplitude', fontsize=5)
ax[1].set_title('Using KNeighboursRegressor', fontsize=7)
ax[1].set_xlim(t.min(), t.max())
ax[1].legend(loc=1, prop={'size':5})
ax[1].grid(linewidth=0.5, alpha=0.5) 
fig.tight_layout()
plt.show()

#applying to earthquake lab sample data from los alamos national laboratory
df = pd.read_csv('seg_004314.csv')
signal = np.array(df['acoustic_data'])
t = np.arange(len(signal))
win_len = 15

#Using convolution
convobj = Convolution()
conv_out = convobj.convolute('flat', window_len, signal)

#Using KNearestNeighbors
regressor = KNeighborsRegressor(n_neighbors=win_len, weights='uniform', n_jobs=-1)
regressor.fit(t[:,np.newaxis], signal)
ai_out = regressor.predict(t[:,np.newaxis])

#visualizations
fig, ax = plt.subplots(nrows=2)
fig.suptitle('Window Length:{}'.format(win_len), fontsize=9)
ax[0].plot(t, signal, color='mediumseagreen', alpha=0.5, label='Original Signal', linewidth=0.5)
ax[0].plot(t, conv_out, color='black', label='Convoluted Signal', linewidth=0.5)
ax[0].set_xlabel('Time Index', fontsize=5)       
ax[0].set_ylabel('Amplitude', fontsize=5)                                                                                       
ax[0].set_title('Using Convolution', fontsize=7)
ax[0].grid(linewidth=0.5, alpha=0.5)
ax[0].set_xlim(t.min(), t.max())
ax[0].legend(loc=1, prop={'size':5})
ax[1].plot(t, signal, color='mediumseagreen', alpha=0.5, label='Original Signal', linewidth=0.5)
ax[1].plot(t, ai_out, color='black', label='Knn', linewidth=0.5)
ax[1].set_xlabel('Time Index', fontsize=5)      
ax[1].set_ylabel('Amplitude', fontsize=5)       
ax[1].set_title('Using Convolution', fontsize=7)
ax[1].set_xlim(t.min(), t.max())     
ax[1].grid(linewidth=0.5, alpha=0.5)
ax[1].legend(loc=1, prop={'size':5})
fig.tight_layout()
plt.show()

#to see the effects more clearly
start=15000
end=28000
signal = signal[start:end]
conv_out = conv_out[start:end]
ai_out = ai_out[start:end]
t = t[start:end]
fig, ax = plt.subplots(nrows=2)                                                  
fig.suptitle('Window Length:{}'.format(win_len), fontsize=9)                     
ax[0].plot(t, signal, color='mediumseagreen', alpha=0.5, label='Original Signal', linewidth=0.5)
ax[0].plot(t, conv_out, color='black', label='Convoluted Signal', linewidth=0.5)                
ax[0].set_xlabel('Time Index', fontsize=5)                                       
ax[0].set_ylabel('Amplitude', fontsize=5)                                        
ax[0].set_title('Using Convolution', fontsize=7)                                 
ax[0].grid(linewidth=0.5, alpha=0.5)                                             
ax[0].set_xlim(t.min(), t.max()) 
ax[0].legend(loc=1, prop={'size':5})
ax[1].plot(t, signal, color='mediumseagreen', alpha=0.5, label='Original Signal', linewidth=0.5)
ax[1].plot(t, ai_out, color='black', label='Knn', linewidth=0.5)                                
ax[1].set_xlabel('Time Index', fontsize=5)                                       
ax[1].set_ylabel('Amplitude', fontsize=5)                                        
ax[1].set_title('Using Convolution', fontsize=7)                                 
ax[1].set_xlim(t.min(), t.max())                                                 
ax[1].grid(linewidth=0.5, alpha=0.5)                                             
ax[1].legend(loc=1, prop={'size':5})
fig.tight_layout()                                                               
plt.show()   




