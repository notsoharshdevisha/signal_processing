import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import collections
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('PPG.csv')

ppg = np.array(data['signal'])
n = len(ppg)
t = np.arange(n)
fs = 64
#visualizing the signal
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(t, ppg, color='lightseagreen', alpha=0.75, linewidth=0.75, label='Signal')
ax.set_xlabel('Samples', fontsize=10)
ax.set_ylabel('Signal', fontsize=10)
ax.legend(prop={'size':9})
ax.grid(linewidth=0.5, alpha=0.5)
ax.set_title('PPG Signal', fontsize=15)
ax.set_xlim(t.min(), t.max())
plt.setp(ax.get_xticklabels(), fontsize=6)
plt.setp(ax.get_yticklabels(), fontsize=6)
plt.show()

#using np.fft.fft
ppg_fft = np.fft.fft(ppg)
ppg_psd = ppg_fft * np.conjugate(ppg_fft) / n
f = (fs/n) * np.arange(n)
#visualizing PSD
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(f[:n//2], ppg_psd[:n//2], color = 'lightseagreen', alpha=0.75, label='fft*conj(fft)/n')
ax.set_title("Power Spectral Density", fontsize=15)
ax.set_xlabel("Frequency", fontsize=12)
ax.set_ylabel("Power", fontsize=12)
ax.grid(linewidth=0.5, alpha=0.5)
ax.set_xlim(min(f), max(f[:n//2]))
ax.legend(prop={'size':10})
plt.setp(ax.get_xticklabels(), fontsize=7)
plt.setp(ax.get_yticklabels(), fontsize=7)
plt.show()

#initializing the thresholds 
thresh1 = 0.25*1e6           
thresh2 = 0.5*1e6            

#viz psd cutoff
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(f[:n//2], ppg_psd[:n//2], color = 'lightseagreen', alpha=0.75, label='fft*conj(fft)/n')
ax.set_title("Power Spectral Density", fontsize=15)                                            
ax.set_xlabel("Frequency", fontsize=12)                                                         
ax.set_ylabel("Power", fontsize=12)                                                             
ax.grid(linewidth=0.5, alpha=0.5)                                                              
ax.set_xlim(min(f), max(f[:n//2]))                                                                                              
ax.axhline(y=thresh1, c='black', label='thresh1', linestyle='--')
ax.axhline(y=thresh2, c='black', label='thresh2', linestyle='--')
plt.setp(ax.get_xticklabels(), fontsize=7)                                                     
plt.setp(ax.get_yticklabels(), fontsize=7)                                                     
ax.legend(prop={'size':10}) 
plt.show()                                                                                     

#Comparing filtered signal with different thresholds
#'filtering'
ppg_fft_thresh1 = [0 if ppg_psd[i]<thresh1 else ppg_fft[i] for i in range(n)]
ppg_fft_thresh2 = [0 if ppg_psd[i]<thresh2 else ppg_fft[i] for i in range(n)]
#computing ift of the 'filtered' signals
ppg_thresh1 = np.fft.ifft(ppg_fft_thresh1)
ppg_thresh2 = np.fft.ifft(ppg_fft_thresh2)
#VIZ
fig, ax = plt.subplots(nrows=2, figsize=(14, 8))
ax[0].plot(t, ppg, color='lightseagreen', alpha=0.75, label='Original Signal', linewidth=0.75)
ax[0].plot(t, ppg_thresh1, color='black', label='PSD > 0.25*10^6', linewidth=0.75)
ax[0].set_title("Threshold : 0.25*10^6", fontsize=12)
ax[0].set_xlabel('Samples', fontsize=10)
ax[0].set_ylabel('Signal', fontsize=10)
ax[0].legend(loc=1, prop={'size':9})
ax[0].grid(linewidth=0.5, alpha=0.5) 
ax[0].set_xlim(t.min(), t.max()) 
ax[1].plot(t, ppg, color='lightseagreen', alpha=0.75, label='Original Signal', linewidth=0.75) 
ax[1].plot(t, ppg_thresh2, color='black', label='PSD > 0.5*10^6', linewidth=0.75) 
ax[1].set_title("Threshold : 0.5*10^6", fontsize=12)
ax[1].set_xlabel('Samples', fontsize=10) 
ax[1].set_ylabel('Signal', fontsize=10)  
ax[1].legend(loc=1, prop={'size':9})    
ax[1].grid(linewidth=0.5, alpha=0.5)    
ax[1].set_xlim(t.min(), t.max())
plt.setp(ax[0].get_xticklabels(), fontsize=5)
plt.setp(ax[0].get_yticklabels(), fontsize=5) 
plt.setp(ax[1].get_xticklabels(), fontsize=5)  
plt.setp(ax[1].get_yticklabels(), fontsize=5)  
fig.tight_layout()
plt.show()
 
# a more minute analysis
fig, ax = plt.subplots(nrows=2, figsize=(14, 8))
ax[0].plot(t[:fs*60], ppg[:fs*60], color='lightseagreen', label='Original Signal', linewidth=1.5)
ax[0].plot(t[:fs*60], ppg_thresh1[:fs*60], color='black', label='PSD > 0.25*10^6', linewidth=0.75)
ax[0].set_title("Threshold : 0.25*10^6", fontsize=12)
ax[0].set_xlabel('Samples', fontsize=10)
ax[0].set_ylabel('Signal', fontsize=10)
ax[0].legend(loc=1, prop={'size':9})
ax[0].grid(linewidth=0.5, alpha=0.5) 
ax[0].set_xlim(t[:fs*60].min(), t[:fs*60].max()) 
ax[1].plot(t[:fs*60], ppg[:fs*60], color='lightseagreen', label='Original Signal', linewidth=1.5) 
ax[1].plot(t[:fs*60], ppg_thresh2[:fs*60], color='black', label='PSD > 0.5*10^6', linewidth=0.75) 
ax[1].set_title("Threshold : 0.5*10^6", fontsize=12)
ax[1].set_xlabel('Samples', fontsize=10) 
ax[1].set_ylabel('Signal', fontsize=10)  
ax[1].legend(loc=1, prop={'size':9})    
ax[1].grid(linewidth=0.5, alpha=0.5)    
ax[1].set_xlim(t[:fs*60].min(), t[:fs*60].max())
plt.setp(ax[0].get_xticklabels(), fontsize=5)
plt.setp(ax[0].get_yticklabels(), fontsize=5) 
plt.setp(ax[1].get_xticklabels(), fontsize=5)  
plt.setp(ax[1].get_yticklabels(), fontsize=5)  
fig.tight_layout()
plt.show()
                        
#using butterworth filter
thresh1 = 1.5
thresh2 = 1
order = 10
sos = signal.butter(order, thresh1, btype='lowpass', output='sos', fs=64)
filtered_signal_1 = signal.sosfilt(sos, ppg)
sos = signal.butter(order, thresh2, btype='lowpass', output='sos', fs=64)
filtered_signal_2 = signal.sosfilt(sos, ppg)

#visualizing the results
fig, ax = plt.subplots(nrows=2, figsize=(14,8))
ax[0].plot(t, ppg, color='lightseagreen', alpha=0.75, label='Original Signal')
ax[0].plot(t, filtered_signal_1, color='black', label='lowpass : {}Hz'.format(str(thresh1)), linewidth=0.75)
ax[0].set_xlabel('Samples', fontsize=10)
ax[0].set_ylabel('Signal', fontsize=10)
ax[0].grid(linewidth=0.5, alpha=0.5)
ax[0].legend(loc=1, prop={'size':9})
ax[0].set_xlim(t.min(), t.max())
ax[0].set_title('Lowpass : {}Hz'.format(thresh1), fontsize=12)
plt.setp(ax[0].get_xticklabels(), fontsize=7)
plt.setp(ax[0].get_yticklabels(), fontsize=7)
ax[1].plot(t, ppg, color='lightseagreen', alpha=0.75, label='Original Signal')                              
ax[1].plot(t, filtered_signal_2, color='black', label='lowpass : {}Hz'.format(str(thresh2)), linewidth=0.75)
ax[1].set_xlabel('Samples', fontsize=10)                                                                     
ax[1].set_ylabel('Signal', fontsize=10)                                                                      
ax[1].grid(linewidth=0.5, alpha=0.5)                                                                        
ax[1].legend(loc=1, prop={'size':9})                                                                        
ax[1].set_xlim(t.min(), t.max())                                                                            
ax[1].set_title('Lowpass : {}Hz'.format(thresh2), fontsize=12)
plt.setp(ax[1].get_xticklabels(), fontsize=7)                                                               
plt.setp(ax[1].get_yticklabels(), fontsize=7)                                                               
fig.suptitle('Using Butterworth Filter', fontsize=12)
fig.tight_layout()
plt.show()

#for a minute comparison
sec = 120
fig, ax = plt.subplots(nrows=2, figsize=(14, 8))                                                                             
ax[0].plot(t[:fs*sec], ppg[:fs*sec], color='lightseagreen', alpha=0.75, label='Original Signal')                              
ax[0].plot(t[:fs*sec], filtered_signal_1[:fs*sec], color='black', label='lowpass : {}Hz'.format(str(thresh1)), linewidth=0.75)
ax[0].set_xlabel('Samples', fontsize=10)                                                                     
ax[0].set_ylabel('Signal', fontsize=10)                                                                      
ax[0].grid(linewidth=0.5, alpha=0.5)                                                                        
ax[0].legend(loc=1, prop={'size':9})                                                                        
ax[0].set_xlim(t[:fs*sec].min(), t[:fs*sec].max())                                                                            
ax[0].set_title('Lowpass : {}Hz'.format(thresh1), fontsize=12)                                               
plt.setp(ax[0].get_xticklabels(), fontsize=7)                                                               
plt.setp(ax[0].get_yticklabels(), fontsize=7)                                                               
ax[1].plot(t[:fs*sec], ppg[:fs*sec], color='lightseagreen', alpha=0.75, label='Original Signal')                              
ax[1].plot(t[:fs*sec], filtered_signal_2[:fs*sec], color='black', label='lowpass : {}Hz'.format(str(thresh2)), linewidth=0.75)
ax[1].set_xlabel('Samples', fontsize=10)                                                                     
ax[1].set_ylabel('Signal', fontsize=10)                                                                      
ax[1].grid(linewidth=0.5, alpha=0.5)                                                                        
ax[1].legend(loc=1, prop={'size':9})                                                                        
ax[1].set_xlim(t[:fs*sec].min(), t[:fs*sec].max())                                                                            
ax[1].set_title('Lowpass : {}Hz'.format(thresh2), fontsize=12)                                               
plt.setp(ax[1].get_xticklabels(), fontsize=7)                                                               
plt.setp(ax[1].get_yticklabels(), fontsize=7)                                                               
fig.suptitle('Using Butterworth Filter', fontsize=12)                                                        
fig.tight_layout()                                                                                          
plt.show()      

#comparing different order of the filter
order1 = 1
order2 = 10
sos = signal.butter(order1, 1, btype='lowpass', output='sos', fs=64)
filterd_signal_1 = signal.sosfilt(sos, ppg)
sos = signal.butter(order2, 1, btype='lowpass', output='sos', fs=64)
filtered_signal_2 = signal.sosfilt(sos, ppg)
sec = 120                                                                                                                     
fig, ax = plt.subplots(nrows=2, figsize=(14, 8))                                                                                               
ax[0].plot(t[:fs*sec], ppg[:fs*sec], color='lightseagreen', alpha=0.75, label='Original Signal')                              
ax[0].plot(t[:fs*sec], filtered_signal_1[:fs*sec], color='black', label='Filtered Signal', linewidth=0.75)
ax[0].set_xlabel('Samples', fontsize=10)                                                                                       
ax[0].set_ylabel('Signal', fontsize=10)                                                                                        
ax[0].grid(linewidth=0.5, alpha=0.5)                                                                                          
ax[0].legend(loc=1, prop={'size':9})                                                                                          
ax[0].set_xlim(t[:fs*sec].min(), t[:fs*sec].max())                                                                            
ax[0].set_title('Order : {}'.format(order1), fontsize=12)                                                                 
plt.setp(ax[0].get_xticklabels(), fontsize=7)                                                                                 
plt.setp(ax[0].get_yticklabels(), fontsize=7)                                                                                 
ax[1].plot(t[:fs*sec], ppg[:fs*sec], color='lightseagreen', alpha=0.75, label='Original Signal')                              
ax[1].plot(t[:fs*sec], filtered_signal_2[:fs*sec], color='black', label='Filtered Signal', linewidth=0.75)
ax[1].set_xlabel('Samples', fontsize=10)                                                                                       
ax[1].set_ylabel('Signal', fontsize=10)                                                                                        
ax[1].grid(linewidth=0.5, alpha=0.5)                                                                                          
ax[1].legend(loc=1, prop={'size':9})                                                                                          
ax[1].set_xlim(t[:fs*sec].min(), t[:fs*sec].max())                                                                            
ax[1].set_title('Order : {}'.format(order2), fontsize=12)                                                                 
plt.setp(ax[1].get_xticklabels(), fontsize=7)                                                                                 
plt.setp(ax[1].get_yticklabels(), fontsize=7)                                                                                 
fig.suptitle('Using Butterworth Filter', fontsize=12)                                                                          
fig.tight_layout()                                                                                                            
plt.show()                                         



