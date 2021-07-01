import numpy as np
import cv2 as cv
import warnings

#initializing the kernels
kernels = {'gblur5' : 1/233 * np.array([[1,  4,  7,  4, 1],
                                        [4, 16, 26, 16, 4],
                                        [7, 26, 41, 26, 7],
                                        [4, 16, 26, 16, 4],
                                        [1,  4,  7,  4, 1]]), 

           'gblur3' : 1/16 * np.array([[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]]), 

           'edge' : np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]]),

           'depth' : np.array([[-2, -1, 0],
                               [-1,  1, 1],
                               [ 0,  1, 2]]),

           'sharpen' : np.array([[ 0, -1,  0],
                                 [-1,  5, -1],
                                 [ 0, -1,  0]]),

           'denoise3' : 1/9 * np.ones((3,3)),

           'denoise5' : 1/25 * np.ones((5,5))}


class Convolution():
        
    def convolute(self, window, window_len, data):
        '''performs 1D convolution on 1D/time-series signals
           window : type of window
           window_len : length of the window
           data : 1D/time-series signal'''

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

    def convolute2d(self, img, kernel, pool_type, channels=None):
        '''performs convolution on images
           img: input image
           kernel : type of filter
           pool_type : one of ['max', 'min', 'average', 'sum']
                       results may vary depending upon the type of filter used
           channels : 1 for B/W and 3 for RGB'''

        if kernel not in ['gblur3', 'gblur5', 'sharpen', 'edge', 'depth', 'denoise3', 'denoise5']:
            raise ValueError("The kernel must be one of ['gblur3', 'gblur5', 'sharpen', 'edge', 'depth', 'denoise3', 'denoise5']")

        if channels:
            if channels != 1 and channels != 3:
                raise ValueError('Number of channels must be either 1 or 3')

        if kernel in  ['sharpen', 'edge', 'depth'] and pool_type != 'sum':
            warnings.warn("Use pool_type='sum' while using any of ['sharpen', 'edge', 'depth'] for optimal reslts")

        if channels == 1:
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        if len(img.shape) != 3:
            img = img.reshape((img.shape[0], img.shape[1], 1))

        channels = img.shape[2]

        #padding image according to the kernel size
        kernel_size = kernels[kernel].shape[0]
        if kernel_size  == 3:
            padded_img = np.zeros((img.shape[0]+2, img.shape[1]+2, channels))
            padded_img[1:-1, 1:-1, 0:channels] = img
        else:  
            padded_img = np.zeros((img.shape[0]+4, img.shape[1]+4, channels))
            padded_img[2:-2, 2:-2, 0:channels] = img

        #initializing output image
        out_img = np.zeros(img.shape)
        
        #performing covolution
        for k in range(channels):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    out_img[i, j, k] = eval('np.'+pool_type+'(padded_img[i:i+kernel_size, j:j+kernel_size, k] * kernels[kernel])') 

        return out_img

    
class fft():

    def filter(self, img, type_, radius, channels=None):
        '''performs fft on the input image and applies the low/high-pass filter'''

        #converts 
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        center = (img.shape[0]//2, img.shape[1]//2)

        if img.shape[0] % 2 == 0:
            y = np.arange(-center[0], center[0])
        else:
            y = np.arange(-center[0], center[0]+1)

        if img.shape[1] % 2 == 0:
            x = np.arange(-center[1], center[1])
        else:
            x = np.arange(-center[1], center[1]+1)

        xx, yy = np.meshgrid(x, y)

        img_fft = np.fft.fft2(img)
        img_fft_shifted = np.fft.fftshift(img_fft)

        if type_ == 'lowpass':
            lp_filt = np.zeros((img.shape[0], img.shape[1]))
            lp_filt[xx**2 + yy**2 < radius**2] = 1
            out_img = img_fft_shifted * lp_filt
        elif type_ == 'highpass':
            hp_filt = np.ones((img.shape[0], img.shape[1]))
            hp_filt[xx**2 + yy**2 < radius**2] = 0
            out_img = img_fft_shifted * hp_filt
        else:
            raise ValueError('type_ must be either lowpass or highpass')

        out_img = np.fft.ifftshift(out_img) 
        out_img = np.fft.ifft2(out_img)      


        return np.real(out_img)
        



