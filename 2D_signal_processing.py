import numpy as np
from Signalprocessing import *
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os                 

try:
    os.mkdir('out_img')
except:
    None
#'''
#Demonstrate denoising/blurring 
img = cv.imread('images/taj_noisy.jpg')
#'''
#'''
for k in ['gaussianblur3', 'gaussianblur5', 'denoise3', 'denoise5']:
    out_img = Convolution().convolute2d(img=img, kernel=k, pool_type='sum')
    cv.imwrite('out_img/out_img_{}.jpg'.format(k), out_img)
#'''
    
#'''
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 15))
for i, k in enumerate(['gaussianblur3', 'gaussianblur5']):
    out_img = mpimg.imread('out_img/out_img_{}.jpg'.format(k))
    ax[i+1].imshow(out_img)
    ax[i+1].set_title('kernel : {}'.format(k))
    ax[i+1].get_xaxis().set_ticks([]) 
    ax[i+1].get_yaxis().set_ticks([])
ax[0].imshow(img[:,:,::-1])
ax[0].set_title('Original Image')
ax[0].get_xaxis().set_ticks([])
ax[0].get_yaxis().set_ticks([])
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 15)) 
for i, k in enumerate(['denoise3', 'denoise5']): 
    out_img = mpimg.imread('out_img/out_img_{}.jpg'.format(k))     
    ax[i+1].imshow(out_img)                                
    ax[i+1].set_title('kernel : {}'.format(k))             
    ax[i+1].get_xaxis().set_ticks([])                      
    ax[i+1].get_yaxis().set_ticks([])                      
ax[0].imshow(img[:,:,::-1])                                
ax[0].set_title('Original Image')                          
ax[0].get_xaxis().set_ticks([])                            
ax[0].get_yaxis().set_ticks([])                            
fig.tight_layout()                                         
plt.show()                                                 

#comparing denising amd blur kernels
for lst in [['gaussianblur3', 'denoise3'], ['gaussianblur5', 'denoise5']]:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10)) 
    for i, k in enumerate(lst):
        out_img = mpimg.imread('out_img/out_img_{}.jpg'.format(k))
        ax[i].imshow(out_img)
        ax[i].set_title('kernel : {}'.format(k))              
        ax[i].get_xaxis().set_ticks([])                       
        ax[i].get_yaxis().set_ticks([])                
    fig.tight_layout()
    plt.show()
#'''

#'''
#demonstrating edge, sharpen, depth kernels
for img_name in ['fox', 'cat', 'taj']:
    img = cv.imread('images/'+img_name+'.jpeg')

    for k in ['sharpen', 'depth', 'edge']:
        out_img = Convolution().convolute2d(img=img, kernel=k, pool_type='sum')
        cv.imwrite('out_img/out_img_{}_{}.jpeg'.format(img_name, k), out_img)


#'''
#'''

for subname in ['fox', 'cat', 'taj']:

    img = mpimg.imread('images/{}.jpeg'.format(subname))

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    ax[0,0].imshow(img)
    ax[0,0].set_title('Original Image')               
    ax[0,0].get_xaxis().set_ticks([])                        
    ax[0,0].get_yaxis().set_ticks([])                        

    image_name_list =  ['out_img_{}_depth.jpeg'.format(subname), 'out_img_{}_sharpen.jpeg'.format(subname),
                        'out_img_{}_edge.jpeg'.format(subname)]

    indices = [(0,1), (1,0), (1,1)]

    title_list = ['depth', 'sharpen', 'edge']

    for title, ((i, j), image) in zip(title_list, zip(indices, image_name_list)):
        image = mpimg.imread('out_img/'+image)
        ax[i, j].imshow(image)
        ax[i, j].set_title('kernel : '+title)
        ax[i, j].get_xaxis().set_ticks([])  
        ax[i, j].get_yaxis().set_ticks([])
    fig.tight_layout()
    plt.show()

#'''

#''' 

#filtering using fft
for subname in ['fox', 'cat', 'taj']:
    img = cv.imread('images/{}.jpeg'.format(subname))
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 15))
    ax[0].imshow(img[:,:,::-1], cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].get_xaxis().set_ticks([])
    ax[0].get_yaxis().set_ticks([])
    for i, filt in enumerate(['lowpass', 'highpass']):

        radius = 65 if filt=='lowpass' else 20
        
        out_img = fft().filter(img=img, type_=filt, radius=radius)
        '''
        #uncomment to save output to out_img
        cv.imwrite('out_img/out_img_{}_{}.jpeg'.format(subname, filt), out_img)
        '''
        ax[i+1].imshow(out_img, cmap='gray')
        ax[i+1].set_title('{}'.format(filt))
        ax[i+1].get_xaxis().set_ticks([]) 
        ax[i+1].get_yaxis().set_ticks([]) 
    fig.tight_layout()
    plt.show()

