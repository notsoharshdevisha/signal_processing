# signal_processing

* 1D_signalprocessing.py : smoothning or denoising signal using convolution using different window funcions and AI using K Nearest Neighbors algorithm using KNeighbourRegressor.
  * uses earthquake lab sample data from Los Alamos National Laboratory.
* ppg_fft.py : denoising using fft.
  * using PSD threshold values and using Butterworth Filter.
  * uses ppg-signal data from Machine Learning Repository.
* 1D_signal_processing.ipynb : an interactive version of 1D_signal_processing.py and ppg_fft.py.
* Signalprocessing.py : Contains all the classes used in above files.
  * implemented 1D and 2D convolution using numpy.
  * implemented fft on images using numpy for filtering and denoising.
