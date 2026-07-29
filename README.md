# PPG-denoising
I uploaded a script for PPG noise reduction. \
The most general baseline method is presented in "[the first method](ppg_fft_hr_analysis.py)", which only uses Butterworth bandpass filtering for signal preprocessing and does not use any noise reduction algorithm. The purpose is to explain why a noise reduction algorithm is needed.

In [the secong baseline method](ppg_nlms_acc_fft_hr_analysis.py), I added an NLMS filter to the previous method and added a triaxial accelerometer as a reference to denoise the PPG signal. The denoising effect of this method is average, but it is significantly improved compared to the baseline method.

In [Temko's on-line version](temko_wfpv_database_analysis.py) and [Temko's offline version](temko_wfpv_database_analysis.py), I referenced the publicly available Matlab code provided by Temko in his papers 
> "[Accurate Wearable Heart Rate Monitoring During Physical Exercises Using PPG](https://doi.org/10.1109/TBME.2017.2676243)", IEEE Transactions on Biomedical Engineering, 2017 ,\
> "[Estimation of heart rate from photoplethysmography during physical exercise using wiener filtering and the phase vocoder](https://doi.org/10.1109/EMBC.2015.7318655)," EMBC 2015 , and \
> "[PPG-Based Heart Rate Estimation Using Wiener Filter, Phase Vocoder and Viterbi Decoding](https://doi.org/10.1109/ICASSP.2017.7952309)," ICASSP 2017, to reproduce his work. The purpose is to use this as a standard method for comparison with the previous two methods.\
> The original code of Temko see [Temko's GitHub](https://github.com/andtem2000/PPG).

The [results](PPG_denoising.ipynb) file contains the processing results after calling all the scripts mentioned above.

My work involved converting his original Matlab code to Python and reproducing his work on the public dataset provided by Alessandra Galli: 
> In the folder '[DATABASE](https://github.com/skyxuan7/PPG)'.

Since the original dataset contains 23 records, while the referenced dataset only has 22, the experimental data will differ from the original author's data.
