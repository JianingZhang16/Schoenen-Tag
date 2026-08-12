#PPG-denoising

I uploaded a script for PPG noise reduction.
The most general baseline method is presented in "the first method", which only uses Butterworth bandpass filtering for signal preprocessing and does not use any noise reduction algorithm. The purpose is to explain why a noise reduction algorithm is needed.

In the secong baseline method, I added an NLMS filter to the previous method and added a triaxial accelerometer as a reference to denoise the PPG signal. The denoising effect of this method is average, but it is significantly improved compared to the baseline method.

In Temko's on-line version and Temko's offline version, I referenced the publicly available Matlab code provided by Temko in his papers

"Accurate Wearable Heart Rate Monitoring During Physical Exercises Using PPG", IEEE Transactions on Biomedical Engineering, 2017 ,
"Estimation of heart rate from photoplethysmography during physical exercise using wiener filtering and the phase vocoder," EMBC 2015 , and
"PPG-Based Heart Rate Estimation Using Wiener Filter, Phase Vocoder and Viterbi Decoding," ICASSP 2017, to reproduce his work. The purpose is to use this as a standard method for comparison with the previous two methods.
The original code of Temko see Temko's GitHub.
The results file contains the processing results after calling all the scripts mentioned above.

My work involved converting his original Matlab code to Python and reproducing his work on the public dataset provided by Alessandra Galli:

In the folder 'DATABASE'.
Since the original dataset contains 23 records, while the referenced dataset only has 22, the experimental data will differ from the original author's data.

