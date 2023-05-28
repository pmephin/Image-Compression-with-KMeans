# Improved Image Compression with KMeans

Image compression is a technique that reduces the size of an image file without significantly affecting its quality. One of the methods for image compression is k-means clustering, which partitions the image pixels into k clusters based on their colors. Each pixel is then assigned to the cluster with the closest mean color, and the image is reconstructed using only k colors. This reduces the amount of information needed to store and transmit the image, while preserving its main features. This method of compression has a downside as it compresses an image indiscriminately ie, without taking care of regions with high or low detail(spatial frequencies).

In this work, I seek to enhance the quality of image compression based on KMeans clustering. The main idea is to segment the image into regions that differ in their level of detail and apply different compression rates to each region. This way, the regions with high detail are preserved better, while the regions with low detail are compressed more efficiently. 

## 1. Models

- Compression based on Eigenvalues of Covariant matrices
- Compression based on Total varaintion in Amplitude Space
### 1.1 Compression based on Eigenvalues of Covariant matrices
This method involves the following steps:
- Split the image into many rectangular blocks or chunks
- Compute the eigenvalues of the covariance matrices of each block
- Use the eigenvalues to measure the variance in each block
- Classify the blocks into high-variance and low-variance regions based on the mean eigenvalue of each block
- Choose the median eigenvalue as the threshold for the classification
- Compress the low variance regions with **$k_1$** clusters and high variance regions with **$k_2$** clusters, where **$k_1 < k_2$**
### 1.2 Compression based on Total varaintion in Amplitude Space
- Transform the image to grayscale and obtain its amplitude space.
- Divide the image into rectangular blocks of equal size.
- Compute the variance within each block using one of two possible metrics
  - The average statistical variance in the amplitude within each block
  - The norm of the two dimensional image gradient within each image block, $g_i(x,y)$ $$TV=|\nabla{g_i} |$$
- Choose the median metric as the threshold for the classification
- Compress the low variance regions with **$k_1$** clusters and high variance regions with **$k_2$** clusters, where **$k_1 < k_2$**
    
## 2. Notable Observations and Results
Test Image Used:

<p align="center">
  
<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/dc35ba36-931d-4ea5-89dd-79ee1ebe039f" width="500" height="400">
  
</p>

### 2.1 Variance Classification Performance 

Eigenvalue Classification             |  Amplitude Variation Classification
:-------------------------:|:-------------------------:
<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/ca33a6bc-ff72-4ab8-bbeb-4df18fa32ad2" width="400" height="300">|<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/b2d68a79-9c4c-4484-9abc-e940fc1666a4" width="400" height="300">

- Here the black pixels represents low variance regions
- Both the methods appears to work well with the test image
- Eigenvalue method should be generally preferred since Amplitude variation cannot account for changes in colors with same amplitude intensities


## 2.2 Compression Performance
No. of clusters($k_1$)| Regular KMeans |  Eigenvalue Enhanced  | Amp Variation Enhanced|Compression Rate
:-----------------|----------------|-----------------------------|---------------------|--------------:
 32|<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/dd5ae0e2-525a-4166-9115-0293114dab26" width="200" height="150">|<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/bc0f90d5-2ee0-431e-8a9e-392e71145387" width="200" height="150">|<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/153c7192-ad4d-42d0-9f8f-8dae5ff77655" width="200" height="150">| 70%
64|<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/750c9853-20b8-48ac-ba60-941dbca7284b" width="200" height="150">|<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/f363595c-91e7-497c-b8f6-997d31b24279" width="200" height="150">|<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/25875411-abdc-4433-89b9-96a759978424" width="200" height="150">|72%
128|<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/6afc78dd-f3cc-4a36-8538-c51d3022dc7b" width="200" height="150">|<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/fc4c3464-563b-401f-9d17-ffccacb1f236" width="200" height="150">|<img src="https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/8f1e81ca-527e-4a32-b495-4ef1beb1cd79" width="200" height="150">|73%

- Here $k2 =256$ for all cases
- We do see noticeable improvements in the picture quality with the enhanced methods at the same compression rate

For quantitative measurement of compression quality, I am electing to use PSNR(Peak Signal-to-Noise Ratio) and SSIM(Structural Similarity Index). 
- PSNR is calculated by using the mean squared error (MSE) between the original and distorted images. A higher PSNR value indicates a higher similarity between the images.
- SSIM is designed to mimic the human visual system and capture the perceptual quality of images. SSIM is calculated by using local statistics of pixel intensities in a sliding window. A higher SSIM value indicates a higher structural similarity between the images.

![graph](https://github.com/pmephin/Image-Compression-with-KMeans/assets/134229875/378aef12-8dac-41f1-bf22-c6f854db55d5)

- For the given image, PSNR and SSIM values show significant improvement going from Regular KMeans to Enhanced Models
