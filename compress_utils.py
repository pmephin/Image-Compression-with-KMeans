from skimage import io
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from numba import njit

#from sklearn.cluster import KMeans  ## enable for cpu only computation

from cuml.cluster import KMeans
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import warnings
warnings.simplefilter(action='ignore')


def scores(new_im,image):
    print(f'PSNR = {peak_signal_noise_ratio(image,new_im)}')
    print(f'SSIM = {structural_similarity(image,new_im,win_size=3)}')
    
def flatten_image(image):
    rows=image.shape[0]
    cols=image.shape[1]

    flat_im=image.reshape(rows*cols,3).copy()
    return flat_im   


#------------#Regular KMeans compression#---------------#
def compress_reg(image,k):
    km=KMeans(n_clusters=k)
    flat_im=flatten_image(image)
    km.fit(flat_im.astype('float32'))
    
    ## finding K centroids and assigning centroid color to the members
    compressed_im=km.cluster_centers_[km.labels_]

    compressed_im=compressed_im.astype('uint8')

    compressed_im=compressed_im.reshape(image.shape)

    
    return compressed_im

#------------#Compression Through Covaraiant matrices#---------------#
@njit
def mean_eig_cov(A):
    return np.mean(la.eigvalsh(np.cov(A,rowvar=False)))

## divides up the image into chunks of 4x4 pixels and assigns a numerical index to each pixel
def assign_chunks_to_var(chunk_index,img_slices,Indices):
    chunks=[];indices=[]
    for i in chunk_index:
        chunks.append(img_slices[i])
        indices.append(Indices[i])
    return chunks,indices

## matches the image chunks with its respective indices
## divides up the image into chunks of 4x4 pixels and assigns a numerical index to each pixel
def slice_image(image,r=4,c=4):
    
    rows,cols = image.shape[:2]
    while True:
        if (rows%r==0 or rows%r>1) :        #making sure there arent any 1x1 chunks
            break
        else: r+=1

    while True:
        if cols%c==0 or cols%c>1 :
            break
        else: c+=1

    indices=np.arange(0,rows*cols).reshape(rows,cols)
    l=[];m=0;n=0;I=[]
    while n<rows:
        m=0
        while m<cols:
            l.append(image[n:n+r,m:m+c])
            I.append(indices[n:n+r,m:m+c])
            m+=c
        n+=r
    return l,I

##concatenates inhomogenous arrays in a list
def concat_image_blocks(image_blocks):
    temp_list=[np.vstack(block) for block in image_blocks]
    return (np.vstack(temp_list))

##concatenates inhomogenous indice blocks in a list
def concat_indices(indices):
    temp_list=[index_blocks.flatten() for index_blocks in indices]
    return(np.hstack(temp_list))
## Function to compress image using k1 and k2 clusters after 
## a discriminator(eg: mean eigenvalues in the case of compression using covariance matrix) has been found

def compress_kmeans(discriminator,img_slices,Indices,k1,k2,image):
    print('Compressing Image...')
    
     ## dividing the image into 'low variance aka low' and 'high varaince aka high' regions
        
    index=np.arange(len(discriminator))
    index_l=index[discriminator<np.median(discriminator)] 
    index_h=index[discriminator>=np.median(discriminator)]
    
    ##setting up kmeans and assigning indices to chunks
    
    kml=KMeans(n_clusters=k1)
    kmh=KMeans(n_clusters=k2)
    low_chunks,low_indices=assign_chunks_to_var(index_l,img_slices,Indices)
    high_chunks,high_indices=assign_chunks_to_var(index_h,img_slices,Indices)

    ## stacking the list of chunks and indices row-wise
    high_chunks=concat_image_blocks(high_chunks)
    low_chunks=concat_image_blocks(low_chunks)

    
    ##clustering and assigning centroid colors to the cluster
    
    kml.fit(low_chunks.astype('float32'))  ##cuml only works with either 'float64' or 'float32'
    kmh.fit(high_chunks.astype('float32'))
    comp_low_chunk=kml.cluster_centers_[kml.labels_].astype('uint8')
    comp_high_chunk=kmh.cluster_centers_[kmh.labels_].astype('uint8')
    
    ##stacking indices
    low_indices=concat_indices(low_indices)
    high_indices=concat_indices(high_indices)
    
    ##reconstructing the image after compression
    new_image=np.zeros(image.shape)
    new_image=flatten_image(new_image)
    new_image[low_indices]=comp_low_chunk
    new_image[high_indices]=comp_high_chunk
    new_image=new_image.reshape(image.shape).astype('uint8')
    print('Done')
    return new_image

def compress_eig(image,k1=64,k2=256):
    print('Preparing Image...')
    img_slices,Indices=slice_image(image)
    
     ## Computing the average mean eigenvalues of covaraince matrices within each chunk
    avg_eig=[mean_eig_cov(flatten_image(block)) for block in img_slices]
           
    return(compress_kmeans(avg_eig,img_slices,Indices,k1,k2,image))


#------------#Compression Through Total Variance in the Amplitude Space#---------------#
def compress_var(image,k1=64,k2=256,TV=False):
    
    print('Preparing Image...')
    ## conversion to amplitude space
    gray_im=rgb2gray(image)

    img_slices,Indices=slice_image(image)
    gray_img_slices,gray_Indices=slice_image(gray_im)

    ## discriminant can be wither total variance using |grad(image)| or average variance withing each chunk
    if TV:
        variances=[]
        for img in gray_img_slices:
            gx,gy=np.gradient(img,axis=1),np.gradient(img,axis=0)
            variances.append(np.sum(np.sqrt(gx**2+gy**2)))
    else:        
        variances=[np.var(img) for img in gray_img_slices]
        
    return (compress_kmeans(variances,img_slices,Indices,k1,k2,image))
