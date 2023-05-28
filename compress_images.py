from compress_utils import *
import pandas as pd

image=io.imread('data/test_image.jpeg')  ##Image path

K_vals=[32,64,128]  #k1 vals

res_reg={'k_vals':K_vals,'PSNR':[],'SSIM':[]}
res_eig={'k_vals':K_vals,'PSNR':[],'SSIM':[]}
res_var={'k_vals':K_vals,'PSNR':[],'SSIM':[]}

for k in K_vals:
    
    im=compress_reg(image,k)
    io.imsave(f'data/results/reg_k{k}.jpeg',im)
    res_reg['PSNR'].append(peak_signal_noise_ratio(image,im))
    res_reg['SSIM'].append(structural_similarity(image,im,win_size=3))    

    im=compress_eig(image,k1=k)
    io.imsave(f'data/results/eig_k{k}.jpeg',im)
    res_eig['PSNR'].append(peak_signal_noise_ratio(image,im))
    res_eig['SSIM'].append(structural_similarity(image,im,win_size=3))
    
    im=compress_var(image,k1=k)
    io.imsave(f'data/results/var_k{k}.jpeg',im)
    res_var['PSNR'].append(peak_signal_noise_ratio(image,im))
    res_var['SSIM'].append(structural_similarity(image,im,win_size=3))    
    
pd.DataFrame(res_reg).to_csv('data/results/res_reg.csv',index=False)
pd.DataFrame(res_eig).to_csv('data/results/res_eig.csv',index=False)
pd.DataFrame(res_var).to_csv('data/results/res_var.csv',index=False)