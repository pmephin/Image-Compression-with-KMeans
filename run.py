from compress_utils import *

image = io.imread('data/test_image.jpeg')  # Select Image

compressed_image=compress_eig(image,k1=128,k2=256)   
#Use `compress_reg` for regular compression and `compress_var` for amplitude variance compression
io.imsave('data/results/compressed_image.jpg',compressed_image)
print(scores(compressed_image,image))
