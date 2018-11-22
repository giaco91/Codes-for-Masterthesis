import numpy as np
import matplotlib.pyplot as plt
import os
from hmmlearn import hmm
#from spectrogram_decoder_funcs import *
from sklearn.externals import joblib
import sys
sys.path.append('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/codes_masterthesis')
from utils import *

#load the GAN
gan_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/nz16_16col'
netG=load_netG(gan_path+'/netG_epoch_44.pth')
netE=load_netE(gan_path+'/netE_epoch_44.pth')

#load a spectrogram
x=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/day47_b7r16/images/41408.npy')

#encode the spectrogram to the latent space
zhat, reconstructed_samples_0=encode_and_decode(x, netE, netG, batch_size=64, method=1, \
                      imageH=129, imageW=16, transform_sample=True, return_tensor=False)
#cut the latent sequece
zhat=zhat[:int(np.floor(x.shape[1]/16)),:]
print(type(zhat))
print('zhat shape after cutting: '+str(zhat.shape))
print('zhat dtype after cutting: '+str(zhat.dtype))
zhat_zeros=np.zeros((9,16))
zhat=np.float32(zhat_zeros)
# zhat[:,:]=zhat_zeros[:,:]
# print(zhat.dtype==zhat_zeros.dtype)
# print(zhat.dtype)
# print(zhat_zeros.dtype)

#---decode the latent sequence back to a power spectrogram
reconstructed_samples, reconstructed_audio = decode(zhat=zhat, netG=netG)
print('reconstructed power spectrogram shape: '+str(reconstructed_samples.shape))

plt.imshow(reconstructed_samples, origin='lower')
plt.show()






