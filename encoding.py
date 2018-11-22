import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/codes_masterthesis')

from utils import *

from_day=27
to_day=28

gan_path='/nz16_16col'
data_dir=''
netG=load_netG(gan_path+'/netG_epoch_44.pth')
netE=load_netE(gan_path+'/netE_epoch_44.pth')

def get_latent_sequence_from_spectrogram(x,netG,netE):
	#x is a spectrogram of shape (129,*,2)
	z, reconstructed_samples=encode_and_decode(x, netE, netG, batch_size=64, method=1, \
                      imageH=129, imageW=16, transform_sample=True, return_tensor=False)
	#the length L of the sequence z with shape (L,16) is fixed to minimum 64. We need to cut
	#out the informative part
	L_informative=int(np.floor(x.shape[1]/16))
	if L_informative!=(reconstructed_samples.shape[1]/16):
		print('Something might be wrong with the encoding, resp. with the length of informative sequence.')
	return z[0:L_informative,:], reconstructed_samples

for current_day in range(from_day,to_day+1):
  prefix=''
  if current_day<10:
    prefix='0'

  data_directory=data_dir+'/day'+prefix+str(current_day)+'_b7r16/images'
  save_path=data_dir+'/day'+prefix+str(current_day)+'_b7r16/z_sequences'
  if not os.path.exists(save_path):
    os.mkdir(save_path)
  for filename in os.listdir(data_directory):
    x=np.load(data_directory+'/'+filename)
    z,reconstructed_samples=get_latent_sequence_from_spectrogram(x,netG,netE)
    print(z.shape)
    np.save(save_path+'/'+filename,z)






