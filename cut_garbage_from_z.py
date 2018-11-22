import numpy as np
import matplotlib.pyplot as plt
import os
from hmmlearn import hmm
from spectrogram_decoder_funcs import *
from sklearn.externals import joblib
import sys

sys.path.append('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/codes_masterthesis')
from utils import *


def get_dimension(sequence):
   if type(sequence)==list:
      D= len(sequence[0])
   else:
      try:
         D= sequence.shape[0]
      except:
         print('The sequence is neither a numpy array nor a list!')
   return D

def read_data(directory):
   print('number of files in folder: '+str(np.size(os.listdir(directory))))
   X=[] #we fill the data in a list
   L=[] #list of length of every sequence
   for filename in os.listdir(directory):
       if filename.endswith(".npy"):
           x=np.load(os.path.join(directory, filename))
           # print('shape: '+str(x.shape))
           x=x.tolist()
           X=X+x
           L.append(len(x))
   D=get_dimension(X)#the dimension of any sequence point
   return X,L,D

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

def cut_garbage_from_latentsequence(z):
  L=z.shape[0]
  l=0
  prev_point=z[0,:]
  next_point=z[1,:]
  while not (prev_point==next_point).all() and l<L-2:
    l+=1
    prev_point=np.copy(next_point)

    next_point=z[l+1,:]
  if l<6:
    print('Warning: there seems to0 be too much garbage in that sequence!')
    # plt.matshow(z[0:l+2,:])
    # plt.show()
  if l==L-2:
    print('Warning: there seems to be no garbage in that sequence!')
    # plt.matshow(z)
    # plt.show()
    l=L
  return z[0:l,:]


# data_directory='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis'
gan_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/nz16_16col'
netG=load_netG(gan_path+'/netG_epoch_44.pth')
netE=load_netE(gan_path+'/netE_epoch_44.pth')
data_directory='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/b7r16_val'
save_directory='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/b7r16_val'


for i in range(0,1):
  # data_dir=data_directory+'/day'+str(i)+'_b7r16/z_sequences'
  # save_path=save_directory+'/day'+str(i)+'_b7r16/z_sequences_cut'
  data_dir=data_directory+'/tutor_b7r16/z_sequences'
  save_path=save_directory+'/tutor_b7r16/z_sequences_cut'
  if not os.path.exists(save_path):
    os.mkdir(save_path)
  print('process day '+str(i)+' ...')
  for filename in os.listdir(data_dir):
    z=np.load(data_dir+'/'+filename)
    z=cut_garbage_from_latentsequence(z)
    np.save(save_path+'/'+filename,z)

# x=np.load(data_directory+'/54783.npy')
# z,reconstructed_samples=get_latent_sequence_from_spectrogram(x,netG,netE)
# np.save(save_path+'/'+'test',z)
# print('shape of original spectrocam: '+str(x.shape))
# print('shape of latent sequence: '+str(z.shape))
# print('shape of reconstructed power-spectrogram: '+str(reconstructed_samples.shape))
# plt.matshow(z)
# plt.show()



