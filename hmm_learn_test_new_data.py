import numpy as np
import matplotlib.pyplot as plt
import os

import warnings
from sklearn.externals import joblib
import sys
sys.path.append('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/codes_masterthesis')
from utils import *
from hmmlearn import hmm

#from utils import *

def plot(x,y):
	plt.plot(x, y, ".-", label="observations", ms=6,
         mfc="orange", alpha=0.7)
	plt.show()


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
   print('de-zeropadd...')
   X=[] #we fill the data in a list
   L=[] #list of length of every sequence
   for filename in os.listdir(directory):
       if filename.endswith(".npy"):
           x=np.load(os.path.join(directory, filename))
           #print(x.dtype)
           x=x.tolist()
           X=X+x
           L.append(len(x))
   D=get_dimension(X)#the dimension of any sequence point
   return X,L,D
gan_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/nz16_16col'
netG=load_netG(gan_path+'/netG_epoch_44.pth')
#netE=load_netE(gan_path+'/netE_epoch_44.pth')
#------ train model continuouse------
X,L,D=read_data('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/day01_b7r16/z_sequences')
#np.save('Data/test_data_nozeros',np.asarray(X))
#np.save('Data/test_data_L',np.asarray(L))
#X=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/Data/test_data_nozeros.npy')
#L=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/Data/test_data_L.npy')
model=joblib.load("/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_6000/m6_b7r16/models/m6_70h_gauss_spherical.pkl") #load model
#write covariance_type='diage' if needed
#model=hmm.GaussianHMM(n_components=20, covariance_type="diag",verbose=True) #define model topology

# print('train model....')
# with warnings.catch_warnings():
#   warnings.filterwarnings('ignore', category=DeprecationWarning)
# model.fit(X,L) #fit model
# print('sample from model...')
X_sample, Z_sample = model.sample(50)

X_sample=np.float32(X_sample)
print(type(X_sample))

# zhat_zeros=np.zeros((9,16))
# X_sample=np.float32(zhat_zeros)

print('latent sample sequence dtype: '+str(X_sample.dtype))
print('shape of latent sequence before decoding: '+str(X_sample.shape))
reconstructed_samples, reconstructed_audio = decode(zhat=X_sample, netG=netG)
plt.imshow(reconstructed_samples, origin='lower')
plt.show()
#joblib.dump(model, "Data/test_model.pkl") #store model
#print('Load model...')
#model=joblib.load("Data/test_model.pkl") #load model
#plot(X_sample[:,0],X_sample[:,1])

#-----decoding------
# netG_file_path ='netG_epoch_28.pth'
# opt=get_Opt()
# netG = load_netG(netG_file_path,opt)
# savepath = 'spectrograms'
# if not os.path.exists(savepath):
#     os.makedirs(savepath)   
# reconstruct_audio_from_z_o(X_sample, netG, opt, savepath+'/'+'test_spec','_1')
#----------------






