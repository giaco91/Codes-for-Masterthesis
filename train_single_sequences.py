import numpy as np
import matplotlib.pyplot as plt
import os

import warnings
from sklearn.externals import joblib

from utils import *
from hmmlearn import hmm

day=0
#sequence_list=['54782.npy','54783.npy','54784.npy','54785.npy','54786.npy','54787.npy','54788.npy','54789.npy','54790.npy','54791.npy']

n_iter=1000
tol=0.0001
N_hiddenstates=[5,10,15,20,25,30]
covariance_types=['diag']

gan_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/nz16_16col'
netG=load_netG(gan_path+'/netG_epoch_44.pth')

data_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_train/day'+str(day)+'_b7r16/z_sequences'
model_store_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/models_on_single_sequences'

def read_data(directory):
   print('number of sequences in folder: '+str(np.size(os.listdir(directory))))
   X=[] #we fill the data in a list
   L=[] #list of length of every sequence
   for filename in os.listdir(directory):
       if filename.endswith(".npy"):
           x=np.load(os.path.join(directory, filename))
           x=x.tolist()
           X=X+x
           L.append(len(x))
   return X,L

def read_sequences(directory,sequence_list):
  #directory is the path tho the sequences
  #sequences_list is a list with the sequence names in that directory
  X=[] #we fill the data in a list
  L=[] #list of length of every sequence
  for sequence in sequence_list:
    x=np.load(os.path.join(directory, sequence))
    x=x.tolist()
    X=X+x
    L.append(len(x))
  return X,L

print('load sequences.')
#X,L=read_sequences(data_path,sequence_list)
X,L=read_data(data_path)

  #------ train model continuouse------
for n_hiddenstates in N_hiddenstates:
  for covariance_type in covariance_types:
    Id=str(day)+'d_'+str(sum(L))+'L_'+str(n_hiddenstates)+'h_gauss_'+str(covariance_type)
    model=hmm.GaussianHMM(n_components=n_hiddenstates, covariance_type=covariance_type,verbose=True,n_iter=n_iter,tol=tol) #define model topology
    print('train model: '+Id+' ...')
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=DeprecationWarning)
      model.fit(X,L) #fit model
    print('store model...')
    save_path=model_store_path+'/day'+str(day)+'_b7r16'
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    joblib.dump(model, save_path+'/'+Id+'.pkl') #store model







