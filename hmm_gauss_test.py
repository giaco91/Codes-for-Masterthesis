import numpy as np
import matplotlib.pyplot as plt
import os

import warnings
from sklearn.externals import joblib

from utils import *
from hmmlearn import hmm

merge=int(1)
n_iter=2
tol=0.01
N_hiddenstates=[2,3,4,5]
covariance_types=['diag','spherical','tied']

gan_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/nz16_16col'
netG=load_netG(gan_path+'/netG_epoch_44.pth')

data_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_train'
model_store_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_6000'
merge_code=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_code_6000.npy')

def n_degFreedom(n_hiddenstates,covariance_type,latent_space_dim=16):
  if covariance_type=='diag':
    return (n_hiddenstates-1)*(1+n_hiddenstates+2*latent_space_dim)
  elif covariance_type=='full':
    return (n_hiddenstates-1)*(1+n_hiddenstates+latent_space_dim+np.power(latent_space_dim,2))
  else:
    raise ValueError('The covariance_type must be either diag or full!')

def read_data(directory):
   print('number of files in folder: '+str(np.size(os.listdir(directory))))
   X=[] #we fill the data in a list
   L=[] #list of length of every sequence
   for filename in os.listdir(directory):
       if filename.endswith(".npy"):
           x=np.load(os.path.join(directory, filename))
           x=x.tolist()
           X=X+x
           L.append(len(x))
   return X,L

def read_data_merge(directory,from_day,to_day):
  #directory is a path to the folder containing the data of all the days
  X=[]
  L=[]
  for i in range(0,int(to_day-from_day)):
    current_day=i+from_day
    prefix=''
    if current_day<10:
      prefix='0'
    X_i,L_i=read_data(directory+'/day'+prefix+str(current_day)+'_b7r16/z_sequences')
    X=X+X_i
    L=L+L_i
  return X,L



print('load data from '+str(merge)+'...')
if merge==0:
  from_day=0
else:
  from_day=merge_code[merge-1]+1
to_day=merge_code[merge]+1

X,L=read_data_merge(data_path,from_day,to_day)


  #------ train model continuouse------
for n_hiddenstates in N_hiddenstates:
  for covariance_type in covariance_types:
    Id='m'+str(merge)+'_'+str(n_hiddenstates)+'h_gauss_'+str(covariance_type)
    model=hmm.GaussianHMM(n_components=n_hiddenstates, covariance_type=covariance_type,verbose=True,n_iter=n_iter,tol=tol) #define model topology
    print('train model: '+Id+' ...')
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=DeprecationWarning)
      model.fit(X,L) #fit model
    
    # print('store model...')
    # save_path=model_store_path+'/m'+str(merge)+'_b7r16'
    # if not os.path.exists(save_path):
    #   os.mkdir(save_path)
    # save_path+='/models'
    # if not os.path.exists(save_path):
    #   os.mkdir(save_path)
    # joblib.dump(model, save_path+'/'+Id+'.pkl') #store model







