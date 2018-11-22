import numpy as np
import matplotlib.pyplot as plt
import os

import warnings
from sklearn.externals import joblib

from utils import *
from hmmlearn import hmm

data_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/b7r16/tutor_b7r16/z_sequences_cut'
save_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/tutor_test_train'
#model_store_path='/cluster/home/sandrog/master_thesis/tutor'
n_iter=1
tol=0.01
covariance_types=['tied','diag','spherical']
N_hiddenstates=[40,50]


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

  
gan_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/nz16_16col'
#gan_path='/cluster/home/sandrog/master_thesis/nz16_16col'
netG=load_netG(gan_path+'/netG_epoch_44.pth')
#------ train model continuouse------
X,L=read_data(data_path)
#write covariance_type='diage' if needed

#------ train models------
for n_hiddenstates in N_hiddenstates:
  for covariance_type in covariance_types:
    start_time=time.time()
    Id=str(n_hiddenstates)+'h_gauss_'+str(covariance_type)
    model=hmm.GaussianHMM(n_components=n_hiddenstates, covariance_type=covariance_type,verbose=True,n_iter=n_iter,tol=tol) #define model topology
    print('train model: '+Id+' ...')
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=DeprecationWarning)
      model.fit(X,L) #fit model

    print('training finished after time: '+str(time.time()-start_time))
    print('store model...')
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    joblib.dump(model, save_path+'/'+Id+'.pkl') #store model






