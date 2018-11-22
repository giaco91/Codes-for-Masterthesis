import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os

import warnings
from sklearn.externals import joblib
import sys
sys.path.append('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/codes_masterthesis')
from utils import *
from hmmlearn import hmm

merge=int(6)
n_h=120

merge_code=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_code_6000.npy')
model_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_6000/m'+str(merge)+'_b7r16/models/m'+str(merge)+'_'+str(n_h)+'h_gauss_diag.pkl'
data_val_dir='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_val'



# gan_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/nz16_16col'
# netG=load_netG(gan_path+'/netG_epoch_44.pth')
# netE=load_netE(gan_path+'/netE_epoch_44.pth')


def get_model_parameters(model_path,type='diag'):
  print('load model....')
  model=joblib.load(model_path)
  startprob=model.startprob_
  transmat=model.transmat_
  means=model.means_
  covars=model.covars_
  return startprob,transmat,means,covars

def plot_covars(means,covars,n_h,n_row=3,n_col=3,data=None):
  if n_row<2 or n_col<2:
    raise ValueError('n_row and n_col must be at least 2')
  x=np.random.randint(16, size=n_row)
  y=np.random.randint(16, size=n_col)
  fig, ax = plt.subplots(n_row, n_col)
  fig.subplots_adjust(hspace=0.7, wspace=0.7)
  for i in range(0,n_row):
    for j in range(0,n_col):
      ells = [Ellipse(xy=means[n,[x[i],y[j]]], width=np.sqrt(covars[n,x[i],x[i]]), height=np.sqrt(covars[n,y[j],y[j]]))
              for n in range(n_h)]
      for e in ells:
        ax[i, j].add_artist(e)
        e.set_clip_box(ax[i, j].bbox)
        e.set_alpha(numpy.random.rand())
        e.set_facecolor(numpy.random.rand(3))
        ax[i, j].set_xlim(-4.5, 4.5)
        ax[i, j].set_ylim(-4.5, 4.5)
      if data is not None:
        ax[i, j].plot(data[:,x[i]],data[:,y[j]],'b.',markersize=0.7,color='black')

  plt.show()

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


def read_data_merge(directory,merge_code,merge):
  #directory is a path to the folder containing the data of all the days
  from_day=0
  if merge>0:
    from_day=merge_code[merge-1]+1
  to_day=merge_code[merge]
  X=[]
  L=[]
  for i in range(0,int(to_day+1-from_day)):
    current_day=i+from_day
    print('get data from day: '+str(current_day))
    X_i,L_i=read_data(directory+'/day'+str(current_day)+'_b7r16/z_sequences_cut')
    X=X+X_i
    L=L+L_i
  return X,L

def plot_means(means):
  x=np.random.randint(16, size=5)
  y=np.random.randint(16, size=5)
  fig, ax = plt.subplots(5, 5)
  fig.subplots_adjust(hspace=0.7, wspace=0.7)
  for i in range(0,5):
      for j in range(0,5):
          ax[i, j].plot(means[:,x[i]],means[:,y[j]],'b.')
  plt.show()

  # for i in range(0,4):
  #   for j in range(5,9):
  #     plt.plot(means[:,i],means[:,j],'b.')
  #     plt.show()

startprob,transmat,means,covars=get_model_parameters(model_path)

val_data,L=read_data_merge(data_val_dir,merge_code,merge)
val_data=np.asarray(val_data)
np.random.shuffle(val_data)
plot_covars(means,covars,n_h,n_row=3,n_col=3,data=val_data[0:200,:])








# print('sample from model...')
# X_sample, Z_sample = model.sample(30)
# X_sample=np.float32(X_sample)
# print('shape of latent sequence before decoding: '+str(X_sample.shape))
# reconstructed_samples, reconstructed_audio = decode(zhat=X_sample, netG=netG)
# plt.imshow(reconstructed_samples, origin='lower')
# plt.show()

#joblib.dump(model, "/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/test_model.pkl") #store model







