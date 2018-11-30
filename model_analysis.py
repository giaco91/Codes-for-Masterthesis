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

#------important functions:
#get_model_parameteres
#plot_covars
#plot_means
#get_truncate_transmat
#sample_mc
#plot_state_imporance

def get_model_parameters(model_path):
  print('load model....')
  model=joblib.load(model_path)
  startprob=model.startprob_
  transmat=model.transmat_
  means=model.means_
  covars=model.covars_
  return startprob,transmat,means,covars

def plot_covars(means,covars,n_h,n_row=3,n_col=3,data=None,save_path=None):
  if n_row<2 or n_col<2:
    raise ValueError('n_row and n_col must be at least 2')
  #---for random axis----
  # x=np.random.randint(16, size=n_row)
  # y=np.random.randint(16, size=n_col)
  #---for fixed axis---
  x=np.linspace(0,n_row-1,n_row).astype(int)
  y=np.linspace(n_row,n_row+n_col-1,n_col).astype(int)
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
  if save_path is not None:
    print('store plot at: '+save_path)
    plt.savefig(save_path+'.eps', format='eps', dpi=1000)
    plt.savefig(save_path+'.png', format='png', dpi=1000)
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
    X_i,L_i=read_data(directory+'/day'+str(current_day)+'_b7r16/z_sequences')
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


def truncate_p(p,T):
  if T<0 or T>1:
    raise ValueError('T must be between zero and one. The larger it is, the more variation is in the sample.')
  if np.abs(np.sum(p)-1)>1e-10:
    raise ValueError('p is not a probability vector! Its sum is '+str(np.sum(p)))
  L=p.shape[0]
  p_new=np.zeros(L)
  idx_sort=np.argsort(p)
  if p[idx_sort[-1]]>=T:
    p_new[idx_sort[-1]]=p[idx_sort[-1]]
  else:
    i=0
    while np.sum(p_new)<=T and i<L:
      p_new[idx_sort[-1-i]]=p[idx_sort[-1-i]]
      i+=1
    p_new[idx_sort[-i]]=0
  return p_new/np.sum(p_new)


def get_truncated_transmat(transmat,T):
  K=transmat.shape[0]
  transmat_trunc=np.zeros((K,K))
  for k in range(0,K):
    transmat_trunc[k,:]=truncate_p(transmat[k,:],T)
  return transmat_trunc

def sample_mc(startprob,transmat,N):
  K=transmat.shape[0]
  if startprob.shape[0]!=K:
    raise ValueError('startprob and transmat have inconsistent dimensions')
  sample=np.zeros(N).astype(int)
  hs_idx = np.linspace(0,K-1,K).astype(int)
  sample[0]=np.random.choice(hs_idx, 1, p=startprob)
  for n in range(1,N):
    sample[n]=np.random.choice(hs_idx, 1, p=transmat[sample[0],:])
  return sample

def n_h_in_sample(sample):
  return len(set(sample))

def get_modelpath(path_to_models,merge,n_h):
  return path_to_models+'/m'+str(merge)+'_b7r16/models/m'+str(merge)+'_'+str(n_h)+'h_gauss_diag.pkl'

def plot_state_activity(path_to_models,merges,n_h,N=10,iter=20):
  #the path where your merges are stored
  #merges: a list of integers for the merges that you want to compare
  #n_h: the # of hiddenstates you want to fix for the comparison
  #N, the number of T-steps
  #iter, the resolution of the statistic
  #smooth_out, True if you want to have a less noisy plot
  T_n=np.linspace(0,1,N)
  occurence=np.zeros((N,iter))
  for merge in merges:
    model_path=get_modelpath(path_to_models,merge,n_h)
    startprob,transmat,_,_=get_model_parameters(model_path)
    for n in range(0,N):
      transmat_trunc=get_truncated_transmat(transmat,T_n[n])
      for i in range(0,iter):
        sample=sample_mc(startprob,transmat_trunc,300)
        occurence[n,i]=n_h_in_sample(sample)
    means=np.mean(occurence,axis=1)
    sigmas=np.std(occurence,axis=1)/np.sqrt(iter)
    plt.errorbar(T_n,means,yerr=sigmas,fmt='-o',markersize=2,capsize=3,label="merge=%d"%(merge,))
  leg = plt.legend(loc='best', ncol=1, shadow=True)
  leg.get_frame().set_alpha(0.5)
  plt.xlabel('Temperature')
  plt.ylabel('Number of active states')
  plt.title('Models with '+str(n_h)+' hidden states')
  plt.show()

def sample_from_params(startprob,transmat,means,covars,N,punish_large_variance=False):
  z_dim=means.shape[1]
  sample=np.zeros((N,z_dim))
  if punish_large_variance==False:
    sampled_mc=sample_mc(startprob,transmat,N)
    print('visited states: '+str(n_h_in_sample(sampled_mc)))
    for n in range(0,N):
      sample[n,:]=means[sampled_mc[n],:]
    return sample
  else:
    print('not coded yet')


merge=int(6)
merge_code=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_code_6000.npy')
path_to_models='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_6000'
data_val_dir='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_val'
save_path='/Users/Giaco/Dropbox/SandroGiacomuzzi/Master_Thesis/merge_6000/m'+str(merge)+'_b7r16/parameter_graphics'
val_data,L=read_data_merge(data_val_dir,merge_code,merge)
val_data=np.asarray(val_data)
np.random.shuffle(val_data)

n_h=50
model_path=get_modelpath(path_to_models,merge,n_h=n_h)
startprob,transmat,means,covars=get_model_parameters(model_path)
plot_covars(means,covars,n_h=n_h,n_row=2,n_col=2,data=val_data[0:250,:],save_path=save_path+'/m'+str(merge)+'_'+str(n_h)+'h_covars')


#plot_state_activity(path_to_models,[0,1,2,3,6],130)
# gan_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/nz16_16col'
# netG=load_netG(gan_path+'/netG_epoch_44.pth')
# netE=load_netE(gan_path+'/netE_epoch_44.pth')
# N=50
# T=1
# startprob=truncate_p(startprob,T)
# transmat=get_truncated_transmat(transmat,T)
# z_sample=sample_from_params(startprob,transmat,means,covars,N)

# #-----workaround
# zhat=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_train/day08_b7r16/z_sequences/84052.npy')
# zhat=np.resize(zhat,(N,16))
# zhat[:,:]=z_sample
# #-----
# reconstructed_samples, reconstructed_audio = decode(zhat=zhat, netG=netG)
# plt.imshow(reconstructed_samples, origin='lower')
# plt.show()









