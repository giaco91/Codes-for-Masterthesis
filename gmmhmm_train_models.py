import numpy as np
import matplotlib.pyplot as plt
import os
import time
import warnings
from sklearn.externals import joblib
from utils import *
from hmmlearn import hmm

merge=int(6)
N_hiddenstates=[2]
n_mixes=[2]
covariance_types=['diag']
init_from_gauss=False
#covariance_types=['tied','diag','spherical']

n_iter=400
tol=0.001

# data_path='/cluster/home/sandrog/master_thesis/data_masterthesis'
# model_store_path='/cluster/home/sandrog/master_thesis/models_gmmhmm'
# merge_code=np.load('/cluster/home/sandrog/master_thesis/merge_code_6000.npy')

gauss_model_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_6000'
data_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_train'
model_store_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/models_gmmhmm'
merge_code=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_code_6000.npy')



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
    print('read data from day: '+str(current_day))
    X_i,L_i=read_data(directory+'/day'+prefix+str(current_day)+'_b7r16/z_sequences')
    X=X+X_i
    L=L+L_i
  return X,L

def get_model_parameters(model_path):
  print('load model....')
  model=joblib.load(model_path)
  startprob=model.startprob_
  transmat=model.transmat_
  means=model.means_
  covars=model.covars_
  return startprob,transmat,means,covars

def get_neighbors(h, means):
  center=means[h]
  d=np.square(means-center)
  d=np.sum(d,axis=1)
  return np.argsort(d)

def check_similarity(h,neighbor,top_flop):
  h_tf=top_flop[h]
  n_tf=top_flop[neighbor]
  return set(h_tf[0]).isdisjoint(n_tf[1]) and set(h_tf[1]).isdisjoint(n_tf[0])

def get_top_flop(h,transmat,crit=6):
  #the higher crit, the harder is the top-flop criterion
  cond=transmat[:,h]
  idx=np.argsort(cond)
  top=idx[-crit:]
  flop=idx[0:crit]
  return [top,flop]

def paired_states_to_set(paired_states,n_h):
  pairs = []
  for i in range(0,n_h):
    if not {i,paired_states[i]} in pairs:
      pairs.append({i,paired_states[i]})
  return pairs

def get_params_from_paired_states(paired_states,startprob,transmat,means,covars):
  n_h=means.shape[0]
  new_n_h=int(n_h/2)
  n_dim=means.shape[1]
  pairs=paired_states_to_set(paired_states,n_h)
  new_startprob=np.zeros(new_n_h)
  new_transmat=np.zeros((new_n_h,new_n_h))
  new_means=np.zeros((new_n_h,2,n_dim))
  new_covars=np.zeros((new_n_h,2,n_dim,n_dim))
  for i in range(0,len(pairs)):
    pair=list(pairs[i])
    new_startprob[i]=startprob[pair[0]]+startprob[pair[1]]
    new_means[i,0,:]=means[pair[0],:]
    new_means[i,1,:]=means[pair[1],:]
    new_covars[i,0,:,:]=covars[pair[0],:,:]
    new_covars[i,1,:,:]=covars[pair[1],:,:]
    for j in range(0,len(pairs)):
      next_pair=list(pairs[j])
      new_transmat[i,j]=transmat[pair[0],next_pair[0]]+transmat[pair[0],next_pair[1]]+transmat[pair[1],next_pair[0]]+transmat[pair[1],next_pair[1]]
  new_transmat/=2
  return new_startprob,new_transmat,new_means,new_covars

def get_initial_parameters(model_path,n_mix=2):
  if n_mix!=2:
    raise ValueError('for n_mix='+str(n_mix)+'I have no parameter initialization code written yet:(')
  startprob,transmat,means,covars=get_model_parameters(model_path)
  n_h=means.shape[0]
  n_dim=means.shape[1]
  if n_h%2!=0:
    raise ValueError('number of hiddenstates must be even')
  paired_states=[]#here I store the partner of every state
  top_flop=[]#here I save for every state the top and flop incomings and top outcomings
  for h in range(0,n_h):
    top_flop.append(get_top_flop(h,transmat,crit=6))
  j=0
  for h in range(0,n_h):
    #check if h has already a partner
    if h in paired_states:
      partner=[i for i,x in enumerate(paired_states) if x == h][0]
      paired_states.append(partner)
      if len(paired_states) > len(set(paired_states)):
        raise ValueError('list has double elements:'+str(paired_states))
    #find good partner
    else:
      partner_found=False
      neighbors=get_neighbors(h, means)
      i=0
      while partner_found==False:
        if not neighbors[i] in paired_states and check_similarity(h,neighbors[i],top_flop)and h!=neighbors[i] and neighbors[i]>h:
          partner_found=True
          paired_states.append(neighbors[i])
          if len(paired_states) > len(set(paired_states)):
            raise ValueError('list has double elements (while loop):'+str(paired_states))
        i+=1
        if i>n_h-1:
          #failed to find a matching partner, take nearest
          print('no good partner found')
          k=0
          while partner_found==False:
            if not neighbors[k] in paired_states and h!=neighbors[k] and neighbors[k]>h:
              paired_states.append(neighbors[k])
              partner_found=True
            k+=1
  return get_params_from_paired_states(paired_states,startprob,transmat,means,covars)
  


print('load data...')
if merge==0:
  from_day=0
else:
  from_day=merge_code[merge-1]+1
to_day=merge_code[merge]+1
X,L=read_data_merge(data_path,from_day,to_day)

  #------ train GMMHMM------
for n_hiddenstates in N_hiddenstates:
  for covariance_type in covariance_types:
    for n_mix in n_mixes:
      start_time=time.time()
      Id='m'+str(merge)+'_'+str(n_hiddenstates)+'h_'+str(n_mix)+'mix_gmmhmm_'+str(covariance_type)
      model=hmm.GMMHMM(n_components=n_hiddenstates,n_mix=n_mix,covariance_type=covariance_type,verbose=True,n_iter=n_iter,tol=tol) #define model topology
      #---initialize parameters
      model_path=gauss_model_path+'/m'+str(merge)+'_b7r16/models/m'+str(merge)+'_'+str(n_hiddenstates*2)+'h_gauss_diag.pkl'
      if n_mix==2 and os.path.exists(model_path) and init_from_gauss:
        print('initializing parameters from trained Gaussian HMM...')
        new_startprob,new_transmat,new_means,new_covars=get_initial_parameters(model_path)
        model.startprob_=new_startprob
        model.transmat_=new_transmat
        model.means_=new_means
        model.covars=new_covars
        #the weights are not initialized
      
      print('train model: '+Id+' ...')
      with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        model.fit(X,L) #fit model
      print('training finished after time: '+str(time.time()-start_time))
      print('store model...')
      if not os.path.exists(model_store_path):
        os.mkdir(model_store_path)
      save_path=model_store_path+'/m'+str(merge)+'_b7r16'
      if not os.path.exists(save_path):
        os.mkdir(save_path)
      save_path+='/models'
      if not os.path.exists(save_path):
        os.mkdir(save_path)
      joblib.dump(model, save_path+'/'+Id+'.pkl') #store model







