import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from hmmlearn import hmm
from sklearn.externals import joblib
import sys


model_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/tutor_models'
data_val_dir='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/b7r16_val/tutor_b7r16/z_sequences_cut'
data_train_dir='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/b7r16/tutor_b7r16/z_sequences_cut'

def n_degFreedom(n_hiddenstates,covariance_type,latent_space_dim=16):
	if covariance_type=='diag':
		return (n_hiddenstates-1)*(1+n_hiddenstates+latent_space_dim+latent_space_dim)
	elif covariance_type=='full':
		return (n_hiddenstates-1)*(1+n_hiddenstates+latent_space_dim+((latent_space_dim+1)*latent_space_dim)/2)
	elif covariance_type=='tied':
		return (n_hiddenstates-1)*(1+n_hiddenstates+latent_space_dim)+((latent_space_dim+1)*latent_space_dim)/2
	elif covariance_type=='spherical':
		return (n_hiddenstates-1)*(1+n_hiddenstates+latent_space_dim+1)		
	else:
		raise ValueError('Unknown covariance type: '+covariance_type)

def get_model_type(model_name):
	split=model_name.split('_')
	n_hidden=int(split[0][:-1])
	emission_type=split[-1][:-4]
	return n_hidden,emission_type	

def get_d_freedom(model_name):
	n_hidden,emission_type=get_model_type(model_name)
	return n_degFreedom(n_hidden,emission_type)

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

def read_data_merge(directory,from_day,to_day):
  #directory is a path to the folder containing the data of all the days
  X=[]
  L=[]
  for i in range(0,int(to_day+1-from_day)):
    current_day=i+from_day
    print('get data from day: '+str(current_day))
    X_i,L_i=read_data(directory+'/day'+str(current_day)+'_b7r16/z_sequences_cut')
    X=X+X_i
    L=L+L_i
  return X,L

def get_n_sequencepoints(directory):
	L=[]
	for sequence in os.listdir(directory):
		if sequence.endswith('.npy'):
			L.append(len(np.load(directory+'/'+sequence).tolist()))
	return sum(L)

def plot_3d_bar(data,title):
	# Plotting
	fig = plt.figure()
	ax = fig.gca(projection = '3d')
	data_shape=data.shape
	H,E=np.meshgrid(np.linspace(1,data_shape[1],data_shape[1]),np.linspace(1,data_shape[0],data_shape[0]))

	Xi = H.flatten()
	Yi = E.flatten()
	Zi = np.zeros(data.size)

	dx =  0.3*np.ones(data.size)
	dy =  0.3*np.ones(data.size)
	dz = data.flatten()
	zero_idx=np.where(dz==0)
	dz[zero_idx]=1e10
	dz = dz-np.min(dz)
	dz[zero_idx]=0
	dz = dz/(2*np.max(dz))
	dz+=0.5
	dz[zero_idx]=0

	ax.set_xlabel('diag - spherical - tied')
	ax.set_ylabel('hidden states')
	ax.set_title(title)
	ax.bar3d(Xi, Yi, Zi, dx, dy, dz, color = 'w')

	plt.show()

def plot_models(models_info):
	h_list=[10,20,30,40,50,60,70,80,90,100]
	e_list=['diag','spherical','tied']
	val_data=np.zeros((len(h_list),len(e_list)))
	AIC_data=np.zeros((len(h_list),len(e_list)))
	AICc_data=np.zeros((len(h_list),len(e_list)))
	BIC_data=np.zeros((len(h_list),len(e_list)))
	for m in range(0,len(models_info)):
		h,e=get_model_type(models_info[m][0])
		i=h_list.index(h)
		j=e_list.index(e)
		val_data[i,j]=models_info[m][1]
		AIC_data[i,j]=models_info[m][2]
		AICc_data[i,j]=models_info[m][3]
		BIC_data[i,j]=models_info[m][4]
	plot_3d_bar(val_data,'val_LL')
	plot_3d_bar(AIC_data,'AIC')
	plot_3d_bar(AICc_data,'AICc')
	plot_3d_bar(BIC_data,'BIC')


val_data,L=read_data(data_val_dir)
n=get_n_sequencepoints(data_train_dir)
print(n)
n_models=len(os.listdir(model_path))
models_info=[]#stores information: 0:modelname, 1:val_score, 2:AIC,3:AICc,4:BIC 4:L
best_val_score=['',-1e10]
for models in os.listdir(model_path):
	dfree=get_d_freedom(models)
	model=joblib.load(model_path+'/'+models)
	val_score=model.score(val_data,L)
	if best_val_score[1]<val_score:
		best_val_score=[models,val_score]
	print(models+', validation score: '+str(val_score))
	AIC=2*(dfree-val_score)
	#I count each sequencepoint as a datapoint (otherwise the punish term becomes negative)
	AICc=AIC+(2*(dfree**2+dfree))/(n-dfree-1)
	BIC=np.log(n)*dfree-2*val_score
	models_info.append([models,val_score,AIC,AICc,BIC,L])
print('best LL model: '+str(best_val_score))
print('plottin...')
plot_models(models_info)



