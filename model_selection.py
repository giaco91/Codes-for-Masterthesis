import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from hmmlearn import hmm
from sklearn.externals import joblib
import sys

merge=6

merge_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_6000/m'+str(merge)+'_b7r16/models'
merge_code=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_code_6000.npy')
data_val_dir='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_val'
data_train_dir='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_train'

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
	n_hidden=int(split[1][:-1])
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

def get_n_sequencepoints(directory,from_day,to_day):
	L=[]
	for i in range(0,int(to_day+1-from_day)):
		current_day=i+from_day
		prefix=''
		if current_day<10:
			prefix='0'
		seq_dir=directory+'/day'+prefix+str(current_day)+'_b7r16/z_sequences'
		for img in os.listdir(seq_dir):
			if img.endswith('.npy'):
				L.append(len(np.load(seq_dir+'/'+img).tolist()))
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

def barplot_models(models_info):
	print('plotting...')
	h_list=[40,50,55,60,65,70,75,80,90,100,110,120,130,140]
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

def lineplot_models(models_info, covar_type='diag'):
	print('plotting...')
	n_h=[]
	LL=[]
	AIC=[]
	AICc=[]
	BIC=[]
	for m in range(0,len(models_info)):
		h,e=get_model_type(models_info[m][0])
		if e==covar_type:
			n_h.append(h)
			LL.append(models_info[m][1])
			AIC.append(models_info[m][2])
			AICc.append(models_info[m][3])
			BIC.append(models_info[m][4])

	n_h=np.array(n_h)
	sorted_idx=n_h.argsort()
	LL=np.array(LL)[sorted_idx]
	AIC=np.array(AIC)[sorted_idx]
	AICc=np.array(AICc)[sorted_idx]
	BIC=np.array(BIC)[sorted_idx]
	n_h=np.array(n_h)[sorted_idx]

	fig, ax = plt.subplots(2, 2)
	fig.subplots_adjust(hspace=0.7, wspace=0.7)
	ax[0,0].plot(n_h,LL,'b-')
	ax[0,0].set_title('log-likelihood')
	ax[0,0].set_xlabel('hidden states')
	#ax[0,0].set_ylabel('log-likelihood')
	ax[1,0].plot(n_h,AIC,'b-')
	ax[1,0].set_title('Validation AIC')
	ax[1,0].set_xlabel('hidden states')
	#ax[1,0].set_ylabel('AIC')	
	ax[0,1].plot(n_h,AICc,'b-')
	ax[0,1].set_title('AICc')
	ax[0,1].set_xlabel('hidden states')
	#ax[0,1].set_ylabel('AICc')
	ax[1,1].plot(n_h,BIC,'b-')
	ax[1,1].set_title('BIC')
	ax[1,1].set_xlabel('hidden states')
	#ax[1,1].set_ylabel('BIC')		
	plt.show()



from_day=0
if merge>0:
	from_day=merge_code[merge-1]+1
val_data,L=read_data_merge(data_val_dir,from_day,merge_code[merge])
n=get_n_sequencepoints(data_train_dir,from_day,merge_code[merge])
n_models=len(os.listdir(merge_path))
models_info=[]#stores information: 0:modelname, 1:val_score, 2:AIC,3:AICc,4:BIC 4:L
best_val_score=['',-1e10]
for models in os.listdir(merge_path):
	#print('degree of freedom in '+str(models)+' is: '+str(get_d_freedom(models)))
	dfree=get_d_freedom(models)
	model=joblib.load(merge_path+'/'+models)
	val_score=model.score(val_data,L)
	if best_val_score[1]<val_score:
		best_val_score=[models,val_score]
	print(models+', validation score: '+str(val_score))
	AIC=2*(dfree-val_score)
	#I count each sequencepoint as a datapoint (otherwise the punish term becomes negative)
	AICc=AIC+(2*(dfree**2+dfree))/(n-dfree-1)
	print((2*(dfree**2+dfree))/(n-dfree-1))
	BIC=np.log(n)*dfree-2*val_score
	models_info.append([models,val_score,AIC,AICc,BIC,L])
print('best LL model: '+str(best_val_score))
#barplot_models(models_info)
lineplot_models(models_info)




