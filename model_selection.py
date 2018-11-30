import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from hmmlearn import hmm
from sklearn.externals import joblib
import sys

merge=4

merge_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_6000/m'+str(merge)+'_b7r16/models'
merge_code=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_code_6000.npy')
data_val_dir='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_val'
data_train_dir='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_train'
save_path='/Users/Giaco/Dropbox/SandroGiacomuzzi/Master_Thesis/merge_6000/m'+str(merge)+'_b7r16/selection_graphics'

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
	n_hidden=int(split[-3][:-1])
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

def read_data_sequencelist(directory,sequence_list):
	X=[]
	L=[]
	for sequence in sequence_list:
		x=np.load(os.path.join(directory,sequence))
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
    X_i,L_i=read_data(directory+'/day'+str(current_day)+'_b7r16/z_sequences')
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
	h_list=[40,50,55,60,65,70,75,80,90,100,110,120,130,140,150,160,170,180,190,200]
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

def lineplot_models(models_info, covar_type='diag', save_path=None):
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
	ax[0,0].set_title('LL of validation data')
	ax[0,0].set_xlabel('hidden states')
	#ax[0,0].set_ylabel('log-likelihood')
	ax[1,0].plot(n_h,AIC,'b-')
	ax[1,0].set_title('AIC')
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
	if save_path is not None:
		print('store plot at: '+save_path)
		plt.savefig(save_path+'.eps', format='eps', dpi=1000)
		plt.savefig(save_path+'.png', format='png', dpi=1000)	
	plt.show()

def get_LL(model,sequences,L):
	n_seq=len(L)
	LL=np.zeros(n_seq)
	pointer=0
	for n in range(n_seq):
		LL[n]=model.score(sequences[pointer:pointer+L[n]])/L[n]
		pointer+=L[n]
	mean=np.mean(LL)
	sigma=np.std(LL)/np.sqrt(n_seq)
	return mean,sigma

def filter_modelinfo_by_covartype(models_info,covar_type):
	filtered_models_type=[]
	for model in models_info:
		h,e=get_model_type(model[0])
		if covar_type==e:
			filtered_models_type.append(model)
	return filtered_models_type

def plot_crossvalidation(models_info,covar_type,title='Crossvalidation',save_path=None):
	models_info=filter_modelinfo_by_covartype(models_info,covar_type)
	n_m=len(models_info)
	val_means=[]
	val_sigmas=[]
	train_means=[]
	train_sigmas=[]
	n_h=[]
	for model in models_info:
		h,e=get_model_type(model[0])
		n_h.append(h)
		val_means.append(model[1])
		val_sigmas.append(model[5])
		train_means.append(model[6])
		train_sigmas.append(model[7])
	sorted_idx=np.array(n_h).argsort()
	n_h=np.array(n_h)[sorted_idx]
	val_means=np.array(val_means)[sorted_idx]
	val_sigmas=np.array(val_sigmas)[sorted_idx]
	train_means=np.array(train_means)[sorted_idx]
	train_sigmas=np.array(train_sigmas)[sorted_idx]

	plt.errorbar(n_h,val_means,yerr=val_sigmas,fmt='-o',markersize=2,capsize=3,label="validation set")
	plt.errorbar(n_h,train_means,yerr=train_sigmas,fmt='-o',markersize=2,capsize=3,label="training set")
	leg = plt.legend(loc='best', ncol=1, shadow=True)
	leg.get_frame().set_alpha(0.5)
	plt.xlabel('Number of hidden states')
	plt.ylabel('Normalized log-likelihood')
	plt.title(title)
	if save_path is not None:
		print('store plot at: '+save_path)
		plt.savefig(save_path+'.eps', format='eps', dpi=1000)
		plt.savefig(save_path+'.png', format='png', dpi=1000)
	plt.show()

def calc_model_info_val(model_name,model_path,val_data,L_val,L_train):
	print('evaluate model: '+str(model_name))
	dfree=get_d_freedom(model_name)
	model=joblib.load(model_path+'/'+model_name)
	n=sum(L_train)
	val_score=model.score(val_data)
	print(model_name+', validation score: '+str(val_score))
	AIC=2*(dfree-val_score)
	#I count each sequencepoint as a datapoint (otherwise the punish term becomes negative)
	AICc=AIC+(2*(dfree**2+dfree))/(n-dfree-1)
	BIC=np.log(n)*dfree-2*val_score
	return [model_name,val_score,AIC,AICc,BIC]

def calc_model_info(model_name,model_path,val_data,L_val,L_train,train_data):
	print('evaluate model: '+str(model_name))
	dfree=get_d_freedom(model_name)
	model=joblib.load(model_path+'/'+model_name)
	n=sum(L_train)
	val_mean,val_sigma=get_LL(model,val_data,L_val)
	print(model_name+', validation score: '+str(val_mean))
	AIC=2*(dfree-val_mean)
	#I count each sequencepoint as a datapoint (otherwise the punish term becomes negative)
	AICc=AIC+(2*(dfree**2+dfree))/(n-dfree-1)
	BIC=np.log(n)*dfree-2*val_mean
	train_mean,train_sigma=get_LL(model,train_data,L_train)
	print(model_name+', training score: '+str(train_mean))
	return [model_name,val_mean,AIC,AICc,BIC,val_sigma,train_mean,train_sigma]


def get_model_info(merge,merge_path,merge_code,data_val_dir,data_train_dir,covar_type=None,train=False):
	from_day=0
	if merge>0:
		from_day=merge_code[merge-1]+1
	val_data,L_val=read_data_merge(data_val_dir,from_day,merge_code[merge])
	print('mean of L_val: '+str(np.mean(L_val)))
	train_data,L_train=read_data_merge(data_train_dir,from_day,merge_code[merge])
	print('mean of L_train: '+str(np.mean(L_train)))
	n=sum(L_train)
	print('Total number of sequence points in training set: '+str(n))
	#n_models=len(os.listdir(merge_path))
	models_info=[]#stores information: 0:modelname, 1:val_score, 2:AIC,3:AICc,4:BIC 4:L
	for model_name in os.listdir(merge_path):
		h,e=get_model_type(model_name)
		if covar_type is None:
			if train==False:
				models_info.append(calc_model_info_val(model_name,merge_path,val_data,L_val,L_train))
			else:
				models_info.append(calc_model_info(model_name,merge_path,val_data,L_val,L_train,train_data))
		elif covar_type==e:
			if train==False:
				models_info.append(calc_model_info_val(model_name,merge_path,val_data,L_val,L_train))			
			else:
				models_info.append(calc_model_info(model_name,merge_path,val_data,L_val,L_train,train_data))
	return models_info

def get_modelinfo_in_folder(path_to_folder,val_data,L_val,train_data,L_train,covar_type=None):
	models_info=[]
	for model_name in os.listdir(path_to_folder):
		h,e=get_model_type(model_name)
		if covar_type is None:
			models_info.append(calc_model_info(model_name,path_to_folder,val_data,L_val,train_data,L_train))
		elif covar_type==e:
			models_info.append(calc_model_info(model_name,path_to_folder,val_data,L_val,train_data,L_train))
	return models_info


#--- cv for merge---
models_info=get_model_info(merge,merge_path,merge_code,data_val_dir,data_train_dir,covar_type='diag',train=False)
lineplot_models(models_info, covar_type='diag', save_path=save_path+'/selection_m'+str(merge))
#plot_crossvalidation(models_info,'diag',title='Crossvalidation, merge='+str(merge),save_path=save_path+'/cv_m'+str(merge))

#---cv for single day-----
# day=0
# path_to_folder='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/models_on_single_sequences'
# path_to_folder=path_to_folder+'/day'+str(day)+'_b7r16'
# sequence_list=['54782.npy','54783.npy','54784.npy','54785.npy','54786.npy','54787.npy','54788.npy','54789.npy','54790.npy','54791.npy']
# val_data,L_val=read_data(data_val_dir+'/day'+str(day)+'_b7r16/z_sequences')
# train_data,L_train=read_data_sequencelist(data_train_dir+'/day'+str(day)+'_b7r16/z_sequences',sequence_list)
# models_info=get_modelinfo_in_folder(path_to_folder,val_data,L_val,train_data,L_train,covar_type='diag')
# plot_crossvalidation(models_info,'diag',title='Crossvalidation, day='+str(day)+' ,L=136')



