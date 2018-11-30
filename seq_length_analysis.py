import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from hmmlearn import hmm
from sklearn.externals import joblib
import sys

merges=[0,1,2,3,4,5,6]

#merge_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_6000/m'+str(merge)+'_b7r16/models'
merge_code=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_code_6000.npy')
data_val_dir='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_val'
data_train_dir='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_train'

# def read_data(directory):
#    print('number of sequences in folder: '+str(np.size(os.listdir(directory))))
#    X=[] #we fill the data in a list
#    L=[] #list of length of every sequence
#    for filename in os.listdir(directory):
#        if filename.endswith(".npy"):
#            x=np.load(os.path.join(directory, filename))
#            x=x.tolist()
#            X=X+x
#            L.append(len(x))
#    return X,L

def get_L(directory,from_day,to_day):
	L=[]
	for i in range(0,int(to_day+1-from_day)):
		current_day=i+from_day
		print('process data from day '+str(current_day)+'...')
		seq_dir=directory+'/day'+str(current_day)+'_b7r16/z_sequences'
		for seq in os.listdir(seq_dir):
			if seq.endswith('.npy'):
				L.append(len(np.load(seq_dir+'/'+seq).tolist()))
	return L

def L_barplot(L_list,title='sequence length distribution'):
	resolution=40
	x=np.linspace(0,120,resolution)
	h=np.zeros(resolution)
	for L in L_list:
		i=0
		while L>x[i]:
			i+=1
		h[i]+=1
	plt.bar(x, h, width=0.8,)
	plt.xlabel('sequence length')
	plt.ylabel('amount of sequences')
	plt.title(title)
	plt.show()


merge=6
from_day=0
if merge>0:
	from_day=merge_code[merge-1]+1
to_day=merge_code[merge]
L_train=get_L(data_train_dir,from_day,to_day)
L_barplot(L_train,title='merge='+str(6)+', sequence distribution')

# means_train=[]
# sigmas_train=[]
# means_val=[]
# sigmas_val=[]
# for merge in merges:
# 	print('processing merge '+str(merge)+'...')
# 	from_day=0
# 	if merge>0:
# 		from_day=merge_code[merge-1]+1
# 	print('analyise trainging set...')
# 	L_train=get_L(data_train_dir,from_day,merge_code[merge])
# 	print('analyse validation set...')
# 	L_val=get_L(data_val_dir,from_day,merge_code[merge])
# 	means_train.append(np.mean(L_train))
# 	sigmas_train.append(np.std(L_train))
# 	means_val.append(np.mean(L_val))
# 	sigmas_val.append(np.std(L_val))
# plt.errorbar(merges,means_val,yerr=sigmas_val,fmt='o',markersize=2,capsize=3,label="validation set")
# plt.errorbar(merges,means_train,yerr=sigmas_train,fmt='o',markersize=2,capsize=3,label="training set")
# leg = plt.legend(loc='best', ncol=1, shadow=True)
# leg.get_frame().set_alpha(0.5)
# plt.xlabel('merge')
# plt.ylabel('points per sequence')
# plt.title('Sequence lengths statistic')
# plt.show()







