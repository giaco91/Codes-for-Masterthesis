import numpy as np
import os

n_seq_merge=6000

n_seq=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/n_seq_train.npy')
#the merge_code tells up to which day is merged. 
#E.g. let i=merg_code[0], then the first merge is from day 0 up to day i
#let j=merge_code[2], then  the 3rd merge is from day merge_code[2-1]+1 up to day j
#in general: the k-th merge is from day merge_code[k-2]+1 up do day merge_code[k-1]
merge_code=[]
seq_counter=0
n=0
for i in n_seq:
	i=int(i)
	seq_counter+=i
	if seq_counter>n_seq_merge:
		if seq_counter-n_seq_merge<n_seq_merge-(seq_counter-i):
			merge_code.append(n)
		else:
			if n-1==0:
				raise ValueError('the number of sequences to be merged is not big enough')
			elif len(merge_code)==0:
				merge_code.append(n-1)	
			elif n-1==merge_code[-1]:
				merge_code.append(n)
				print('Warning! the '+str(n)+'th day has much more sequences than the numberof sequences to be merged.')
			else:
				merge_code.append(n-1)
		seq_counter=0

	n+=1
if merge_code[-1]!=len(n_seq)-1:
	merge_code.append(len(n_seq)-1)

print(merge_code)
np.save('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/merge_code_'+str(n_seq_merge),merge_code)