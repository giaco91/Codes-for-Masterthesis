
import sys, os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import soundfile as sf
import wave
from sklearn.model_selection import train_test_split
from scipy.signal import resample
import librosa.core as lc
from librosa.util import fix_length
from sklearn.utils import shuffle
from scipy.misc import imsave
from datetime import datetime as dt
from PIL import Image
import librosa
import sounddevice as sd
import pickle
from sklearn.model_selection import train_test_split
from os.path import join
from pomegranate import *
import pandas as pd
import scipy as sc
import pdb

#from networks_audio_nophase_8col import _netG, _netE, weights_init
from networks_16col import _netG, _netE, weights_init



if not sys.platform=='linux':
    import librosa.display
    import matplotlib.pyplot as plt

def deZeropadd(data):
    #input: data.shape=(sequence_length, sequence_dimension)
    data_shape = data.shape
    while np.sum(np.absolute(data[-1,:]))==0:
        data=data[0:-1,:]    
    return data

def closest_centroids(centroids,x):
    #input x is a sequence in the 16-dim latent space
    #output is a sequence of the represenative centroids of the k-mean clustering
    x=np.asarray(x)
    L=x.shape[0]
    y=np.zeros(L)
    for i in range(L):
        represenative=0
        dist=1e10
        for k in range(centroids.shape[0]):
            d = numpy.linalg.norm(centroids[k,:]-x[i,:])
            if d<dist:
                dist=d
                represenative=k
        y[i]=represenative
    return y

def k_mean_encoding(centroids,directory):
    print('number of files in folder: '+str(np.size(os.listdir(directory))))
    print('de-zeropadd...')
    X=[]
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            x=np.load(os.path.join(directory, filename))
            x=deZeropadd(x)
            x=closest_centroids(centroids,x).tolist()
            X.append(x)
    return X

def get_disc_distribution(centroids,directory):
    #centroids is a list of k-means
    #directory contains sequences
    #the functions output is the emp. distribution the centroids
    #given the sequences
    print('get distribution from '+str(directory))
    C=centroids.shape[0]
    distribution=np.zeros(C)
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            x=np.load(os.path.join(directory, filename))
            x=np.asarray(deZeropadd(x))
            L=x.shape[0]
            for i in range(L):
                representative=0
                dist=1e10
                for k in range(C):
                    d=np.linalg.norm(centroids[k,:]-x[i,:])
                    if d<dist:
                        dist=d 
                        representative=k
                distribution[representative]+=1
    return distribution/np.sum(distribution)


def k_mean_decoding(centroids,x):
    #input x is a sequence of a k-vocabualry 
    #output is the sequence in the 16-dim latent space
    L=np.asarray(x).shape[0]
    X=np.zeros((L,centroids.shape[1]))
    for l in range(L):
        X[l,:]=centroids[int(x[l]),:]
    return X

def entropy(distribution):
    #calculates the entropy of a given discrete distribution
    C=len(distribution)
    entropy=0
    for i in range(C):
        if distribution[i]>0:
            entropy-=distribution[i]*np.log2(distribution[i])
    return entropy

def read_data(directory):
    print('number of files in folder: '+str(np.size(os.listdir(directory))))
    print('de-zeropadd...')
    X=[] #we fill the data in a list
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            x=np.load(os.path.join(directory, filename))
            x=deZeropadd(x)
            x=x.tolist()
            X.append(x)
    return X

def log_prob_sequences(model,sequences):
    #sequences is a list of sequences
    L=len(sequences)
    log_prob=0
    for k in range(L):
        log_prob+=model.log_probability(sequences[k])
    return log_prob/L

def store_model(modelname):
    print('Save model...')
    with open(modelname, 'wb') as output:
        pickle.dump(model, output, -1)    

def load_model(modelname):
    print('load model...')
    with open(modelname, 'rb') as input:
        return pickle.load(input)


def random_crop(im,width=8):
	ceil = im.shape[1]-width
	ind = np.random.randint(ceil)
	return im[:,ind:ind+width]


def segment_image(im,width=8):
    segments = [im[:,i*width:(i+1)*width] for i in range(im.shape[1]//width)]
    return segments

def segment_image_withoverlap(im,width=8,nonoverlap=4):
    segments = []
    idx = 0
    nbins = int((im.shape[1]-width)//nonoverlap)
    for n in range(nbins):
        segments.append(im[:,idx:idx+width])
        idx += nonoverlap
    return segments


def to_batches(segments,batch_size):
    n_batches = int(np.ceil(len(segments)/batch_size))
    batches = [np.zeros(shape=(batch_size,)+tuple(segments[0].shape)) for i in range(n_batches)]
    for i in range(len(segments)):
        batch_idx = i//batch_size
        idx = i%batch_size
        batches[batch_idx][idx] = segments[i]
    return np.array(batches), len(segments)


def get_random_sample(directory):
    try:
        files = os.listdir(join(directory,'images'))
        randint = np.random.randint(len(files))
        return join(directory,'images',files[randint])
    except:
        dirs = [i for i in os.listdir(directory) if not len(os.listdir(join(directory,i)))==0]
        rand_dir = dirs[np.random.randint(len(dirs))]
        files = os.listdir(join(directory,rand_dir))
        return join(directory,rand_dir,files[np.random.randint(len(files))])

def downsample(x,down_factor):
    n = x.shape[0]
    y = np.floor(np.log2(n))
    nextpow2 = int(np.power(2, y + 1))
    x = np.concatenate((np.zeros((nextpow2-n), dtype=x.dtype), x))
    x = resample(x,len(x)//down_factor)
    return x[(nextpow2-n)//down_factor:]

def play_clip(data, fs=44100):
    sd.play(data, fs)

def play_file(file_path):
    data,fs = sf.read(file_path)
    print("playing from file: ",file_path)
    sd.play(data, fs)

def load_from_folder(base_path,folder_path):
    files = os.listdir(join(base_path,folder_path,'songs'))
    files = [i for i in files if '.wav' in i.lower()]
    data = [sf.read(join(base_path,folder_path,'songs',i)) for i in files]
    rates = [i[1] for i in data]
    data = [i[0] for i in data]
    if len(data)==0:
        return None,None
    if len(set(rates))>1:
        print("Sample rates are not the same")
        print("Sample rates are: ")
        print(rates)
    else:
        print("Sample rate is : ",rates[0])
    return data,rates[0]

def play(f):
    import pyaudio
    CHUNK = 1024
    print('playing file: ' + f.split('\\')[-1])
    wf = wave.open(f, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(CHUNK)
    try:
        while data != '':
            stream.write(data)
            data = wf.readframes(CHUNK)
    except KeyboardInterrupt:
        pass
    stream.stop_stream()
    stream.close()
    p.terminate()


def normalize_image(image):
    return image / np.std(image)

def update_progress(progress):
    print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50),progress * 100),end='')


def phase_restore(mag, random_phases,n_fft, N=50):
    p = np.exp(1j * (random_phases))

    for i in range(N):
        _, p = librosa.magphase(librosa.stft(
            librosa.istft(mag * p), n_fft=n_fft))
    #    update_progress(float(i) / N)
    return p


def to_image(seq,nfft):
    nfft_padlen = int(len(seq) + nfft / 2)
    stft = lc.stft(fix_length(seq, nfft_padlen), n_fft=nfft)
    return np.array([np.abs(stft), np.angle(stft)]).transpose(1, 2, 0)

def from_polar(image):
    return image[:, :, 0]*np.cos(image[:, :, 1]) + 1j*image[:,:,0]*np.sin(image[:,:,1])


def from_image(image,clip_len=None):
    if clip_len:
        return fix_length(lc.istft(from_polar(image)), clip_len)
    else:
        return  lc.istft(from_polar(image))

def save_image(image,save_path,save_idx,amplitude_only=False):
    if amplitude_only:
        np.save(join(save_path, str(save_idx) + '.npy'), image[:,:,0])
    else:
        np.save(join(save_path, str(save_idx) + '.npy'), image)

def get_spectrogram(data,log_scale=False,show=False,polar_form_input=False):
    if polar_form_input:
        image = from_polar(data)
    elif len(data.shape)==2:
        image=data
    else:
        image = lc.stft(data)
    D = librosa.amplitude_to_db(image, ref=np.max)
    if show:
        plt.figure()
        if log_scale:
            librosa.display.specshow(D, y_axis='log')
        else:
            librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Frequency power spectrogram')
        plt.show()
    return D


def save_spectrogram(filename,D):
    if D.min() < 0:
        D = D-D.min()
    D = D/D.max()
    I = Image.fromarray(np.uint8(D*255))
    I.save(filename)


def save_audio_sample(sample,path,samplerate):
    sf.write(path,sample,samplerate=int(samplerate))

def longest_sequence(X):
    #X is a list of sequences
    #returns the longest sequence in X
    L=len(X)
    x=np.asarray(X[0])
    for i in range(L-1):
        y=np.asarray(X[i+1])
        if x.shape[0]<y.shape[0]:
            x=y

    return list(x)

def discrete_viterbi_sample(X,model):
    #X is a sequence
    #calculate most likely path given X under the model
    #return a sample from that path
    viterbi=model.viterbi(X)[1]#contains information for all states
    L=len(viterbi)-1#all states minus the start-state which is not defined in our case
    sequence=[]
    #take a sample from each state-emission
    for i in range(L):
        params_i=viterbi[i+1][1].distribution.parameters
        sequence.append(DiscreteDistribution(params_i[0]).sample(1)[0])
    return sequence    

def viterbi_sample(X,model):
    #X is a sequence
    #calculate most likely path given X under the model
    #return a sample from that path
    viterbi=model.viterbi(X)[1]#contains information for all states
    L=len(viterbi)-1#all states minus the start-state which is not defined in our case
    sequence=[]
    #take a sample from each state-emission
    for i in range(L):
        params_i=viterbi[i+1][1].distribution.parameters
        mean_i=params_i[0]
        sigma_i=params_i[1]
        sequence.append(MultivariateGaussianDistribution(mean_i,sigma_i).sample(1)[0])
    return sequence

def normalize_spectrogram(image,threshold):
    if threshold>1.0:
        image = image/threshold
    image = np.minimum(image,np.ones(shape=image.shape))
    image = np.maximum(image,-np.ones(shape=image.shape))
    return image


def generate_dataset(base_path,out_path,nfft,max_num=None,amplitude_only=False,downsample_factor=None):
    save_idx=0
    folders = os.listdir(base_path)
    out_path = join(out_path,'images')
    os.makedirs(out_path,exist_ok=True)
    for folder in folders:
        songs, fs = load_from_folder(base_path, folder)
        if songs:
            if downsample_factor:
                songs = [downsample(i, downsample_factor) for i in songs]
            ims = [to_image(i,nfft) for i in songs]
            for im in ims:
                save_image(im,out_path,save_idx,amplitude_only=amplitude_only)
                save_idx+=1
                if max_num:
                    if save_idx>=max_num:
                        return

def concat(X):
    L=len(X)
    Y=X[0]
    for k in range(L-1):
        Y=np.concatenate((Y,X[k+1]), axis=0)
    return Y

def init_with_GMM(n_mixtures,X):
    max_iterations=400
    X=concat(X)
    print('training GMM...')
    GMM = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=n_mixtures, X=X,verbose=True, max_iterations=max_iterations)
    #s=[]
    dist=[]
    for k in range(n_mixtures):
        mean=GMM.distributions[k].parameters[0]
        cov=GMM.distributions[k].parameters[1]
        dist.append(MultivariateGaussianDistribution(mean,cov))

    trans_mat=np.ones((n_mixtures,n_mixtures))/n_mixtures
    starts=np.ones(n_mixtures)/n_mixtures
    hmm=HiddenMarkovModel.from_matrix(trans_mat,dist,starts)
    return hmm

def inverse_transform(im,N=50):
    random_phase = im.copy()
    np.random.shuffle(random_phase)
    p = phase_restore((np.exp(im) - 1), random_phase, 256, N=N)
    return (np.exp(im) - 1) * p

def transform(im):
    im = from_polar(im)
    im,phase = lc.magphase(im)
    im = np.log1p(im)
    return im

def generate_dataset_by_day(base_path,out_path,nfft,amplitude_only=False,downsample_factor=None,train_val_split=False,name=''):
    save_idx=0
    folders = os.listdir(base_path)
    days = []
    for f in folders:
        try:
            day = dt.strptime('-'.join(f.split('-')[:3]), '%Y-%m-%d')
            days.append(day)
        except:
            continue
    days.sort()
    for folder in folders:
        try:
            day = dt.strptime('-'.join(folder.split('-')[:3]), '%Y-%m-%d')
            num = (day-days[0]).days
            out_path_full = join(out_path,"day%d_%s"%(num,name),"images")
        except:
            out_path_full = join(out_path,folder+'_'+name,"images")

        os.makedirs(out_path_full, exist_ok=True)
        songs, fs = load_from_folder(base_path, folder)
        if not songs:
            continue
        if downsample_factor:
            songs = [downsample(i, downsample_factor) for i in songs]
        ims = [to_image(i,nfft) for i in songs]
        if train_val_split:
            out_path_test = out_path_full.replace(out_path,out_path[:-1]+'_test/')
            out_path_val = out_path_full.replace(out_path,out_path[:-1]+'_val/')
            os.makedirs(out_path_test, exist_ok=True)
            os.makedirs(out_path_val, exist_ok=True)
            im_train,im_test = train_test_split(ims,test_size=0.1)
            im_val,im_test = train_test_split(im_test,test_size=0.5)
            for im in im_train:
                save_image(im, out_path_full, save_idx, amplitude_only=amplitude_only)
                save_idx+=1
            for im in im_val:
                save_image(im, out_path_val, save_idx, amplitude_only=amplitude_only)
                save_idx+=1
            for im in im_test:
                save_image(im, out_path_test, save_idx, amplitude_only=amplitude_only)
                save_idx+=1
        else:
            for im in ims:
                save_image(im,out_path_full,save_idx,amplitude_only=amplitude_only)
                save_idx+=1


def encode_sample(path,sample,epoch=None):
    with open(join(path, 'opt.pkl'), 'rb') as f:
        opt = pickle.load(f)
    ngpu = opt.ngpu
    nz = opt.nz
    ngf = opt.ngf
    nc = opt.nc
    input = torch.FloatTensor(opt.batchSize, nc, opt.imageH, opt.imageW)
    input = Variable(input)

    if epoch is not None:
        net_E_file = join(path,'netE_epoch_%d.pth'%(epoch))
    else:
        E_files = [i for i in os.listdir(path) if 'netE' in i]
        net_E_file = join(path,'netE_epoch_%d.pth'%(len(E_files)-1))
    try:
        from networks_1d import _netG, _netE, _netD, weights_init, GANLoss
        netE = _netE(ngpu, nz, ngf, nc)
        netE.apply(weights_init)
        netE.load_state_dict(torch.load(net_E_file))
        opt.imageH=opt.nc
    except:
        from networks_audio_nophase import _netG, _netE, _netD, weights_init, GANLoss
        netE = _netE(ngpu, nz, ngf, nc)
        netE.apply(weights_init)
        netE.load_state_dict(torch.load(net_E_file))

    netE.cuda()
    encoded = []
    sample_segments = segment_image(sample, width=opt.imageW)
    sample_segments = [transform(k) for k in sample_segments]
    sample_batches, num_segments = to_batches(sample_segments, opt.batchSize)
    cnt = 0
    sequence=[]
    for j in range(len(sample_batches)):
        input.data.copy_(torch.from_numpy(sample_batches[j]))
        encoding = netE(input)
        for k in range(opt.batchSize):
            if cnt >= num_segments:
                sequence.append(np.zeros(shape=sequence[-1].shape))
            else:
                sequence.append(encoding.data[k].cpu().numpy())
                cnt += 1
    return np.array([i for i in sequence if not np.sum(np.abs(i))==0])

def reconstruct_sample(path,sample,epoch=None):
    with open(join(path, 'opt.pkl'), 'rb') as f:
        opt = pickle.load(f)
    ngpu = opt.ngpu
    nz = opt.nz
    ngf = opt.ngf
    nc = opt.nc

    if epoch is not None:
        net_G_file = join(path,'netG_epoch_%d.pth'%(epoch))
    else:
        G_files = [i for i in os.listdir(path) if 'netG' in i]
        net_G_file = join(path,'netG_epoch_%d.pth'%(len(G_files)-1))
        if not os.path.isfile(net_G_file):
            net_G_file = G_files[-1]

    try:
        from networks_1d import _netG, _netE, _netD, weights_init, GANLoss
        netG = _netG(ngpu, nz, ngf, nc)
        netG.apply(weights_init)
        netG.load_state_dict(torch.load(net_G_file))
        opt.imageH=opt.nc
    except:
        from networks_audio_nophase import _netG, _netE, _netD, weights_init, GANLoss
        netG = _netG(ngpu, nz, ngf, nc)
        netG.apply(weights_init)
        netG.load_state_dict(torch.load(net_G_file))

    netG.cuda()
    netG.mode(reconstruction=True)
    reconstructed_samples = []
    cnt=0
    if len(sample)%opt.batchSize==0:
        r=len(sample)//opt.batchSize
    else:
        r=len(sample)//opt.batchSize + 1
    for j in range(r):
        encoding = Variable(torch.from_numpy(sample[j*opt.batchSize:(j+1)*opt.batchSize].astype(np.float32))).cuda()
        reconstruction = netG(encoding)
        for k in range(reconstruction.data.cpu().numpy().shape[0]):
            if cnt>=len(sample):
                break
            reconstructed_samples.append(
                inverse_transform(reconstruction.data[k, :, :, :].cpu().numpy().reshape([opt.imageH,opt.imageW]),N=500))
            cnt+=1
    reconstructed_samples = np.concatenate(reconstructed_samples, axis=1)
    reconstructed_audio = lc.istft(reconstructed_samples)
    return reconstructed_audio


def encode_and_decode(sample, netE, netG, batch_size=64, method=1, \
                      imageH=129, imageW=8, transform_sample=True, return_tensor=False):
    
    sample_segments = segment_image(sample,width=imageW)
    if transform_sample:
        sample_segments = [transform(k) for k in sample_segments]
    sample_batches, num_segments = to_batches(sample_segments, batch_size)

    reconstructed_samples = []    
    z = []
    input = torch.FloatTensor(batch_size, 1, imageH, imageW)
    input=Variable(input)
    out_shape = [imageH, imageW]
    cnt = 0
    for j in range(len(sample_batches)):
        input.data.copy_(torch.from_numpy(sample_batches[j].reshape(input.size())))
        #pdb.set_trace()
        zhat = netE(input)
        if return_tensor:
            z.append(zhat)
        else:
            z.append(zhat.data.cpu().numpy().reshape(batch_size, zhat.size(1)))
        
        reconstruction = netG(zhat)
        if return_tensor:
            reconstructed_samples.append(reconstruction)
        else:
            for k in range(reconstruction.data.cpu().numpy().shape[0]):
                if cnt<num_segments:
                    if method==1:
                        reconstructed_samples.append(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape))
                    elif method==2:
                        reconstructed_samples.append( \
                                                    inverse_transform(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape)))
                    elif method==3:
                        reconstructed_samples.append(get_spectrogram(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape)))
                    cnt+=1
    
    if return_tensor:
        z = torch.cat(z, dim=0)
        reconstructed_samples = torch.cat(reconstructed_samples, dim = 1)
    else:
        reconstructed_samples = np.concatenate(reconstructed_samples,axis=1)
        z = np.concatenate(z, axis = 0)
    return z, reconstructed_samples


def renormalize_spectrogram(s):
    s = np.exp(s) - 1
    if np.min(s) < 0:
        s = s - np.min(s)
    s = s/np.max(s)
    return 10*np.log10(s + 0.01)

def rescale_spectrogram(s):
    if np.min(s) < 0:
        s = s - np.min(s) 
    s = s/np.max(s)
    return np.log1p(s)


def load_spec_no_concatenate(path):
    spec = []
    for f in os.listdir(path):
        pth2fil = os.path.join(path, f)
        if pth2fil.endswith('.npy'):
            s = np.load(pth2fil)
            spec.append(s)
    return spec


def load_netG(netG_file_path, ngpu = 1, nz = 16, ngf = 128, nc = 1, cuda = False):
    netG = _netG(ngpu, nz, ngf, nc)
    netG.apply(weights_init)
    loaded_with_torch=torch.load(netG_file_path,map_location='cpu')
    netG.load_state_dict(loaded_with_torch)

    if cuda:
        netG.cuda()
    netG.mode(reconstruction=True)
    return netG


def load_netE(netE_file_path, ngpu = 1, nz = 16, ngf = 128, nc = 1, cuda = False):
    netE = _netE(ngpu, nz, ngf, nc)
    netE.apply(weights_init)
    netE.load_state_dict(torch.load(netE_file_path,map_location='cpu'))

    if cuda:
        netE.cuda()
    return netE

def decode(zhat, netG, method=1, give_audio = False, imageH=129, imageW=16):
    '''
    Input zhat should have correct number of steps, i.e. if the spectrogram to be decoded has 
    X time frames, then you should input the vector z[: round(X / imageW), :]. This is due to 
    batch_size effects in pytorch!
    '''
    if type(zhat)==numpy.ndarray:
        zhat = torch.from_numpy(zhat)
        zhat = zhat.resize_(zhat.shape[0], zhat.shape[1], 1, 1)
    out_shape = [imageH, imageW]
    reconstructed_samples = []
    reconstruction = netG(zhat)
     
    for k in range(reconstruction.data.cpu().numpy().shape[0]):
        if method==1:
            reconstructed_samples.append(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape))
        elif method==2:
            reconstructed_samples.append( \
                                        inverse_transform(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape), N=500))
        elif method==3:
            reconstructed_samples.append(get_spectrogram(rescale_spectrogram(reconstruction.data[k,:,:,:].cpu().numpy().reshape(out_shape))))
        
    reconstructed_samples = np.concatenate(reconstructed_samples, axis=1)
    if give_audio and method==2:
        reconstructed_audio = lc.istft(reconstructed_samples)
    else:
        reconstructed_audio = []
    return rescale_spectrogram(reconstructed_samples), reconstructed_audio 


