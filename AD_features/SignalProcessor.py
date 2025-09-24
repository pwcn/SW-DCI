# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:35:43 2021

@author: wdl
"""

import torch as t
import os
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from neurodsp.utils import create_times
from neurodsp.plts.time_series import plot_time_series
from scipy.fftpack import fft,ifft
import math
import scipy
import samplerate

## 滑动平均滤波
def np_move_avg(a,n,mode="same"):
	return(np.convolve(a, np.ones((n,))/n, mode=mode))

def get_rms(records):
	return math.sqrt(sum([x ** 2 for x in records]) / len(records))

# 带通滤波
def ideal_passing(input,fl,fh,samplingRate):
    input=np.array(input)

    if len(input.shape)==2:
        input=input.reshape(input.shape[1],)
    n = input.shape[0]
    Freq=np.arange(1,n+1)
    Freq = (Freq - 1) / n * samplingRate
    mask = np.where((Freq<=fl) | (Freq>=fh))
    F=fft(input)
    F[mask]=0
    return ifft(F).real

# 去基线,data为二维原始数据(np.array)，fr为采样率，数据默认按行排列
# 返回值为np.ndarry类型
def detrendPulse(data,fr):
    if(isinstance(data,list)):
        data=np.array(data)
        data = data.reshape((data.shape[0], 1))
    elif(isinstance(data,np.ndarray)):
        # 如果是行向量转为列向量
        if(len(data.shape)==1):
            data = data.reshape((data.shape[0], 1))
        else:
            data = np.transpose(data)
    else:
        print("error")

    N = data.shape[0]

    lamda = 4 * np.floor(fr)

    I = np.eye(N)
    D2 = scipy.sparse.spdiags((np.ones((N - 2, 1)) * [1, -2, 1]).transpose(), [0, 1, 2], N - 2, N)
    a = np.zeros((N - 2, N), dtype=np.int)
    a[N - 4, N - 2] = 1
    a[N - 3, N - 2] = -2
    a[N - 3, N - 1] = 1
    D2 = D2 + a

    weight_matrix = np.linalg.inv(I + math.pow(lamda, 2) * D2.transpose() * D2)

    trend = weight_matrix * data

    detrend = data - trend
    detrend = np.transpose(detrend)
    return detrend, trend

# 求单周期脉冲特定height处的脉宽
def fwhm(x,y,height):
	y=y/np.max(y)
	N=y.shape[0]
	lev=height
	if(y[0]<lev):
		garbage=np.max(y)
		centerindex=np.argmax(y)
		pol=1
	else:
		garbage=np.min(y)
		centerindex=np.argmin(y)
		pol=-1
	
	if(pol==1):
		i=1
		while(np.sign(y[i]-lev)==np.sign(y[i-1]-lev)):
			i=i+1
		interp = (lev-y[i-1]) / (y[i]-y[i-1])
		tlead = x[i-1] + interp*(x[i]-x[i-1])
		i = centerindex+1
		while(i<N-1 and np.sign(y[i]-lev)==np.sign(y[i-1]-lev)):
			i=i+1
		if(i<N):
			interp = (lev-y[i-1]) / (y[i]-y[i-1])
			ttrail = x[i-1] + interp*(x[i]-x[i-1])
			width = ttrail-tlead
		else:
			ttrail=-1
			width=-1
	else:
		tlead=-1
		ttrail=-1
		width=-1
	return [width,tlead,ttrail]

## 数据加载及预处理
def data_loader(path, fr):

	fs = 200.0
	files=os.listdir(path)
	raw=[]
	arr=[]
	SBP=[]
	DBP=[]
	print('preprocessing ...')
	for file in files:
		position=path+file
		filename=file.split(".")[0]
		suffix = file.split(".")[1]
		if(suffix=="txt"):
# 			print(position)
			BP_bins=filename.split("-")
			SBP.append(int(BP_bins[2]))
			DBP.append(int(BP_bins[1]))
			f=open(position,'r+')
			temp=f.read()
			list_temp=temp.split('\n')
			for i in range(3000):
				strN = list_temp[i+1000]
				arr.append(int(strN))
			arr = samplerate.resample(arr, fs/fr, 'sinc_best')
			raw.append(arr)
			arr=[]
	raw = np.array(raw, dtype=float)	
	SBP = np.array(SBP, dtype=float)
	DBP = np.array(DBP, dtype=float)
	
	
	## 脉搏数据预处理及脉搏特征提取
	PPG = np.zeros([raw.shape[0],raw.shape[1]], dtype=float)
	for i in range(raw.shape[0]):
# 		detrended = nk.signal_detrend(raw[i])
		detrended, trend = detrendPulse(raw[i], fs)
		detrended=detrended.reshape(-1,1)
		detrended = t.Tensor(detrended)
		detrended2 = t.squeeze(detrended)
		detrended2=np.array(detrended2)
		PPG[i] = np_move_avg(detrended2, int(fs/10))
		times = create_times(len(detrended)/fs, fs)
		_, axs = plt.subplots(3, 1, figsize=(15, 6))
		if(i==22):
			plot_time_series(times, raw[i], 'Raw PPG', xlim=[0, 10], xlabel=None, ax=axs[0,0])
			plot_time_series(times, detrended2, 'Detrended PPG', xlim=[0, 10], xlabel=None, ax=axs[1,0])
			plot_time_series(times, PPG[i], 'Smoothed PPG', xlim=[0, 10], xlabel=None, ax=axs[2,0])
			plt.legend(loc='upper right')

	print('preprocessing done')
	return [raw, PPG, SBP, DBP]


























