# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import neurokit2 as nk
import SignalProcessor as sp
import pandas as pd
import numpy.fft as nf
from scipy import signal
from sampen import sampen2

###### presData.mat is the data after pre-procesing and adaptive signal interception, is the best that.

X_data = loadmat('G:\\动脉硬化\\presData.mat')
figer_data = X_data['PPG']

fs = 200
features = np.zeros([figer_data.shape[0], 57], dtype=float)
for i in range(figer_data.shape[0]):
    
    ## 第i个病人的最优PPG片段
    sig = figer_data[i]
    ## plt.plot(sig)
    
   
    ######## PPG整个信号片段的特征
    
    ##峰值检测，波峰波谷数量一致
    signal1, info1 = nk.ppg_process(sig, sampling_rate=fs)
    # plt.plot(signal1.PPG_Peaks)
    peaks = info1['PPG_Peaks']
    x = -sig
    signal2, info2 = nk.ppg_process(x, sampling_rate=fs)
    valleys = info2['PPG_Peaks']
    
    
    
    HRArray=np.array(signal1['PPG_Rate'].values.tolist(),dtype='float')
    HR = np.mean(HRArray) #心率
    
    #特征定义，常规特征
    PHArray=[]  
    NHArray=[]
    DHArray=[]
    CTArray=[]
    NTArray=[]
    CLArray=[]
    KurtArray=[]
    SkewArray=[]
    
    # 新用特征
    MSHArray=[]
    MPHArray=[]
    MSLArray=[]
    MCTArray=[]
    MCLArray=[]
    APNArray=[]
    RPAArray=[]
    VPoArray=[]
    
    # 之前用过的特征
    PNHArray=[]
    RIArray=[]
    MWSArray=[]
    ACArray=[]
    DCArray=[]
    
    CTRArray=[]
    NTRArray=[]
    NIArray=[]
    PNLArray=[]
    P2ocdArray=[]
   
    PW90Array=[]
    PW75Array=[]
    PW66Array=[]
    PW50Array=[]
    PW33Array=[]
    PW25Array=[]
    PW10Array=[]

    AreaArray=[]
    AosArray=[]
    AsoArray=[]
    RSDArray=[]
    
    ## 需要注意的特征
    PTTArray=[]
    #SIArray=[]
    KArray=[]
    AIArray=[]
    mNPVArray=[]
    
    
    
    ################# 导数特征：
    # 一阶导
    AmsArray=[]
    PDAArray=[]
    SFmArray=[]
    AFmArray=[]
    AFeArray=[]
    VFDArray=[]
    
    # 二阶导
    HaArray=[]
    HbArray=[]
    HcArray=[]
    HdArray=[]
    HeArray=[]
    HbaArray=[]
    HcaArray=[]
    HdaArray=[]
    HeaArray=[]
    AGIArray=[]


    #### 单周期 PPG 特征提取
    
    for j in range(valleys.shape[0]):
        if(j>=1):
            pulse_len = valleys[j]-valleys[j-1]
            
            if pulse_len<100:
                continue
            
            seg=sig[valleys[j-1]:valleys[j]]   # 单周期脉搏        
            seg_normalized = (seg-np.min(seg))/(np.max(seg)-np.min(seg))  # 标准化的单周期脉搏
            
            V_seg = np.diff(seg)     # FD-PPG

            A_seg = 100*np.diff(V_seg) # # SD-PPG,由于数值较小，数值均扩大100倍
            
            ## 四个关键点位置确定
            max_loc = np.argmax(seg)         #峰值坐标
            if seg[max_loc]<1:
                continue
            
            ms_loc = np.argmax(V_seg) 
            
            V_peaks = signal.find_peaks(V_seg,height=-1)[0]
            
            if len(V_peaks) ==1:
                Dia_loc = 83
                continue
            else:
                Dia_loc = V_peaks[1]
                
            A_peaks = signal.find_peaks(A_seg,height=-1)[0]
            
            
            if len(A_peaks) ==1:
                Notch_loc = 68
            else:
                Notch_loc = A_peaks[1]
                
            if Notch_loc == max_loc:
                continue
            
            # 常规特征，PH, NH, DH, CT, NT, CL, Kurt and Skew
            PHArray.append(seg[max_loc]-seg[0])
            NHArray.append(seg[Notch_loc]-seg[0])
            DHArray.append(seg[Dia_loc]-seg[0])
            MSHArray.append(seg[ms_loc]-seg[0])
            PNHArray.append((seg[Notch_loc]-seg[0])/(seg[max_loc]-seg[0]))
            RIArray.append((seg[Dia_loc]-seg[0])/(seg[max_loc]-seg[0]))
            MPHArray.append((seg[ms_loc]-seg[0])/(seg[max_loc]-seg[0]))
            
            CTArray.append(max_loc/200)
            NTArray.append(Notch_loc/200)
            MSLArray.append(ms_loc/200)
            CLArray.append(seg.shape[0]/200)
            CTRArray.append((max_loc/200)/(seg.shape[0]/200))
            NTRArray.append((Notch_loc/200)/(seg.shape[0]/200))
            MCTArray.append((ms_loc/200)/(max_loc/200))
            MCLArray.append((ms_loc/200)/(seg.shape[0]/200))
            
            s = pd.Series(seg)
            # 偏度和峰度
            KurtArray.append(s.kurt())
            SkewArray.append(s.skew())
            
            area=np.sum(seg_normalized[0:seg_normalized.shape[0]])
            apn = np.sum(seg_normalized[max_loc:Notch_loc])
            
            # 收缩期面积
            sum1=np.sum(seg_normalized[0:max_loc])
            # 舒张期面积
            sum2=np.sum(seg_normalized[max_loc:seg_normalized.shape[0]])

            AreaArray.append(area)
            APNArray.append(apn)
            RPAArray.append(apn/area)            
            AosArray.append(sum1)
            AsoArray.append(sum2)
            RSDArray.append(sum1/sum2)
            VPoArray.append(np.var(seg_normalized[max_loc:seg_normalized.shape[0]]))
            
            # PW：width 脉搏宽度
            [width,tlead,ttrail]=sp.fwhm(np.linspace(1,seg.shape[0],seg.shape[0]), seg, 0.75)
            if(width != -1):
                PW75Array.append(width)
            [width,tlead,ttrail]=sp.fwhm(np.linspace(1,seg.shape[0],seg.shape[0]), seg, 0.5)
            if(width != -1):
                PW50Array.append(width)

            [width,tlead,ttrail]=sp.fwhm(np.linspace(1,seg.shape[0],seg.shape[0]), seg, 0.33)
            if(width != -1):
                PW33Array.append(width)

            [width,tlead,ttrail]=sp.fwhm(np.linspace(1,seg.shape[0],seg.shape[0]), seg, 0.9)
            if(width != -1):
                PW90Array.append(width)

            [width,tlead,ttrail]=sp.fwhm(np.linspace(1,seg.shape[0],seg.shape[0]), seg, 0.66)
            if(width != -1):
                PW66Array.append(width)

            [width,tlead,ttrail]=sp.fwhm(np.linspace(1,seg.shape[0],seg.shape[0]), seg, 0.25)
            if(width != -1):
                PW25Array.append(width)
                
            [width,tlead,ttrail]=sp.fwhm(np.linspace(1,seg.shape[0],seg.shape[0]), seg, 0.10)
            if(width != -1):
                PW10Array.append(width)
            
            # K值
            KArray.append((np.mean(seg_normalized)-np.min(seg_normalized))/(np.max(seg_normalized)-np.min(seg_normalized)))                                   
            
            # PTT特征 
            
            if len(V_peaks) ==1:                 # 其一阶导仅有一个峰值，PTT赋负值，便于后续的检查
                PTT=-10
            else:                                  # 暂不考虑有明显重搏波的情况，其计算过于复杂
                PTT = (V_peaks[1] - max_loc)/200
                    
            PTTArray.append(PTT)   
            
            MWSArray.append((seg[max_loc]-seg[0])/max_loc)
            PNLArray.append((Notch_loc-max_loc)/200)
            NIArray.append(Notch_loc/(seg.shape[0]-Notch_loc))
            AIArray.append((seg[max_loc]-seg[Dia_loc])/(seg[max_loc]-seg[0]))
            
            ##SI 需结合受试者的身高，由于PTT已知，SI已知
            P2ocdArray.append((seg.shape[0]-max_loc)/200)
            ACArray.append(np.max(seg)-np.min(seg))
            DCArray.append(np.mean(seg_normalized))
            mNPVArray.append((np.max(seg)-np.min(seg))/((np.max(seg)-np.min(seg))+np.mean(seg_normalized)))

            ############## 导数特征
            AmsArray.append(np.max(V_seg))
            PDAArray.append(np.max(V_seg)/(seg[max_loc]-seg[0]))
            SFmArray.append((np.max(V_seg))/(ms_loc/200))
            AFmArray.append(np.sum(V_seg[0:max_loc]))
            AFeArray.append(np.sum(V_seg[max_loc:V_seg.shape[0]]))
            VFDArray.append(np.var(V_seg[max_loc:V_seg.shape[0]])*500)  
            
            
            A_vallys = signal.find_peaks(-A_seg,height=-1)[0]
            
            HaArray.append(A_seg[A_peaks[0]])
            HbArray.append(A_seg[A_vallys[0]])
            HcArray.append(A_seg[A_peaks[1]])
            HdArray.append(A_seg[A_vallys[1]])
            
            if len(A_peaks) ==2: 
                HeArray.append(0)
                HeaArray.append(0)
                AGIArray.append((A_seg[A_vallys[0]]-A_seg[A_peaks[1]]-A_seg[A_vallys[1]])/A_seg[A_peaks[0]])
            else: 
                HeArray.append(A_seg[A_peaks[2]])
                HeaArray.append(A_seg[A_peaks[2]]/A_seg[A_peaks[0]])
                AGIArray.append((A_seg[A_vallys[0]]-A_seg[A_peaks[1]]-A_seg[A_vallys[1]]-A_seg[A_peaks[2]])/A_seg[A_peaks[0]])
            
            HbaArray.append(A_seg[A_vallys[0]]/A_seg[A_peaks[0]])
            HcaArray.append(A_seg[A_peaks[1]]/A_seg[A_peaks[0]])
            HdaArray.append(A_seg[A_vallys[1]]/A_seg[A_peaks[0]])
            
            
            
            
     
    PH=np.mean(np.array(PHArray,dtype=float))
    NH=np.mean(np.array(NHArray,dtype=float))
    DH=np.mean(np.array(DHArray,dtype=float))
    MSH=np.mean(np.array(MSHArray,dtype=float))
    PNH=np.mean(np.array(PNHArray,dtype=float))
    RI=np.mean(np.array(RIArray,dtype=float))
    MPH=np.mean(np.array(MPHArray,dtype=float))
    CT=np.mean(np.array(CTArray,dtype=float))
    NT=np.mean(np.array(NTArray,dtype=float))
    MSL=np.mean(np.array(MSLArray,dtype=float))
    CL=np.mean(np.array(CLArray,dtype=float))
    CTR=np.mean(np.array(CTRArray,dtype=float))
    NTR=np.mean(np.array(NTRArray,dtype=float))
    MCT=np.mean(np.array(MCTArray,dtype=float))
    MCL=np.mean(np.array(MCLArray,dtype=float))
    Kurt=np.mean(np.array(KurtArray,dtype=float))
    Skew=np.mean(np.array(SkewArray,dtype=float))
    Area=np.mean(np.array(AreaArray,dtype=float))
    APN=np.mean(np.array(APNArray,dtype=float))
    RPA=np.mean(np.array(RPAArray,dtype=float))
    Aos=np.mean(np.array(AosArray,dtype=float))
    Aso=np.mean(np.array(AsoArray,dtype=float))
    RSD=np.mean(np.array(RSDArray,dtype=float))
    VPo=np.mean(np.array(AosArray,dtype=float))
        
    PW90=np.mean(np.array(PW90Array,dtype=float))
    PW75=np.mean(np.array(PW75Array,dtype=float))
    PW66=np.mean(np.array(PW66Array,dtype=float))
    PW50=np.mean(np.array(PW50Array,dtype=float))
    PW33=np.mean(np.array(PW33Array,dtype=float))
    PW25=np.mean(np.array(PW25Array,dtype=float))
    PW10=np.mean(np.array(PW10Array,dtype=float)) 

    K=np.mean(np.array(KArray,dtype=float))
    PTT=np.mean(np.array(PTTArray,dtype=float))
    MWS=np.mean(np.array(MWSArray,dtype=float))
    PNL=np.mean(np.array(PNLArray,dtype=float))
    NI=np.mean(np.array(NIArray,dtype=float))
    AI=np.mean(np.array(AIArray,dtype=float))
    #SI=np.mean(np.array(SIArray,dtype=float))
    P2ocd=np.mean(np.array(P2ocdArray,dtype=float))
    AC=np.mean(np.array(ACArray,dtype=float))
    DC=np.mean(np.array(DCArray,dtype=float))
    mNPV=np.mean(np.array(mNPVArray,dtype=float))
    
    Ams=np.mean(np.array(AmsArray,dtype=float))
    PDA=np.mean(np.array(PDAArray,dtype=float))
    SFm=np.mean(np.array(SFmArray,dtype=float))
    AFm=np.mean(np.array(AFmArray,dtype=float))
    AFe=np.mean(np.array(AFeArray,dtype=float))
    VFD=np.mean(np.array(VFDArray,dtype=float))
    Ha=np.mean(np.array(HaArray,dtype=float))
    Hb=np.mean(np.array(HbArray,dtype=float))
    Hc=np.mean(np.array(HcArray,dtype=float))
    Hd=np.mean(np.array(HdArray,dtype=float))
    He=np.mean(np.array(HeArray,dtype=float))
    Hba=np.mean(np.array(HbaArray,dtype=float))
    Hca=np.mean(np.array(HcaArray,dtype=float))
    Hda=np.mean(np.array(HdaArray,dtype=float))
    Hea=np.mean(np.array(HeaArray,dtype=float))
    AGI=np.mean(np.array(AGIArray,dtype=float))
    
    
   
    feature=[round(PH, 3),round(NH, 3),round(DH, 3),round(MSH, 3),round(PNH, 3),round(RI, 3),round(MPH, 3),round(CT, 3),
             round(NT, 3),round(MSL, 3),round(CL, 3),round(CTR, 3),round(NTR, 3),round(MCT, 3),round(MCL, 3),round(Kurt, 3),
             round(Skew, 3),round(Area, 3),round(APN, 3),round(RPA, 3),round(Aos, 3),round(Aso, 3),round(RSD, 3),round(VPo, 3),
             round(PW90, 3), round(PW75, 3), round(PW66, 3), round(PW50, 3), round(PW33, 3), round(PW25, 3), round(PW10, 3),
             round(K, 3),round(PTT, 3),round(MWS, 3),round(PNL, 3),round(NI, 3),round(AI, 3),round(P2ocd, 3),round(AC, 3),round(DC, 3),
             round(mNPV, 3),round(Ams, 3),round(PDA, 3),round(SFm, 3),round(AFm, 3),round(AFe, 3),round(VFD, 3),round(Ha, 3),
             round(Hb, 3),round(Hc, 3),round(Hd, 3),round(He, 3),round(Hba, 3),round(Hca, 3),round(Hda, 3),round(Hea, 3),round(AGI, 3)]
    features[i]=feature
head = ['PH','NH','DH','MSH','PNH','RI','MPH','CT','NT','MSL','CL','CTR','NTR','MCT','MCL','Kurt','Skew','Area',
        'APN','RPA','Aos','Aso','RSD','VPo',
         'PW90', 'PW75', 'PW66', 'PW50', 'PW33', 'PW25', 'PW10','K','PTT','MWS','PNL','NI','AI','P2ocd','AC','DC',
         'mNPV','Ams',
         'PDA','SFm','AFm','AFe','VFD','Ha','Hb','Hc','Hd','He','Hba','Hca','Hda','Hea','AGI']
np.savetxt('Extractrd_Features.csv', features, delimiter = ',') 

# data=pd.read_csv('Extractrd_Features.csv',header=None,names=head)
# data.to_csv('Extractrd_Features.csv',index=False)

        
        
        
        
        
        
        
        
        
        
        
