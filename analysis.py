# analysis.py
import numpy as np

def MeanFiringRate(spikes,Ncells=64,TimeWindow=-1,tlim=[]):
    '''
        spikes
        Ncells: if -1 it is estimated from spikes 
        TimeWindow: if -1 it is estimated from last spike events of spikes (in seconds)
        tlim:   if given, computes mfr on the selected window, otherwise over the entire spikes (in ms)
    '''
    if TimeWindow==-1: TimeWindow=spikes[1,-1]/1000
    if Ncells==-1: Ncells=len(np.unique(spikes[0,:]))
    if len(tlim)==2:
        idx=np.where((spikes[1,:]>tlim[0])&(spikes[1,:]<tlim[1]))[0]
        Nspikes=len(idx)
        TimeWindow=(tlim[1]-tlim[0])/1000 
    else: 
        Nspikes=spikes.shape[1]
    mfr=Nspikes/(TimeWindow*Ncells)
    return mfr 

def MFRtime(spikes,TimeWindow=100000,Ncells=64,Tend=-1):
    '''
    '''
    if Ncells==-1: Ncells=len(np.unique(spikes[0,:])) 
    if Tend==-1: Tend=spikes[1,-1] 
    Tinspect=np.arange(0,Tend,TimeWindow)
    L=len(Tinspect)-1
    mfr_vect=np.zeros(L)
    for k in range(L): 
        mfr_vect[k]=MeanFiringRate(d['allspikes'],Ncells=Ncells,tlim=[Tinspect[k],Tinspect[k+1]])
    if plotIt: plt.plot(Tinspect[:-1],mfr_vect,'ko-')
    return Tinspect[:-1],mfr_vect

def ChannelMeanFiringRate(spikes,Ncells=64,TimeWindow=-1,tlim=[]):
    '''
        spikes
        Ncells: if -1 it is estimated from spikes 
        TimeWindow: if -1 it is estimated from last spike events of spikes (in seconds)
        tlim:   if given, computes mfr on the selected window, otherwise over the entire spikes (in ms)
    '''
    if len(tlim)==2:
        idx=np.where((spikes[1,:]>tlim[0])&(spikes[1,:]<tlim[1]))[0]
        TimeWindow=(tlim[1]-tlim[0])/1000 # ms -> s
        spikesCP=np.copy(spikes[:,idx])
    else:
        spikesCP=np.copy(spikes)
    #
    if TimeWindow==-1: TimeWindow=spikesCP[1,-1]/1000  # ms -> s
    if Ncells==-1: Ncells=len(np.unique(spikesCP[0,:]))
    chan_mfr=np.unique(spikesCP[0,:],return_counts=True)[1]
    return chan_mfr/TimeWindow

def ChannelMFRtime(spikes,TimeWindow=100000,Ncells=64,Tend=-1):
    '''
    '''
    if Ncells==-1: Ncells=len(np.unique(spikes[0,:])) 
    if Tend==-1: Tend=spikes[1,-1] 
    Tinspect=np.arange(0,Tend,TimeWindow)
    L=len(Tinspect)-1
    mfr_vect=np.zeros(L)
    mfr_err_vect=np.zeros(L)
    for k in range(L): 
        mfr=ChannelMeanFiringRate(spikes,Ncells=Ncells,tlim=[Tinspect[k],Tinspect[k+1]])        
        mfr_vect[k]=np.mean(mfr)
        mfr_err_vect[k]=np.std(mfr)
    return Tinspect[:-1],mfr_vect,mfr_err_vect

