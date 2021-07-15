# circles.py
import numpy as np
import pylab as plt 
from progress.bar import IncrementalBar #Bar

from math import ceil

import time
from uti import parse_file
import uti
import plot_func as pf
import analysis as a

IndexVisualize=100000

def simulate(R,fnameOUT='out.npy'):
    '''
        R dictionary with parameters

        VERY IMPORTANT !!! 
            Since R is updated along the script remember to call simulate each time with the proper/new R dictionary.

    '''
    t0=time.time()

    R['n']=int(R['n'])
    

    R['nsteps']=int(R['totaltime']/R['tstep'])
    R['k']=R['k1']
    R['stepToSaveChanges']=R['timeToSaveChanges']/R['tstep']
    R['w']=R['w0']*np.ones((R['n'],R['n']))
    R['endingTimeStep']=0

    if not(('x' in R)|('y' in R)): # (x,y) can be given (e.g. regular grid)), otherwise generate random cells on a unit square field 
        R['x']=np.random.rand(R['n'])
        R['y']=np.random.rand(R['n'])

    '''
    R.expName = nameDir;
    if R.implementFJittering
        R.f0=R.f0 + (R.f0/R.jitteredSigmaConst)*randn(1,R.n);
    end

    % parameters controlling the stop phase
    R.intervalForStopping=ceil(R.timeToStop/R.tstep); % interval after which to stop in # of steps
    R.stepsToBeStopped=round(R.timeToBeStopped/R.tstep); %number of steps to be resting

    #R.totalTimeOfCurrentPhase=0; # what is current phase?
    #R.startingTimeStep=0; # used? 
    #R.endingTimeStep=0;
    R.isCurrentPhaseStop='false';
    R.currentSavedPhase=0;
  
    '''

    # declare structures to save data 
    '''
    %
    if ~R.implementStop
    dimHistory=floor(R.nsteps/R.stepToSaveChanges);
    else
    if R.intervalForStopping>0
        dimHistory=floor(R.intervalForStopping/R.stepToSaveChanges);
    else
        dimHistory=0;
    end
    end

    %variables to take into consideration the contribution of excitatory and inhibitory cells
    R.AllExCont=zeros(dimHistory,R.n);
    R.AllInCont=zeros(dimHistory,R.n);
    %
    R.AllCHistory=zeros(dimHistory,R.n);
    R.AllRHistory=zeros(dimHistory,R.n);

    R.allspikes=[]; %vector in which all the spikes will be stored, first row: neuron number, second row: spike timestamp
    %in case the neuron is inhibitory the sign will be changed

    if (isequal(R.pinh,0) && isequal(R.implementInibition,1)) || (~isequal(R.pinh,0) && isequal(R.implementInibition,0))%%
    display(['exp in ' nameDir ' had mismatching implementInhibition flag and percentage inhibitory cells']);
    return
    elseif (~isequal(R.pinh,0) && isequal(R.implementInibition,1))


    '''    

    #idx_max=ceil(1.35*(R['catarget']/R['tauca'])*R['n']*R['totaltime'])
    idx_max=ceil(50*(R['catarget']/R['tauca'])*R['n']*R['totaltime'])
    allspikes=np.zeros((2,idx_max))
    #zeros(2,ceil(1.35*(R.catarget/R.tauca)*R.n*R.totaltime)); % 35% over the expected number of spikes

    AllCHistory=[]
    AllRHistory=[]
    AllfHistory=[]

    # ...inhibition ..
    R['ni']=int(np.ceil(R['n']*R['pinh']))

    if 'nType' in R:
        idx_in=np.where(R['nType']==-1)[0] 
        R['w'][idx_in,:]*=-1
        if R['ni']!=len(idx_in):
            print('Error in the given inhibitory nType ')
            return -1 
    else:
        idx_in=np.random.choice(np.arange(R['n']),R['ni'],replace=False)
        R['w'][idx_in,:]*=-1
        R['nType']=np.sign(R['w'][:,0])

    # f0 
    f0VAL=R['f0']
    R['f0']=np.ones(R['n'])
    idxInh=R['nType']<0
    if R['ni']: R['f0'][idxInh]*=(f0VAL[idxInh]*R['f0factor']) # check if update 
    R['f0'][R['nType']>0]*=f0VAL[R['nType']>0] 

    # tauf 
    taufVAL=R['tauf']
    R['tauf']=np.ones(R['n'])
    if R['ni']: R['tauf'][idxInh]*=(taufVAL[idxInh]*R['tauffactor']) # check if update
    R['tauf'][R['nType']>0]*=taufVAL[R['nType']>0]

    '''
    % TN
    f0VAL=R.f0;
    R.f0=ones(1,R.n);
    R.f0(R.nType<0)=R.f0(R.nType<0)*f0VAL*R.f0factor; % TN
    R.f0(R.nType>0)=R.f0(R.nType>0)*f0VAL;

    taufVAL=R.tauf;
    R.tauf=ones(1,R.n);
    R.tauf(R.nType<0)=R.tauf(R.nType<0)*taufVAL*R.tauffactor; % TN
    R.tauf(R.nType>0)=R.tauf(R.nType>0)*taufVAL;
    '''


    '''
    tcheckconnectivity=50000;
    changedconnectivity=0;
    justchangedconnectivity=0;
    % % % % % % % % % % % Parameters End % % % % % % % % % % %
    '''

    # initialization 

    ilastspike=np.ones(R['n'])*(-1e12) # long time ago .. 
    
    d=uti.distance(R['x'],R['y'])
    #R['d']=d # useful ?
    
    r=np.clip(np.random.rand(R['n'])*R['maxinitialr'],R['minr'],None)  # choose the radius for each channel
    f=R['f0']*np.ones(R['n']) # set the initial frequencies for all the neurons (f0)
    ca=np.zeros(R['n']) # set to zero the initial Calcium concentration (0)

    '''
    #R.startingTimeStep=1;
    #actualSavedRow=0;
    '''

    numspikes=0 # actual number of spikes recorded

    #bar=IncrementalBar('Simulation',max=ceil(R['nsteps']/IndexVisualize))

    for i in range(R['nsteps']):
        if i%IndexVisualize==0: 
            print('steps in current developing phase: %d out of %d'%(i,R['nsteps'])) 
            #bar.next()

        # establish who fires  
        fired=np.where( (np.random.rand(R['n'])<(f*R['tstep'])) & (ilastspike<(i-R['trefractory']/R['tstep'])) )[0] # look at which neurons have fired
        Lfired=len(fired)

        # update firing 
        f=f+R['tstep']*(R['f0']-f)/R['tauf']  # update the firing rate

        # update calcium
        ca=ca*(1-R['tstep']/R['tauca']) # % update the Calcium concentration
        
        # update radius 
        idxP=R['nType']>0
        r[idxP]=np.clip(r[idxP]+R['k']*R['tstep']*(R['catarget']-ca[idxP]),R['minr'],a_max=None)
        idxN=R['nType']<0
        r[idxN]=np.clip(r[idxN]+R['k']*R['tstep']*(R['catarget']-ca[idxN]),R['minr'],a_max=None) # check different multiplication factor for inh

        if i%R['timeToSaveChanges']==0:
            # store sim data 
            #print("mean r=",np.mean(r))
            #if len(AllRHistory): print((r-AllRHistory[-1]).max())
            AllCHistory.append(ca.copy())
            AllRHistory.append(r.copy())
            AllfHistory.append(f.copy())

        if Lfired:
            ilastspike[fired]=i
            allspikes[:,range(numspikes,(numspikes+Lfired))]=[fired,i*R['tstep']*np.ones(Lfired)]
            #R.allspikes(:,numspikes+1:numspikes+length(fired))=[fired;i*R.tstep*ones(1,length(fired))]; % update the vector with the spikes fired
            numspikes+=Lfired
            #
            exCont=inCont=inContAbs=0
            firedEx=fired[R['nType'][fired]==1]     # CRC ok
            firedIn=fired[R['nType'][fired]==-1]    # CRC ok
            if len(firedEx):
                r1=np.kron(np.ones((R['n'],1)),r[firedEx])
                r2=np.transpose(np.kron(np.ones((len(firedEx),1)),r))
                d0=d[:,firedEx]
                areasEx=uti.overlap(r1,r2,d0) 
                exCont=R['gex']*np.sum(np.transpose(areasEx)*R['w'][firedEx,:],axis=0) # R.gex*sum(areasEx'.*R.w(firedEx,:),1);
            if len(firedIn):   
                r1=np.kron(np.ones((R['n'],1)),r[firedIn])
                r2=np.transpose(np.kron(np.ones((len(firedIn),1)),r))
                d0=d[:,firedIn]
                areasIn=uti.overlap(r1,r2,d0) 
                inCont=R['gin']*np.sum(np.transpose(areasIn)*R['w'][firedIn,:],axis=0) # R.gin*sum(areasIn'.*R.w(firedIn,:),1);
                # gabashift
                if i<R['nstepswitch']: inCont*=-1 # change sign 
                inContAbs=-inCont # np.abs(inCont)    # in line with circles_gap.m now  

            if R['implementInibition']:
                if R['inhibitionMethod']=='normInFixedParam':
                    f=f+exCont-f*(inContAbs/(inContAbs+R['normInFixedParamValue']));

                if R['inhibitionMethod']=='normInNeuronArea':
                    areasNeurons=r**2*np.pi
                    areasNeurons[areasNeurons==0]=1e-12
                    f=f+exCont-f*(inContAbs/(R['gin']*areasNeurons))
                if R['inhibitionMethod']=='step':
                    f=np.clip(f+exCont-inContAbs,a_min=0,a_max=None)
            else:
                f=f+exCont

            ca[fired]+=1 # increment the Ca concentration of the cells that fired by one

    # bar.finish()

    idx=np.where(allspikes[1,:]>0)[0]

    deltaT=time.time()-t0

    print("Simulation finished in %g seconds "%deltaT)

    #dout=dict(R=R,ca=np.array(AllCHistory),r=np.array(AllRHistory),f=np.array(AllRHistory),allspikes=allspikes[:,idx])
    dout=dict(R=R,r=np.array(AllRHistory),allspikes=allspikes[:,idx],CompTimeSec=deltaT)
    np.save(fnameOUT,dout)

    # free some space 
    AllCHistory,AllRHistory,AllfHistory,allspikes=[],[],[],[]

    #return AllCHistory,AllRHistory,AllfHistory,spikes


'''
%implement the change of k in case the user chose
if (R.implementKChange) && (~changedconnectivity) && (mod(i,round(tcheckconnectivity/R.tstep))==0)
    if mean(sum(overlap(repmat(r,[R.n,1]),repmat(r',[1,R.n]),R.d)))>0.75*(1-R.f0*R.tauca/R.catarget)/(R.gex*R.tauf)
        changedconnectivity=1;
        justchangedconnectivity=1;
        R.k=R.k2;
    end
end
%in case we stop for the stop phase, save and call the function
if R.implementStop && (mod(i,R.intervalForStopping)==0);
    R.currentSavedPhase=R.currentSavedPhase+1;
    R.phaseName=[(num2str(zeros((3-length(num2str(R.currentSavedPhase))),1)))' num2str(R.currentSavedPhase)];
    stoppedPhase(R, i, f, r, ca);
end
%%%% Drawing phase %%%%
if (mod(i,round(R.tdisplay/R.tstep))==0)% || (~isempty(fired))
    %compute the numberOfSpikes times the number of firing channels
    idxEnd=find(R.allspikes(2,:)==0,1,'first')-1;
    [ilastdrawn, justchangedconnectivity]=plotCircles(R,ilastspike,ilastdrawn,r,i,justchangedconnectivity);
end
'''

'''
R.currentSavedPhase=R.currentSavedPhase+1;
R.phaseName=[(num2str(zeros((3-length(num2str(R.currentSavedPhase))),1)))' num2str(R.currentSavedPhase)];
R.endingTimeStep=i*R.tstep;
saveData(R, r,numspikes);
'''

