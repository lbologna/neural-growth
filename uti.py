import numpy as np
import pylab as plt
import plot_func as pf

def is_number(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

def num2str(num,L=4):
    '''
    '''
    snum='%d'%num 
    while len(snum)<L: snum='0'+snum
    return snum

def parse_file(fn='p.txt'):
    '''
    '''
    d={}
    with open(fn, 'r') as file: 
        for line in file:
            line, _, comment = line.partition('%')
            #print(line)
            if line.strip(): # non-blank line
                key, value = line.split()
                d[key]=float(value) if is_number(value) else value.replace('\'','')
    return d            

def load_param(fn_par):
    '''
    '''
    R=np.load(fn_par,allow_pickle=True).item()
    return R

def overlap(r1,r2,d0):
    '''
    %   r1 is a repetition of the radii of all the cells that fired
    %   during the actual timestep
    %   r2 is a repetition of the radii of all the cells
    %   d0 is the matrix of the distances between the neurons that fired and
    %   all the other neurons
    '''
    # make sure r1>=r2 always
    temp=np.zeros(r1.shape)
    disordered=np.where(r1<r2) # axis? find MAT 
    temp[disordered]=r1[disordered]
    r1[disordered]=r2[disordered]
    r2[disordered]=temp[disordered]
    
    #    
    s=np.zeros(r1.shape) # consider to use sparse structure s=sparse(zeros(size(r1)));
    find1=np.where(d0<=np.abs(r1-r2))
    s[find1]=np.pi*r2[find1]**2 # find overlapping neurons which are entirely contained in others
    others=np.where((d0<(r1+r2)) & (d0>np.abs(r1-r2))) # find overlapping neurons which are not entirely contained in the others
    r=r2[others]/r1[others]
    d=d0[others]/r1[others]
    stemp1=1+d**2-r**2
    stemp = -.5 * np.sqrt(4*d**2-stemp1**2) + np.arccos(stemp1/(2*d)) + r**2 * np.arccos((d**2+r**2-1)/(2*d*r))
    s[others]=stemp*r1[others]**2 
    return s

def distance(x,y):
    ''' 
        alternative implementation 
        from scipy.spatial.distance import pdist,squareform
        D # n rows x 2 columns (x,y)
        dist=squareform(pdist(D))
    '''
    N=len(x)
    xd=np.kron(np.ones((N,1)),x)  
    yd=np.kron(np.ones((N,1)),y)
    d=np.sqrt((xd-np.transpose(xd))**2+(yd-np.transpose(yd))**2)
    np.fill_diagonal(d,1e12)
    return d

def MoveCells(x,y,radius,idxNM=[],BounceStep=0.01,scaleY=0.01): # not used at the moment !
    '''
        x,y centers of the circles 
        radius radius of the circles (same for all)
        idxNM index of the circles not to move 
        BounceStep:   x-shift of the bouncing process
    '''
    def AreaOverlap(d_over):
        '''
        '''
        q=2*np.arccos(d_over/(2*radius))
        return (q-np.sin(q))*radius**2

    # displacement matrices, describe interaction among the N receptors
    N=len(x)
    DX=np.zeros((N,N))
    DY=np.zeros((N,N))

    # compute distances
    xd=np.kron(np.ones((N,1)),x)  # ones((numrec,1)) precomputed
    yd=np.kron(np.ones((N,1)),y)
    d=np.sqrt((xd-np.transpose(xd))**2+(yd-np.transpose(yd))**2)
    np.fill_diagonal(d,1e12)
    FlagOverlap=d<(2*radius)

    # now computes the direction of the displacement
    q=(yd-np.transpose(yd))/(xd-np.transpose(xd)) # direction of displacement
    v=np.sign(np.transpose(xd)-xd)                # versus of displacement

    # matrix of displacements, (BounceStep might also be proportional to the % of overlap)

    AO=AreaOverlap(d[FlagOverlap])
    DX[FlagOverlap]=v[FlagOverlap]*BounceStep*(AO/(np.pi*radius**2))      
    DY[FlagOverlap]=DX[FlagOverlap]*np.sin(np.arctan(q[FlagOverlap]))

    # sum up all contributes to displacements
    dx=DX.sum(axis=1)
    dy=DY.sum(axis=1)

    idx0=np.where((dx==0)&(FlagOverlap.any(axis=1)))[0]
    dy[idx0]=scaleY*np.random.randn(len(idx0))

    dx[idxNM]=0
    dy[idxNM]=0
    x=np.transpose(x)+dx
    y=np.transpose(y)+dy 
    return x,y,np.mean(AO)

def XYregular(Nrow=8,Ncol=8):
    '''
        x->col
        y->row
        note: instead of two for cycles can use transpose and kron 
    '''
    drow=1/Nrow #(Nrow+1)    
    dcol=1/Ncol #(Ncol+1)
    row_rng=drow/2+np.arange(Nrow)*drow
    col_rng=dcol/2+np.arange(Ncol)*dcol
    xList=[]
    yList=[]
    for r in row_rng:
        for c in col_rng:
            xList.append(c)
            yList.append(r)
    return np.array(xList),np.array(yList)

def InterDistance(dist,idx_cells):
    '''
        dist Ncells x Ncells 
    '''
    dist_sub=dist[np.ix_(idx_cells,idx_cells)]
    L=len(idx_cells)
    return dist_sub[np.triu_indices(L,k=1)].mean()

'''
idxI=np.where(R['nType']==-1)[0]
dist=uti.distance(R['x'],R['y'])
uti.InterDistance(dist,idxI)
'''     

def MeanIDinh(R):
    '''
    '''
    idxI=np.where(R['nType']==-1)[0]
    dist=distance(R['x'],R['y'])
    return InterDistance(dist,idxI)
    
def Percentile_inh(d,q=10):
    '''
    '''
    dist=distance(d['R']['x'],d['R']['y'])
    idx_cells=np.where(d['R']['nType']==-1)[0];
    L=len(idx_cells)
    dist_sub=dist[np.ix_(idx_cells,idx_cells)]
    return np.percentile(dist_sub[np.triu_indices(L,k=1)],q)

def load_py(fn):
    return np.load(fn,allow_pickle=1).item()

def InhCoord(fn):
    '''
    '''
    d=load_py(fn)
    idx_cells=np.where(d['R']['nType']==-1)[0]
    L=len(idx_cells)
    XY=np.zeros((L,2))
    XY[:,0]=d['R']['x'][idx_cells]
    XY[:,1]=d['R']['y'][idx_cells]
    return XY

def Clust(fn,eps=0.2,min_samples=2,discard_minus_one=False):
    ''' cluster inhibitory neurons with DBSCAN
    '''
    from sklearn.cluster import DBSCAN
    from sklearn import metrics

    d=load_py(fn)
    # select inh neurons 
    idx_cells=np.where(d['R']['nType']==-1)[0]
    L=len(idx_cells)
    XY=np.zeros((L,2))
    XY[:,0]=d['R']['x'][idx_cells]
    XY[:,1]=d['R']['y'][idx_cells]
    # cluster 
    db = DBSCAN(eps=eps,min_samples=min_samples).fit(XY)
    labels = db.labels_
    lunLab = len(np.unique(labels))
    if lunLab>1: 
        if discard_minus_one:
            if lunLab>2:
                idx=np.where(labels!=-1)[0]            
                S=metrics.silhouette_score(XY[idx],labels[idx])            
            else:
                S=-1
        else:
            S=metrics.silhouette_score(XY,labels)
    else:
        S=-1
    # plot 
    pf.plot_circles(d['R']['x'],d['R']['y'],d['R']['nType'],d['r'][0,:])
    for u in np.unique(labels):
        idx=np.where(labels==u)[0]
        if u==-1:
            plt.plot(XY[idx,0],XY[idx,1],c='k',marker='o',markersize=14,lw=0)
        else:
            plt.plot(XY[idx,0],XY[idx,1],marker='o',markersize=14,lw=0)
    return S

'''
fn='/home/thierry/collaborations/others/LucaBologna/CirclesCodePy/results/multiple_runs/out_p1a_run%d.npy'
plt.ioff()
Sv=np.zeros(20)
for k in range(20):
    gn=fn%k    
    Sv[k]=uti.Clust(gn)
    plt.title('Silhouette=%g'%Sv[k],fontsize=16)
    plt.savefig(gn.replace('.npy','_cluster.png'))
    plt.close()
plt.ion()      
'''

def MeanOfMindistI(R):
    ''' computes the mean of the minimum distances between inh neurons 
    '''
    idxI=np.where(R['nType']==-1)[0]
    dist=distance(R['x'],R['y'])
    dist_sub=dist[np.ix_(idxI,idxI)]
    return np.mean(dist_sub.min(axis=1))

'''
fn='/home/thierry/collaborations/others/LucaBologna/CirclesCodePy/results/multiple_runs/out_p1a_run%d.npy'
mindistI=np.zeros(20)
for k in range(20):
    gn=fn%k    
    d=load_py(gn)
    mindistI[k]=uti.MeanOfMindistI(d['R'])

plt.plot(mindistI,100*(mfrPeak-mfrPlat)/mfrPlat) # no trend!
'''

def EntropyDistI(R,rng=[0,1.5],nbins=50):
    ''' computes entropy of the "inhibitory inter-distances" distribution 
    '''
    idxI=np.where(R['nType']==-1)[0]
    dist=distance(R['x'],R['y'])
    dist_sub=dist[np.ix_(idxI,idxI)]
    bins=np.linspace(rng[0],rng[1],nbins)
    h=np.histogram(dist_sub,bins=bins)[0]
    p=h/h.sum()
    Ent=-np.sum(p[p>0]*np.log2(p[p>0]))
    return Ent/np.log2(nbins)
    
'''
fn='/home/thierry/collaborations/others/LucaBologna/CirclesCodePy/results/multiple_runs/out_p1a_run%d.npy'
EntdistI=np.zeros(20)
for k in range(20):
    gn=fn%k    
    d=load_py(gn)
    EntdistI[k]=uti.EntropyDistI(d['R'])
plt.plot(EntdistI,100*(mfrPeak-mfrPlat)/mfrPlat,'ko',lw=2) 
'''

def DisposeInhibitoryNeurons(R,perc=0.5,verbose=False):
    ''' The algorithm:
            1) position all inh neurons in the lower corner of the network
            2) randomly moves a fraction (perc) of inh neurons across the network (the moved inh neurons are replaced by exc neurons)
            3) returns the same dict R with the updated 'nType' field  
        notes:
            1) the dict R needs to be populated by the 'nType' field 
            2) 
    '''
    import copy 
    Rcp=copy.deepcopy(R)
    Rcp['nType']=np.sort(Rcp['nType'])
    nInhTot=(Rcp['nType']==-1).sum()
    nInhMove=int(round(perc*nInhTot))
    Rcp['nType'][nInhTot-nInhMove:nInhTot]=1
    NrnTot=len(Rcp['nType'])
    idxI=np.random.choice(np.arange(nInhTot,NrnTot),nInhMove,replace=False)
    Rcp['nType'][idxI]=-1
    nInhTotnew=(Rcp['nType']==-1).sum()
    if verbose: print('%d inh neurons %d moved'%(nInhTot,nInhMove))
    return Rcp

def mindistEI(R):
    '''
    '''
    dd=distance(R['x'],R['y'])
    idxE=np.where(R['nType']==1)[0]
    idxI=np.where(R['nType']==-1)[0]
    distEI=dd[np.ix_(idxE,idxI)].min(axis=1) # for each exc neuron get the min distance with respect to inh neurons 
    return distEI.mean()
    
    


