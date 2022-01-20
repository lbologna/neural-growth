# 
import numpy as np
import pylab as plt 

def plot_circles_v2(R,radius,ms=10,figsize=(5,5),alpha=.1):
    '''
    '''
    plot_circles(R['x'],R['y'],R['nType'],radius,ms=ms,figsize=figsize,alpha=alpha)
    
def plot_circles(x,y,CellType,radius,ms=10,figsize=(5,5),alpha=.1):
    ''' plot circles with different colors (red exc, blue inh, based on CellType 1 exc, -1 inh)
        radius can either be an array (same size of x,y,CellType) or a constant (same radius for all)   
    '''
    if type(radius)==float: radius=np.repeat(radius,len(x))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x,y,c='k',marker='.',markersize=ms,linestyle='None')
    for i in range(len(x)):
        if CellType[i]==1:
            circle1 = plt.Circle((x[i],y[i]), radius[i], color = 'r', alpha=alpha) #,fill=False)
        else:
            circle1 = plt.Circle((x[i],y[i]), radius[i], color = 'b', alpha=alpha) #,fill=False)
        ax.add_artist(circle1)
    plt.axis([0,1,0,1])
    plt.axis('equal')

def plot_mfr_time(time,mfr,mfr_err):
    '''
    '''
    plt.errorbar(time,y=mfr,yerr=mfr_err,c='k',marker='o',markersize=8,lw=2,linestyle='-')
    plt.xlabel('time (ms)',fontsize=14)
    plt.ylabel('MFR (Hz)',fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)        

