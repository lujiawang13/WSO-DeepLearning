import pandas as pd
import numpy as np
def Datapreprocessing_f(Data,features):
    y = Data.loc[:, 'y']
    yl = Data.loc[:, 'yl']
    yr = Data.loc[:, 'yr']
    x = Data.loc[:,features]
    x = x.to_numpy()
    y = y.to_numpy().astype(int)
    yl = yl.to_numpy().astype(int)
    yr = yr.to_numpy().astype(int)
    K = int(y.max())

    xlist =dict()
    ylist =dict()
    Yaug=np.array([])
    for k in range(K-1):
        indx_left =np.where(yr <= (k+1))[0]
        indx_right =np.where(yl >= (k+2))[0]

        xlist[k * 2] =x[indx_left,:]
        xlist[k * 2+1] =x[indx_right,:]
        ylist[k * 2] =np.repeat(1, len(indx_left), axis=0)
        ylist[k * 2+1] =np.repeat(-1, len(indx_right), axis=0)

        Yaug=np.concatenate([Yaug,ylist[k * 2] , ylist[k * 2+1] ])
        if k==0:
            Xaug = np.concatenate((xlist[k * 2], xlist[k * 2 + 1]), axis=0)
        else:
            Xaug = np.concatenate((Xaug,xlist[k * 2], xlist[k * 2+1]), axis=0)

    Yaug=(Yaug+1)/2
    Yaug=Yaug.astype(int)
    #Xaug=pd.concat(xlist, axis=0)

    return Xaug,Yaug,K,xlist, ylist
