import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from AppStatFunctions import Chi2Regression,UnbinnedLH, BinnedLH, add_text_to_ax, nice_string_output

def hist(data,bins=None,range=None,integers=False):
    if integers:
        vals, binedges = np.histogram(data,bins=len(range(min(data),max(data)+1)),range=(min(data)-0.5,max(data)+0.5))
    else:
        if bins == None and range == None:
            vals, binedges = np.histogram(data)
        elif range==None:
            vals, binedges = np.histogram(data,bins=bins)
        elif bins==None:
            vals, binedges = np.histogram(data,range=range)
        else:
            vals, binedges = np.histogram(data,bins=bins,range=range)
            
    bincenter = 0.5*(binedges[1:] + binedges[:-1])
    binwidth = np.mean(binedges[1:] - binedges[:-1])
    return vals, bincenter, binwidth

def gauss_chi2(data,ax,bins=None,range=None,coords=(0.05,0.95),decimals=4):
    vals, bincenter, binwidth = hist(data,bins,range)
    svals = np.sqrt(vals)
    mask = vals > 0
    
    def fitfunc(x,mu,sigma):
        normalization = len(data)*binwidth
        return normalization/np.sqrt(2*np.pi)/sigma*np.exp(-0.5*(x-mu)**2/sigma**2)
    
    chi2_obj = Chi2Regression(fitfunc,bincenter[mask],vals[mask],svals[mask])
    minuit = Minuit(chi2_obj,pedantic=False,mu=np.mean(data),sigma=np.std(data))
    minuit.migrad()
    
    if not minuit.migrad_ok():
        print('minuit.migrad() did not converge!')
    if not minuit.matrix_accurate():
        print('Hesse matrix is not accurate!')
    
    ax.errorbar(bincenter[mask],vals[mask],svals[mask],drawstyle='steps-mid',capsize=2,linewidth=1,color='k',ecolor='r',label='data')
    ax.plot(bincenter[np.logical_not(mask)],vals[np.logical_not(mask)],'gx',label='Bins with 0')
               
    x = np.linspace(min(bincenter),max(bincenter),500)
    ax.plot(x,fitfunc(x,*minuit.args),'b',label='$\\chi^2$ fit')
    
    ndof = np.sum(mask)-len(minuit.args)
    d = {'Chi2/ndof:': f"{minuit.fval:.3f}/{ndof:d}",
        "p": stats.chi2.sf(minuit.fval,ndof),
        "mu": minuit.values['mu'],
        "sigma": minuit.values['sigma']}
    add_text_to_ax(*coords,nice_string_output(d,decimals=decimals),ax,fontsize=12)
    return minuit

from iminuit import Minuit
from AppStatFunctions import Chi2Regression, nice_string_output, add_text_to_ax
from scipy.optimize import curve_fit
from scipy.stats import norm, chi2

def normalize_dataframe(dataframe, mc=False, truecol='trueKs'):
    """ Give truecol either as string or not, doesn't matter. Returns scaled dataframe w/o truelabel, truelabel series and 
    mean and standard deviation for each variable as a dictionary"""
    mustd={}
    if mc==True:
        label=dataframe[str(truecol)]
        dataframe=dataframe.drop(str(truecol), axis=1)
    df_all_norm = pd.DataFrame(None)
    for col in dataframe.columns[:57]:
        if dataframe[col].std() != 0:
            mustd[col]=(dataframe[col].mean(), dataframe[col].std())
            df_all_norm[col] = (dataframe[col] - dataframe[col].mean()) / dataframe[col].std()
    if mc==True:
        return df_all_norm, label, mustd
    else:
        return df_all_norm, mustd