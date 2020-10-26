import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from AppStatFunctions import Chi2Regression,UnbinnedLH, BinnedLH, add_text_to_ax, nice_string_output
import AppStatFunctions as ExternalFunctions
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

def logit(p):
    return np.log(p/(1-p))

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
    
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# For final fits
from iminuit import Minuit

# For primary guess fit
from scipy.optimize import curve_fit
from scipy.stats import norm, chi2

# Roc-curve related
from sklearn.metrics import auc

def double_gauss_fit(mass, bins = 100, range = (400, 600), ax = None, verbose = True, guesses = None, max_size = None, plimit = 0, color = "red"):
    # Fit third degree polynomium to background
    def background_fit(x, a, b, c, d):
        res = a * (x- 498) ** 3 + b * (x-498) ** 2 + c * (x-498) + d
        return res
    
    # The double gauss signal
    def add_signal(x, mean, sig, size, ratio, sig_ratio):
        return size * binwidth *  (ratio * norm.pdf(x, mean, sig) + \
                                   (1 - ratio) * norm.pdf(x, mean, sig_ratio * sig))
    
    # The full fit
    def full_fit(x, mean, sig, size, ratio, sig_ratio, a, b, c, d):
        return background_fit(x, a, b, c, d) + add_signal(x, mean, sig, size, ratio, sig_ratio)
    
    
    # Make histogram
    vals, edges = np.histogram(mass, bins = bins, range = range)
    xs = (edges[1:] + edges[:-1])/2

    binwidth = xs[1] - xs[0]

    mask = vals > 0
    vals = vals[mask]  
    xs = xs[mask]
    errs = np.sqrt(vals)

    # Get guesses for a background fit
    if not guesses:
        back_data_mask = abs(xs - xs[np.argmax(vals)]) > 10
        background_guess =  [0, 0, (vals[-1]- vals[0]) / 100, vals.min()]

        if len(vals[back_data_mask]) == 0:
            return None, None, None, None

        try: 
            vals_b, cov_b = curve_fit(background_fit, xs[back_data_mask], vals[back_data_mask], p0 = background_guess)
        except:
            vals_b = background_guess
        b1, b2, b3, b4 = vals_b
        
        bkgr_chi2 = Chi2Regression(background_fit, xs[back_data_mask], vals[back_data_mask], errs[back_data_mask])
        bkgr_min  = Minuit(bkgr_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4)
        bkgr_min.migrad()
        counter = 0
        while not bkgr_min.valid and counter<50:
            bkgr_min.migrad()
            counter += 1
        if not bkgr_min.valid: print("No background valid minimum found!")
        
        #Save guesses 
        b1, b2, b3, b4 = bkgr_min.args

        guesses_sig = [498, 7, 2000, 0.5, 2]
        try:
            vals_f, cov_f = curve_fit(full_fit, xs, vals, p0 = guesses_sig + [b1, b2, b3, b4])
        except:
            vals_f = np.hstack([guesses_sig, vals_b])

        s1, s2, s3, s4, s5, b1, b2, b3, b4 = vals_f
    else:
        s1, s2, s3, s4, s5, b1, b2, b3, b4 = guesses

    full_chi2 = Chi2Regression(full_fit, xs, vals, errs)
    full_min  = Minuit(full_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4, \
                       mean = s1, sig = s2, size = s3, ratio = s4, sig_ratio = s5, limit_sig_ratio = (1, 4), \
                       limit_ratio = (0, 1.0), limit_mean = (490, 510), limit_size = (0, max_size), limit_sig = (3, 10))
    full_min.migrad()
    
    full_min.migrad()
    counter = 0
    while not full_min.valid and counter<200:
        full_min.migrad()
        counter += 1
    if not full_min.valid: print("No valid minimum found!")

    # Check fit
    chi = full_min.fval 
    pval = chi2.sf(chi, np.sum(mask) - len(full_min.args))

    if verbose:
        print(f"Completed fit with Chi2: {chi:.1f}, p-val: {pval:.3f} and the total amount of signal " + \
            f"{full_min.values['size']:.0f} +/- {full_min.errors['size']:.0f}, background: {len(mass) - int(full_min.values['size'])}")


    if ax:
        ax.plot(xs, vals, alpha = 1, color = color)
#         ax.errorbar(xs, vals, errs, elinewidth = 1, color = 'k', capsize = 2, linestyle = 'none', alpha = 0.25)
#         ax.plot(xs, full_fit(xs, *full_min.args), '--', alpha = 0.5)

    if True:#full_min.errors['size'] < full_min.values['size'] and full_min.valid and pval > plimit:
        return full_min.values['size'], len(mass) - full_min.values['size'], full_min.errors['size'], full_min.args
    else:
        return None, None, None, None


def roc_curve_data(mass, probs, Npoints = 10, bins = 100, range = (400, 600), ax_roc = None , ax_fits = None, verbose = True, plimit = 0.01, ax_hist = None):
    sigs, bkgrs, errs = [], [], []
    mass = np.array(mass)
    mass = mass[np.argsort(probs)]
    cuts = (len(mass) / Npoints * np.arange(0, Npoints)).astype(int)
    args = None
    max_size = None
    from matplotlib.cm import winter
    colors = winter(np.linspace(0, 1, Npoints)[::-1])
    from scipy.special import logit
    lprobs = np.sort(logit(probs))
    if ax_hist:
        n, edges, patches = ax_hist.hist(lprobs, bins = bins, histtype = 'stepfilled', color = 'gray')
#         print(n)
        for cut, color in zip(cuts, colors):
            ax_hist.vlines(lprobs[cut], 0, max(n), color = color, linestyle = "dashed", alpha = 0.5)
            
    for i, c in zip(cuts, colors):
        bkgr, sig, err, args = double_gauss_fit(mass[i:], bins = bins, range = range, ax = ax_fits, verbose = verbose, guesses = args, max_size = max_size, plimit = plimit, color = c)
        if bkgr:
            bkgrs.append(bkgr)
            sigs.append(sig)
            errs.append(err)
            if len(sigs) == 1:
                max_size = 1 * args[2]

    sigs, bkgrs, errs = np.array(sigs), np.array(bkgrs), np.array(errs)
    y = sigs/sigs.max()
    x = bkgrs/bkgrs.max()

#     x = np.append(x, 0)[::-1]
#     y = np.append(y, 0)[::-1]

    AUC_estimate = np.trapz(np.append(x, 0), np.append(y, 0))

#     x_errs = np.sqrt((errs/sigs.max()) ** 2 + (errs[sigs.argmax()] * sigs / sigs.max() ** 2) ** 2)
#     y_errs = np.sqrt((errs/bkgrs.max()) ** 2 + (errs[bkgrs.argmax()] * bkgrs / bkgrs.max() ** 2) ** 2)
    
    if ax_roc:
        ax_roc.scatter(x, y, c = colors)
        ax_roc.vlines([0,1], 0, 1, ls = '--', color = "gray", zorder = -1)
        ax_roc.hlines([0,1], 0, 1, ls = '--', color = "gray", zorder = -1)
#         ax_roc.errorbar(x[:-1], y[:-1], x_errs, y_errs, elinewidth = 1, capsize = 2, color = 'k', ls = 'none')
        ax_roc.set(xlim = (-0.2, 1.2), ylim = (-0.2, 1.2))
    
    return AUC_estimate