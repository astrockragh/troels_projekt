import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from ExternalFunctions import Chi2Regression,UnbinnedLH, BinnedLH, add_text_to_ax, nice_string_output
import pandas as pd

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
        print('Hessematrix is not accurate!')
    
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
from ExternalFunctions import Chi2Regression, nice_string_output, add_text_to_ax
from scipy.optimize import curve_fit
from scipy.stats import norm, chi2

def fit_mass(xs, vals, errs, ax = None, guesses_bkgr = [0, 0, -10, 2000], guesses_sig = [498, 6, 17000]):
    if not ax:
        fig, ax = plt.subplots(figsize = (16, 10), ncols = 2)
        ax_sig = ax[1]
        ax_all = ax[0]
        ax_all.plot(xs, vals, 'r.')
        ax_all.errorbar(xs, vals, errs, color = 'k', elinewidth = 1, capsize = 2, ls = 'none')

    def background_fit(x, a, b, c, d):
        return a * (x- 498) ** 3 + b * (x-498) ** 2 + c * (x-498) + d
    
    # The signal fit  Here gauss
    def add_signal(x, mean, sig, size):
        return size * norm.pdf(x, mean, sig)
    
    # The full fit
    def full_fit(x, mean, sig, size, a, b, c, d):
        return background_fit(x, a, b, c, d) + add_signal(x, mean, sig, size)
    
     # Background fit under here
    vals_b, cov_b = curve_fit(background_fit, xs, vals, p0 = guesses_bkgr)
    
    b1, b2, b3, b4 = vals_b
    
    bkgr_chi2 = Chi2Regression(background_fit, xs, vals, errs)
    bkgr_min  = Minuit(bkgr_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4)
    
    bkgr_min.migrad()
    
    # Plot result and save guesses
#     ax_all.plot(xs, background_fit(xs, *bkgr_min.args),'b--',  label = "background_fit")
    
    b1, b2, b3, b4 = bkgr_min.args
    s1, s2, s3 = guesses_sig
    
    # Full fit
    full_chi2 = Chi2Regression(full_fit, xs, vals, errs)
    full_min  = Minuit(full_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4, \
                       mean = s1, sig = s2, size = s3)
    
    full_min.migrad()
    
    s1, s2, s3, b1, b2, b3, b4 = full_min.args
    
    ax_all.plot(xs, full_fit(xs, *full_min.args), "k-", label = "full_fit")
    ax_all.plot(xs, background_fit(xs, *full_min.args[3:]),'b--',  label = "background_fit")
    
    ax_all.legend(loc = "upper right")
    
    # Details:
    text = {'chi2': full_min.fval, \
            'pval': chi2.sf(full_min.fval, len(xs) - len(full_min.args)), \
            'mean': f"{full_min.values['mean']:.1f} +/- {full_min.errors['mean']:.1f}",\
            'N':    f"{full_min.values['size']:.1f} +/- {full_min.errors['size']:.1f}"}
    
    text_output = nice_string_output(text)
    add_text_to_ax(0.60, 0.925, text_output, ax_all)
    
    # Plot signal seperately
    ax_sig.fill_between(xs, add_signal(xs, s1, s2, s3), color = 'red', alpha = 0.5, label = "sig fit")
    
    vals_sig = vals - background_fit(xs, b1, b2, b3, b4)
    
    ax_sig.plot(xs, vals_sig, 'r.')
    ax_sig.errorbar(xs, vals_sig, errs, color = 'k', elinewidth = 1, capsize = 2, ls = 'none')
    
    sig_amount = np.sum(add_signal(xs, s1, s2, s3))
    bak_amount = np.sum(background_fit(xs, b1, b2, b3, b4))
    
    text_a = {'sig': np.round(sig_amount), \
              'bkgr': np.round(bak_amount), \
              's/b': sig_amount / bak_amount}
    
    text_output = nice_string_output(text_a, decimals = 2)
    add_text_to_ax(0.70, 0.90, text_output, ax_sig)
    
    fig.tight_layout()
    
    bak_func = lambda x: background_fit(x, b1, b2, b3, b4)
    sig_func = lambda x: add_signal(x, s1, s2, s3)
        
    return fig, ax, full_min, bak_func, sig_func, [s1, s2, s3, b1, b2, b3, b4]

from iminuit import Minuit
from ExternalFunctions import Chi2Regression, nice_string_output, add_text_to_ax
from scipy.optimize import curve_fit
from scipy.stats import norm, chi2

def fit_mass2(xs, vals, errs, ax = None, guesses_bkgr = [0, 0, -10, 2000], guesses_sig = [498, 6, 17000],plot=True):
    guesses_bkgr[-1] = 0.5*(vals[0] + vals[-1])
    guesses_sig[-1] = 20*max(vals)#np.sqrt(2*np.pi*guesses_sig[1]**2)*(max(vals))# - guesses_bkgr[-1])
    
    if not ax and plot:
        fig, ax = plt.subplots(figsize = (16, 10), ncols = 2)
        ax_sig = ax[1]
        ax_all = ax[0]
        ax_all.plot(xs, vals, 'r.')
        ax_all.errorbar(xs, vals, errs, color = 'k', elinewidth = 1, capsize = 2, ls = 'none')

    def background_fit(x, a, b, c, d):
        return a * (x- 498) ** 3 + b * (x-498) ** 2 + c * (x-498) + d
    
    # The signal fit  Here gauss
    def sig1(x, mean, sig, size):
        return size*norm.pdf(x, mean, sig)
    
    def sig2(x, mean, sig, size):
        return size*norm.pdf(x, mean, sig)
    
    # The full fit
    def full_fit(x, mean, sig, size, f, sigmp, a, b, c, d):
        return background_fit(x, a, b, c, d) + f*sig1(x, mean, sig, size) + (1-f)*sig2(x, mean, sigmp*sig, size)
    
     # Background fit under here
    bkgr_mask = (xs < 475) | (xs > 525)
    vals_b, cov_b = curve_fit(background_fit, xs[bkgr_mask], vals[bkgr_mask], p0 = guesses_bkgr)
    b1, b2, b3, b4 = vals_b
    bkgr_chi2 = Chi2Regression(background_fit, xs[bkgr_mask], vals[bkgr_mask], errs[bkgr_mask])
    bkgr_min  = Minuit(bkgr_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4)
    bkgr_min.migrad()
    counter = 0
    while not bkgr_min.valid and counter<50:
        bkgr_min.migrad()
        counter += 1
    if not bkgr_min.valid: print("No background valid minimum found!")
    
    #Save guesses 
    b1, b2, b3, b4 = bkgr_min.args
    s1, s2, s3 = guesses_sig
    
    # Full fit
    full_chi2 = Chi2Regression(full_fit, xs, vals, errs)
    full_min  = Minuit(full_chi2, pedantic = False, a = b1, b = b2, c = b3, d = b4, \
                       mean = s1, sig = s2, size = s3, f = 0.5, sigmp = 2, \
                      limit_mean=(475,525), limit_f=(0,1), limit_size=(0,None))
    
    full_min.migrad()
    counter = 0
    while not full_min.valid and counter<200:
        full_min.migrad()
        counter += 1
    if not full_min.valid: print("No valid minimum found!")
    
    mean, sig, size, f, sigmp, b1, b2, b3, b4 = full_min.args
    
    sig_amount = np.sum(f*sig1(xs, mean, sig, size) + (1-f)*sig2(xs, mean, sigmp*sig, size))
    bak_amount = np.sum(background_fit(xs, b1, b2, b3, b4))
    
    if plot:
        ax_all.plot(xs, full_fit(xs, *full_min.args), "k-", label = "full_fit")
        ax_all.plot(xs, background_fit(xs, b1, b2, b3, b4),'b--',  label = "background_fit")

        ax_all.legend(loc = "upper right")

        # Details:
        text = {'chi2': full_min.fval, \
                'pval': chi2.sf(full_min.fval, len(xs) - len(full_min.args)), \
                'mean': f"{full_min.values['mean']:.1f} +/- {full_min.errors['mean']:.1f}",\
                'N':    f"{full_min.values['size']:.1f} +/- {full_min.errors['size']:.1f}"}

        text_output = nice_string_output(text)
        add_text_to_ax(0.60, 0.925, text_output, ax_all)

        # Plot signal seperately
        ax_sig.fill_between(xs, f*sig1(xs, mean, sig, size) + (1-f)*sig2(xs, mean, sigmp*sig, size), color = 'red', alpha = 0.5, label = "sig fit")
        ax_sig.plot(xs, f*sig1(xs, mean, sig, size),ls = '--', color = 'blue', alpha = 0.5, label = "sig fit")
        ax_sig.plot(xs, (1-f)*sig2(xs, mean, sigmp*sig, size),ls = '--', color = 'green', alpha = 0.5, label = "sig fit")

        vals_sig = vals - background_fit(xs, b1, b2, b3, b4)

        ax_sig.plot(xs, vals_sig, 'r.')
        ax_sig.errorbar(xs, vals_sig, errs, color = 'k', elinewidth = 1, capsize = 2, ls = 'none')

        text_a = {'sig': np.round(sig_amount), \
                  'bkgr': np.round(bak_amount), \
                  's/b': sig_amount / bak_amount}

        text_output = nice_string_output(text_a, decimals = 2)
        add_text_to_ax(0.70, 0.90, text_output, ax_sig)

        fig.tight_layout()
        return fig, ax, full_min, sig_amount, bak_amount
        
    return full_min, sig_amount, bak_amount

def assign_pseudolabels(train_data):
    vals, binc, binw = hist(train_data.v0_ks_mass,bins=100)
    fig, ax, M, sig, bkgr = fit_mass2(binc,vals,np.sqrt(vals))

    mean, sigma = M.values['mean'], M.values['sig']
    signal = train_data.loc[(train_data.v0_ks_mass > mean - sigma) & (train_data.v0_ks_mass < mean + sigma)]
    bkgr_l = train_data.loc[(train_data.v0_ks_mass > mean - 15*sigma) & (train_data.v0_ks_mass < mean - 10*sigma)]
    bkgr_r = train_data.loc[(train_data.v0_ks_mass > mean + 10*sigma) & (train_data.v0_ks_mass < mean + 15*sigma)]
    ax[0].vlines([mean-sigma,mean+sigma,mean-15*sigma,mean-10*sigma,mean+10*sigma,mean+15*sigma],min(vals),max(vals))

    min_sample = min([len(signal),len(bkgr_l),len(bkgr_r)])
#     if min_sample != len(signal):print("WARNING! increase background samplesize or increase signal sample-size")

    train_data = pd.concat([signal.sample(min_sample),
                       bkgr_l.sample(min_sample),
                       bkgr_r.sample(min_sample)])
    train_data['y'] = np.append(np.ones(min_sample),[np.zeros(min_sample),np.zeros(min_sample)])
    return train_data

def ROC_data(mass, p,thresholds=20,eq_intervals=False):
    if eq_intervals:
        p_ranges = np.linspace(0,1,thresholds)
    else:
        tmp_p = np.sort(p)
        p_ranges = tmp_p[np.linspace(0,len(tmp_p)-1,thresholds,dtype=int)]
    
    sig_count, bkgr_count = [], []
#     sig_count_err, bkgr_count_err = [], []
    mask = np.ones(len(p_ranges),dtype=bool)
    
    mass_fig, mass_ax = plt.subplots(figsize=(16,10))
    
    i = 0
    for score in p_ranges:
        print(i)
        vals, binc, binw = hist(mass.loc[p<score],bins=100)
        if all(vals == 0):
            print("No values, skipping..")
            mask = mask[:-1]
            continue
        
        mass_ax.plot(binc,vals,c='grey',alpha=0.5)
        
        val_mask = vals > 0
        
        fig, ax, M, sig, bkgr = fit_mass2(binc[val_mask],vals[val_mask],np.sqrt(vals[val_mask]),plot=True)
        
        if not M.valid or bkgr<0:
            mask[i] = False
        
        sig_count.append(sig)
        bkgr_count.append(bkgr)
        i += 1
    return sig_count, bkgr_count, mask