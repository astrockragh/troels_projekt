import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from AppStatFunctions import Chi2Regression,UnbinnedLH, BinnedLH, add_text_to_ax, nice_string_output
import pandas as pd
import warnings

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
from AppStatFunctions import Chi2Regression, nice_string_output, add_text_to_ax
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
from AppStatFunctions import Chi2Regression, nice_string_output, add_text_to_ax
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
        return a * (x- guesses_sig[0]) ** 3 + b * (x-guesses_sig[0]) ** 2 + c * (x-guesses_sig[0]) + d
    
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

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        full_min.migrad()
        counter = 0
        while not full_min.valid and counter<200:
            full_min.migrad()
            counter += 1
    if not full_min.valid: print("No valid minimum found!")
    
    mean, sig, size, f, sigmp, b1, b2, b3, b4 = full_min.args
    
    sig_amount = np.sum(f*sig1(xs, mean, sig, size) + (1-f)*sig2(xs, mean, sigmp*sig, size))
    bak_amount = np.sum(background_fit(xs, b1, b2, b3, b4))
    
    neg_bkgr = any(background_fit(xs, b1, b2, b3, b4)<0)
    
    def signal():
        def signal_func(x):
            return f*sig1(x, mean, sig, size) + (1-f)*sig2(x, mean, sigmp*sig, size)
        return signal_func
    
    def background():
        def background_func(x):
            return background_fit(x, b1, b2, b3, b4)
        return background_func
    
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
        return {'fig': fig,'ax': ax,'M': full_min,'sig': sig_amount,'bkgr': bak_amount,'neg_bkgr': neg_bkgr,
               'sig_func': signal(), 'bkgr_func': background()}
        
    return {'M': full_min,'sig': sig_amount,'bkgr': bak_amount,'neg_bkgr': neg_bkgr,
               'sig_func': signal(), 'bkgr_func': background()}

def assign_pseudolabels(train_data):
    vals, binc, binw = hist(train_data.v0_ks_mass,bins=100)
    d = fit_mass2(binc,vals,np.sqrt(vals))
    fig, ax, M, sig, bkgr = d['fig'], d['ax'], d['M'], d['sig'], d['bkgr']
    
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

def ROC_data(mass, p,thresholds=20,eq_intervals=False,plot_fit=True,plot_ROC=True):
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
        vals, binc, binw = hist(mass.loc[p<score],bins=100)
        if all(vals == 0):
            print("No values, skipping..")
            mask = mask[:-1]
            continue
        
        mass_ax.plot(binc,vals,c='grey',alpha=0.5)
        
        val_mask = vals > 0
        if plot_fit:
            d = fit_mass2(binc[val_mask],vals[val_mask],np.sqrt(vals[val_mask]),plot=True)
            fig, ax, M, sig, bkgr = d['fig'], d['ax'], d['M'], d['sig'], d['bkgr']
        else:
            d = fit_mass2(binc[val_mask],vals[val_mask],np.sqrt(vals[val_mask]),plot=False)
            M, sig, bkgr = d['M'], d['sig'], d['bkgr']
        
        if not M.valid or bkgr<0 or d['neg_bkgr']:
            mask[i] = False
        
        sig_count.append(sig)
        bkgr_count.append(bkgr)
        i += 1
        
    if plot_ROC:
        from scipy.integrate import quad
        
        s = np.array(sig_count)
        b = np.array(bkgr_count)
        
        auc = quad(lambda xx: np.interp(xx,np.sort(1 - b[mask]/b[mask].max()),np.sort(1 - s[mask]/s[mask].max())), \
                   0,1,points=np.sort(1 - b[mask]/b[mask].max()))
        
        fig, ax = plt.subplots(figsize=(16,7),ncols=2)
        ax[0].set_title('Unmasked')
        ax[1].set_title(f'Masked: approx AUC = {auc[0]}')
        
        ax[0].plot(1 - b/b.max(), 1 - s/s.max(),'b.')
        ax[1].plot(1 - b[mask]/b[mask].max(), 1 - s[mask]/s[mask].max(),'b.')
        
        return fig, ax, np.array(sig_count), np.array(bkgr_count), mask
        
    return np.array(sig_count), np.array(bkgr_count), mask

################################# JOHANN FUNCTION
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# For final fits
from iminuit import Minuit
from AppStatFunctions import Chi2Regression

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
    lprobs = np.sort(probs)#np.sort(logit(probs))
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
#         ax_roc.errorbar(x[:-1], y[:-1], x_errs, y_errs, elinewidth = 1, capsize = 2, color = 'k', ls = 'none')
        ax_roc.set(xlim = (-0.2, 1.2), ylim = (-0.2, 1.2))
        ax_roc.vlines([0,1],0,1,ls='--',color='gray',zorder=-1)
        ax_roc.hlines([0,1],0,1,ls='--',color='gray',zorder=-1)
        
    return AUC_estimate, cuts