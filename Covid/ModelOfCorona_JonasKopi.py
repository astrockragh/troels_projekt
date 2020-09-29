# ----------------------------------------------------------------------------------------------------------- #
#
# Program developped for simple S(E)IR model of corona virus infections in Denmark.
#
# Data from:
#
#   Developed by: Troels C. Petersen (NBI, petersen@nbi.dk), Mathias Heltberg, and Christian Michelsen
#   First version:  24. Mar. 2020 by Troels Petersen
#   Latest version: 15. Sep. 2020
#
# ----------------------------------------------------------------------------------------------------------- #

from __future__ import division, print_function
from numba import jit
import importlib
import Config
from ExternalFunctions import nice_string_output, add_text_to_ax
from ExternalFunctions import UnbinnedLH, Chi2Regression

import sys
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from iminuit import Minuit
from scipy import stats
import pandas as pd
from scipy import interpolate

sys.path.append('External_Functions')


# ----------------------------------------------------------------------------------------------------------- #
plt.close('all')

Config = importlib.reload(Config)
case = Config.load_case

# 61000 tested with 2 days between
# 21000 Tested positive
# 10000 from week 19 and forward

# Antal tested 22576, med 10053 efter uge 19 (hvor tracing startede) hvoraf 650 positive (6.5%).

# Ialt er der 10000 testede positive efter uge 19.

# Det lader til, at dem, som tester negativ paa dag 4, men positiv paa dag 6 (69 ialt) er "meget normalt",
# idet de udgoer 0.3% af alle testede, hvilket ogsaa er hvad man finder generelt.

# Saa skal man stadig teste paa dag 6?


# Desuden:
# Lokale udbrud bliver maaske "forstoerret" i data, fordi der bliver testet mere (ved mobilt telt).

# ----------------------------------------------------------------------------------------------------------- #
# Load specific case from config file


# # Municipalities considered Copenhagen area (from a Covid-19 outbreak point of view):
# area_cph = ["Copenhagen", "Frederiksberg", "Gentofte", "Dragør", "Vallensbæk", "Ballerup", "Gladsaxe", "Herlev", "Hvidovre", "Lyngby-Taarbæk", "Rødovre", "Brøndby", "Glostrup", "Høje-Taastrup", "Tårnby", "Ishøj", "Albertslund"]

# # Municipalities considered Aarhus area (from a Covid-19 outbreak point of view):
# area_aarhus = ["Aarhus"]


# ----------------------------------------------------------------- #
# The data:
# ----------------------------------------------------------------- #

data_posi = pd.read_csv("Municipality_cases_time_series.csv",
                        sep=';', thousands='.', index_col=0)
data_test = pd.read_csv(
    "Municipality_tested_persons_time_series.csv", sep=';', thousands='.', index_col=0)

print(data_test.head(10))
print("...")
print(data_test.tail(10))

# Load Municipalities
Nposi = data_posi.loc["2020-03-01":"2020-09-24"][case['Municipalities']
                                                 ].values.sum(axis=1)
Ntest = data_test.loc["2020-03-01":"2020-09-24"][case['Municipalities']
                                                 ].values.sum(axis=1)
eNtest = np.sqrt(Ntest)
# # Two 1D arrays:
# Nposi_aarhus = data_posi.loc["2020-03-01":"2020-09-16"][area_aarhus].values.sum(axis=1)
# Ntest_aarhus = data_test.loc["2020-03-01":"2020-09-16"][area_aarhus].values.sum(axis=1)
# eNtest_aarhus = np.sqrt(Ntest_aarhus)

# Nposi_cph = data_posi.loc["2020-03-01":"2020-09-16"][area_cph].values.sum(axis=1)
# Ntest_cph = data_test.loc["2020-03-01":"2020-09-16"][area_cph].values.sum(axis=1)
# eNtest_cph = np.sqrt(Ntest_cph)

# # All of Denmark
# Nposi_all = data_posi.loc["2020-03-01":"2020-09-16"].sum(axis=1)
# Ntest_all = data_test.loc["2020-03-01":"2020-09-16"].sum(axis=1)
# eNtest_all = np.sqrt(Ntest_all)


# Make a list of days with the proper length:
print("Number of days for which there is data:", len(data_posi))

# Get the number of positive tests and uncertainty, assume a fixed number of daily tests (=3000):
dayLockdown_aarhus = case['dayLockdown']
# nAveDailyTests_aarhus = case['scale_test']
fPos = Nposi / Ntest
nPos_aarhus = case['scale_test'] * fPos
enPos_aarhus = case['scale_test'] * \
    np.sqrt(fPos*(1.0-fPos)/Ntest + case['SystError']**2)
day = np.arange(0, len(nPos_aarhus))
# dayLockdown_cph = 192   # Unknown!
# nAveDailyTests_cph = 12000.0
# fPos_cph = Nposi_cph / Ntest_cph
# nPos_cph  = nAveDailyTests_cph * fPos_cph
# enPos_cph = nAveDailyTests_cph * np.sqrt(fPos_cph*(1.0-fPos_cph)/Ntest_cph + SystError**2)

Plot_StartDay, Plot_EndDay = case['Plot_range']
Fit_StartDay, Fit_EndDay = case['Fit_range']
save_plots = case['save_plots']
if False:
    # Number of tests in Aarhus and Cph as a function of time:
    fig_Ntests, ax_Ntests = plt.subplots(figsize=(12, 7))
    ax_Ntests.set(xlabel="Day (1st of March is day 0)",
                  ylabel="Tests / day", title=f"Test per day for {case['Name']}")
    ax_Ntests.errorbar(day[Plot_StartDay:Plot_EndDay], Ntest[Plot_StartDay:Plot_EndDay],
                       yerr=eNtest[Plot_StartDay:Plot_EndDay], fmt='.', linewidth=2, label=f"N tests in {case['Name']}", color='red')
    # ax_Ntests.errorbar(day[Plot_StartDay:Plot_EndDay], Ntest_cph[Plot_StartDay:Plot_EndDay], yerr=eNtest_cph[Plot_StartDay:Plot_EndDay], fmt='.', linewidth=2, label='N tests in CphArea', color='blue')
    ax_Ntests.legend(loc=(0.05, 0.75), fontsize=16)
    d_N = {
        'Mean daily tests:': f"{int(np.mean(Ntest[Fit_StartDay:Fit_EndDay]))}"}
    add_text_to_ax(0.05, 0.65, nice_string_output(
        d_N, decimals=0), ax=ax_Ntests, fontsize=12)
    plt.tight_layout()
    plt.show(block=False)
    if (save_plots):
        plt.savefig(f"figures/{case['Name']}_NDailyTests.pdf")


# ----------------------------------------------------------------- #
# SIR model with time variation in beta:
# ----------------------------------------------------------------- #

# SIR Model, including modelling varying beta:
def func_SIRmodel(x, dayLockdown, nI0, beta0, beta1, gamma):

    # These numbers are the current status for each type of case, NOT cumulative.
    NdaysModelExtendsBeyondData = 5
    NdaysModel = len(nPos_aarhus) + NdaysModelExtendsBeyondData
    mS = np.zeros(NdaysModel)   # Susceptible
    mI = np.zeros(NdaysModel)   # Infected
    mR = np.zeros(NdaysModel)   # Recovered
    mTot = np.zeros(NdaysModel)   # Overall number (for check)

    # Initial numbers:
    dayN = Fit_StartDay
    mS[dayN] = 5800000-nI0
    mI[dayN] = nI0
    mR[dayN] = 0
    mTot[dayN] = mS[dayN]+mI[dayN]     # There are no Recovered, yet!

    # Model loop:
    # -----------------
    while (dayN < len(nPos_aarhus)-1 and mI[dayN] > 0):
        dayN += 1
        if (dayN < dayLockdown):     # Could potentially be a fitting parameter!
            beta = beta0
#        elif (dayN < 25) :
#            beta = beta1
        else:
            beta = beta1
            # beta = max(beta0 - dbeta*(dayN-dayLockdown), beta_inf)

        dI = beta*mI[dayN-1] * (mS[dayN-1] / mTot[dayN-1])
        dR = gamma*mI[dayN-1]
        mS[dayN] = mS[dayN-1] - dI
        mI[dayN] = mI[dayN-1] + dI - dR
        mR[dayN] = mR[dayN-1] + dR
        mTot[dayN] = mS[dayN] + mI[dayN] + \
            mR[dayN]       # Should remain constant!

    return mI[x]


func_SIRmodel_vec = np.vectorize(func_SIRmodel)


# ----------------------------------------------------------------- #
# SEIR model with time variation in beta:
# ----------------------------------------------------------------- #

# SEIR model, including modelling of time delays and varying beta:
@jit
def func_SEIRmodel(x, dayLockdown, nI0, beta0, beta1, lambdaE, lambdaI):

    # Initial numbers:
    N_tot = 5800000
    S = N_tot - nI0

    # The initial number of exposed and infected are scaled to match beta0. Factors in front are ad hoc!
    Norm = np.exp(0.8*lambdaE * beta0) + np.exp(0.7*lambdaE * beta0) + np.exp(0.6*lambdaE * beta0) + np.exp(0.5*lambdaE * beta0) +\
        np.exp(0.4*lambdaI * beta0) + np.exp(0.3*lambdaI * beta0) + \
        np.exp(0.2*lambdaI * beta0) + np.exp(0.1*lambdaI * beta0)
    E1 = nI0 * np.exp(0.8*lambdaE * beta0) / Norm
    E2 = nI0 * np.exp(0.7*lambdaE * beta0) / Norm
    E3 = nI0 * np.exp(0.6*lambdaE * beta0) / Norm
    E4 = nI0 * np.exp(0.5*lambdaE * beta0) / Norm
    I1 = nI0 * np.exp(0.4*lambdaI * beta0) / Norm
    I2 = nI0 * np.exp(0.3*lambdaI * beta0) / Norm
    I3 = nI0 * np.exp(0.2*lambdaI * beta0) / Norm
    I4 = nI0 * np.exp(0.1*lambdaI * beta0) / Norm
    """
    E1 = nI0/8
    E2 = nI0/8
    E3 = nI0/8
    E4 = nI0/8
    I1 = nI0/8
    I2 = nI0/8
    I3 = nI0/8
    I4 = nI0/8
    """
    R = 0
    # We define the first day given in the array of days (x) as day 0:
    dayN = Fit_StartDay
    NdaysModelExtendsBeyondData = 5
    NdaysModel = len(nPos_aarhus) + NdaysModelExtendsBeyondData

    # We store the results (time, S, E, I, and R) for each day here:
    SEIR_result = np.zeros((NdaysModel, 5))
    SEIR_result[dayN] = [dayN, S, E1+E2+E3+E4, I1+I2+I3+I4, R]

    # Numerical settings:
    nStepsPerDay = 24           # We simulated in time steps of 1 hour
    dt = 1.0 / nStepsPerDay     # Time step length in days

    # Model loop:
    # -----------------
    while (dayN < NdaysModel-1 and I1 >= 0):

        dayN += 1
        if (dayN < dayLockdown):
            beta = beta0
        else:
            beta = beta1
            # beta = max(beta0 - dbeta*(dayN-dayLockdown), beta_inf)   # Minimum value of beta (at time infinity)

        # Now divide the daily procedure into time steps:
        for _ in range(nStepsPerDay):
            dS = -beta*(I1+I2+I3+I4) * (S / N_tot)
            dE1 = -dS - lambdaE * E1
            dE2 = lambdaE * E1 - lambdaE * E2
            dE3 = lambdaE * E2 - lambdaE * E3
            dE4 = lambdaE * E3 - lambdaE * E4
            dI1 = lambdaE * E4 - lambdaI * I1
            dI2 = lambdaI * I1 - lambdaI * I2
            dI3 = lambdaI * I2 - lambdaI * I3
            dI4 = lambdaI * I3 - lambdaI * I4
            dR = lambdaI * I4

            S += dt * dS
            E1 += dt * dE1
            E2 += dt * dE2
            E3 += dt * dE3
            E4 += dt * dE4
            I1 += dt * dI1
            I2 += dt * dI2
            I3 += dt * dI3
            I4 += dt * dI4
            R += dt * dR
            Tot = S + E1+E2+E3+E4 + I1+I2+I3+I4 + R

        # Record status of model every day:
        SEIR_result[dayN] = [dayN, S, E1+E2+E3+E4, I1+I2+I3+I4, R]

        # print(dayN, E1+E2+E3+E4, I1+I2+I3+I4, R)

    # Return only the number of infected and only for the relevant days:
    return SEIR_result[x, 3]


func_SEIRmodel_vec = np.vectorize(func_SEIRmodel)


# ----------------------------------------------------------------- #
# Plot/print SIR and/or SEIR model with given parameters (for testing):
# ----------------------------------------------------------------- #

# Plot data (scaled to a fixed number of daily tests):
doSIRModelPlot = case['doSIRModelPlot']
doSEIRModelPlot = case['doSEIRModelPlot']
if (doSIRModelPlot or doSEIRModelPlot):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set(xlabel="Day (1st of March is day 0)",
           ylabel="Newly infected / day", title="")
    ax.errorbar(day[Plot_StartDay:Plot_EndDay], nPos_aarhus[Plot_StartDay:Plot_EndDay],
                yerr=enPos_aarhus[Plot_StartDay:Plot_EndDay], fmt='.', linewidth=2, label='Data (scaled)', color='red')


# Plot SIR model on top of data (with fixed parameters):
if (doSIRModelPlot):

    # Model (fixed parameters):
    nI0_plot = 2.0
    beta0_plot = 0.34
    beta1_plot = 0.06
    gamma_plot = 1.0/7.0
    ax.plot(day[Fit_StartDay:Fit_EndDay], func_SIRmodel_vec(day[Fit_StartDay:Fit_EndDay], dayLockdown_aarhus,
                                                            nI0_plot, beta0_plot, beta1_plot, gamma_plot), 'blue', linewidth=1.0, label='SIR Model')

    # Calculate Chi2, Ndof, and Chi2-Probability for this model:
    nPos_est = func_SIRmodel(day[Fit_StartDay:Fit_EndDay],
                             dayLockdown_aarhus, nI0_plot, beta0_plot, beta1_plot, gamma_plot)
    chi2 = np.sum(((nPos_aarhus[Fit_StartDay:Fit_EndDay] -
                    nPos_est) / enPos_aarhus[Fit_StartDay:Fit_EndDay])**2)
    Ndof = len(day[Fit_StartDay:Fit_EndDay]) - 5
    Prob = stats.chi2.sf(chi2, Ndof)
    print(
        f"  SIR Model (fixed for plot):  Prob(Chi2={chi2:6.1f}, Ndof={Ndof:3d}) = {Prob:7.5f}")


# Plot SEIR model on top of data (with fixed parameters):
if (doSEIRModelPlot):

    # Model (fixed parameters):
    nI0_plot = case['nI0_plot']
    beta0_plot = case['beta0_plot']
    beta1_plot = case['beta1_plot']
    lambdaE_plot = case['lambdaE_plot']
    lambdaI_plot = case['lambdaE_plot']
    ax.plot(day[Fit_StartDay:Fit_EndDay], func_SEIRmodel_vec(day[Fit_StartDay:Fit_EndDay], dayLockdown_aarhus-3,
                                                             nI0_plot, beta0_plot, beta1_plot, lambdaE_plot, lambdaI_plot), 'green', linewidth=1.0, label='SEIR Model')

    # Calculate Chi2, Ndof, and Chi2-Probability for this model:
    nPos_est = func_SEIRmodel_vec(day[Fit_StartDay:Fit_EndDay], dayLockdown_aarhus,
                                  nI0_plot, beta0_plot, beta1_plot, lambdaE_plot, lambdaI_plot)
    chi2 = np.sum(((nPos_aarhus[Fit_StartDay:Fit_EndDay] -
                    nPos_est) / enPos_aarhus[Fit_StartDay:Fit_EndDay])**2)
    # There are 5 parameters in the fit!
    Ndof = len(day[Fit_StartDay:Fit_EndDay]) - 5
    Prob = stats.chi2.sf(chi2, Ndof)
    print(
        f"  SEIR Model (fixed for plot):  Prob(Chi2={chi2:6.1f}, Ndof={Ndof:3d}) = {Prob:7.5f}")


if (doSIRModelPlot or doSEIRModelPlot):
    ax.legend(loc=(0.05, 0.75), fontsize=16)
    plt.tight_layout()
    plt.show(block=False)
    # if (save_plots) :
    #     plt.savefig("figures/fig_CoronaDataDK_ModelPlot_SEIR2_Sep23.pdf")


# ----------------------------------------------------------------- #
# Fit to data with SIR model:
# ----------------------------------------------------------------- #

# Plot Data:
# ----------
if (case['doFitSIR'] or case['doFitSEIR']):
    print("\n  ------------------------------------  Fitting Stage  --------------------------------------- \n")

    fig_fit, ax_fit = plt.subplots(figsize=(12, 7))
    ax_fit.errorbar(day[Plot_StartDay:Plot_EndDay], nPos_aarhus[Plot_StartDay:Plot_EndDay], yerr=enPos_aarhus[Plot_StartDay:Plot_EndDay],
                    fmt='.', linewidth=2, color="red", label='Newly infected (scaled to 3000 tests)')
    ax_fit.set(xlabel="Day (1st of March is day 0)",
               ylabel="Newly infected / day (scaled)", title="")


# SIR model fit:
# --------------
if (case['doFitSIRinit']):

    endday_OK = Fit_StartDay
    for i_endday in range(Fit_StartDay, Fit_EndDay):
        # Defining Chi2 calculation:
        def ChiSquareCalcSIRinit(dayL, nI0, beta0, beta1, gamma):
            nPos_est = func_SIRmodel(
                day[Fit_StartDay:i_endday], dayL, nI0, beta0, beta1, gamma)
            chi2 = np.sum(
                ((nPos_aarhus[Fit_StartDay:i_endday] - nPos_est) / enPos_aarhus[Fit_StartDay:i_endday])**2)
            return chi2

        minuit_SIRinit = Minuit(ChiSquareCalcSIRinit, pedantic=False, print_level=0, dayL=1000,
                                fix_dayL=True, nI0=0.5, beta0=0.45, beta1=0.06, gamma=0.14, fix_gamma=True)
        minuit_SIRinit.migrad()
        # if (not minuit_SIRinit.fmin.is_valid) :
        # print("  WARNING: The ChiSquare fit DID NOT converge!!! ")

        chi2 = minuit_SIRinit.fmin.fval
        Ndof = len(day[Fit_StartDay:Fit_EndDay]) - len(minuit_SIRinit.args)
        Prob = stats.chi2.sf(chi2, Ndof)
        print(
            f"  SIRinit Model fit:  Day={i_endday:3d}   Prob(Chi2={chi2:6.1f}, Ndof={Ndof:3d}) = {Prob:7.5f}")

        # If model is acceptable (prob > 5%), then save the parameters:
        if (Prob > 0.05):
            dayL_fit, nI0_fit, beta0_fit, beta1_fit, gamma_fit = minuit_SIRinit.args
            endday_OK = i_endday
        else:
            continue

    # Plot the last acceptable fitted model on top of data extending in time:
    ax_fit.plot(day[Fit_StartDay:endday_OK], func_SIRmodel_vec(day[Fit_StartDay:endday_OK], dayL_fit, nI0_fit,
                                                               beta0_fit, beta1_fit, gamma_fit), 'blue', linewidth=1.0, label='Fit with SIR Model - no intervention')
    ax_fit.plot(day[endday_OK-1:endday_OK+2], func_SIRmodel_vec(day[endday_OK-1:endday_OK+2],
                                                                dayL_fit, nI0_fit, beta0_fit, beta1_fit, gamma_fit), 'blue', linewidth=1.0, linestyle="dotted")


Fit_EndDay_delay = case['Fit_EndDay_delay']

# SIR model fit:
# --------------
if (case['doFitSIR']):

    Fit_EndDay = endday_OK + Fit_EndDay_delay

    # Defining Chi2 calculation:
    def ChiSquareCalcSIR(dayL, nI0, beta0, beta1, gamma):
        nPos_est = func_SIRmodel(
            day[Fit_StartDay:Fit_EndDay], dayL, nI0, beta0, beta1, gamma)
        chi2 = np.sum(((nPos_aarhus[Fit_StartDay:Fit_EndDay] -
                        nPos_est) / enPos_aarhus[Fit_StartDay:Fit_EndDay])**2)
        return chi2

    minuit_SIR = Minuit(ChiSquareCalcSIR, pedantic=False, print_level=0,
                        dayL=dayLockdown_aarhus, limit_dayL=[150, 208], nI0=2.0, beta0=0.45, beta1=0.04, gamma=0.14, fix_gamma=True)
    minuit_SIR.migrad()
    if (not minuit_SIR.fmin.is_valid):
        print("  WARNING: The ChiSquare fit DID NOT converge!!! ")

    # Same as minuit_SIR.args
    dayL_fit, nI0_fit, beta0_fit, beta1_fit, gamma_fit = minuit_SIR.values.values()
    edayL_fit, enI0_fit, ebeta0_fit, ebeta1_fit, egamma_fit = minuit_SIR.errors.values()
    if dayL_fit > max(day):
        dayL_fit = 208
    for name in minuit_SIR.parameters:
        print("Fit value (exp): {0} = {1:.5f} +/- {2:.5f}".format(name,
                                                                  minuit_SIR.values[name], minuit_SIR.errors[name]))
    chi2 = minuit_SIR.fmin.fval
    Ndof = len(day[Fit_StartDay:Fit_EndDay]) - len(minuit_SIR.args)
    Prob = stats.chi2.sf(chi2, Ndof)
    print(
        f"  SIR Model fit:  Prob(Chi2={chi2:6.1f}, Ndof={Ndof:3d}) = {Prob:7.5f}")

    # Plot fitted model on top of data extending in time:
    ax_fit.plot(day[Fit_StartDay:Fit_EndDay], func_SIRmodel_vec(day[Fit_StartDay:Fit_EndDay], dayL_fit,
                                                                nI0_fit, beta0_fit, beta1_fit, gamma_fit), 'blue', linewidth=2.0, label='Fit with SIR Model')
    ax_fit.plot(day[Fit_EndDay-1:Fit_EndDay+15], func_SIRmodel_vec(day[Fit_EndDay-1:Fit_EndDay+15],
                                                                   dayL_fit, nI0_fit, beta0_fit, beta1_fit, gamma_fit), 'blue', linewidth=1.0, linestyle="dotted")
    ax_fit.legend(loc=(0.05, 0.80), fontsize=12)
    d_fit = {'SIR-model': "",
             'Prob Chi2': f"{Prob:.5f}",
             r'$\beta_0$     ': f"{minuit_SIR.values['beta0']:.4f}+-{minuit_SIR.errors['beta0']:.4f}",
             r'$\beta_1$    ': f"{minuit_SIR.values['beta1']:.4f}+-{minuit_SIR.errors['beta1']:.4f}",
             "Lockdown": f"{data_posi.iloc[int(dayL_fit)].name}, ({int(dayL_fit)})"}
    add_text_to_ax(0.05, 0.55, nice_string_output(
        d_fit, decimals=0), ax=ax_fit, fontsize=9)

animate = False
# SEIR model fit:
# ---------------
if (case['doFitSEIR']):
    Fit_EndDay = endday_OK + Fit_EndDay_delay

    # Defining Chi2 calculation:
    def ChiSquareCalcSEIR(dayL, nI0, beta0, beta1, lambdaE, lambdaI):
        nPos_est = func_SEIRmodel(
            day[Fit_StartDay:Fit_EndDay], dayL, nI0, beta0, beta1, lambdaE, lambdaI)
        chi2 = np.sum(((nPos_aarhus[Fit_StartDay:Fit_EndDay] -
                        nPos_est) / enPos_aarhus[Fit_StartDay:Fit_EndDay])**2)
        return chi2

    minuit_SEIR = Minuit(ChiSquareCalcSEIR, pedantic=False, print_level=0, dayL=dayLockdown_aarhus-4, fix_dayL=False,
                         limit_dayL=[150, 208], nI0=5.0, beta0=0.40, beta1=0.15, lambdaE=1.0, lambdaI=1.0, fix_lambdaE=True, fix_lambdaI=True)
    minuit_SEIR.migrad()
    if (not minuit_SEIR.fmin.is_valid):
        print("  WARNING: The ChiSquare fit DID NOT converge!!! ")

    dayL_fit, nI0_fit, beta0_fit, beta1_fit, lambdaE_fit, lambdaI_fit = minuit_SEIR.args
    dayL_fiterr, nI0_fiterr, beta0_fiterr, beta1_fiterr = minuit_SEIR.errors[
        "dayL"], minuit_SEIR.errors["nI0"], minuit_SEIR.errors["beta0"], minuit_SEIR.errors["beta1"]
    if dayL_fit > 208:
        dayL_fit = 208
    for name in minuit_SEIR.parameters:
        print("Fit value (exp): {0} = {1:.5f} +/- {2:.5f}".format(name,
                                                                  minuit_SEIR.values[name], minuit_SEIR.errors[name]))
    chi2 = minuit_SEIR.fmin.fval
    Ndof = len(day[Fit_StartDay:Fit_EndDay]) - len(minuit_SEIR.args)
    Prob = stats.chi2.sf(chi2, Ndof)
    print(
        f"  SEIR Model fit:  Prob(Chi2={chi2:6.1f}, Ndof={Ndof:3d}) = {Prob:7.5f}")

    # Plot fitted model on top of data extending in time:
    ax_fit.plot(day[Fit_StartDay:Fit_EndDay], func_SEIRmodel_vec(day[Fit_StartDay:Fit_EndDay], dayL_fit, nI0_fit,
                                                                 beta0_fit, beta1_fit, lambdaE_fit, lambdaI_fit), 'green', linewidth=2.0, label='Fit with SEIR Model')
    ax_fit.plot(day[Fit_StartDay:Fit_EndDay+15], func_SEIRmodel_vec(day[Fit_StartDay:Fit_EndDay+15], dayL_fit,
                                                                    nI0_fit, beta0_fit, beta1_fit, lambdaE_fit, lambdaI_fit), 'green', linewidth=2.0, linestyle="dotted")
    ax_fit.legend(loc=(0.03, 0.75), fontsize=12)

    # Plot an "envelope" of possible predictions:
    ax_fit.plot(day[Fit_StartDay:Fit_EndDay+10], func_SEIRmodel_vec(day[Fit_StartDay:Fit_EndDay+10], dayL_fit, nI0_fit,
                                                                    beta0_fit, beta1_fit+1*ebeta1_fit, lambdaE_fit, lambdaI_fit), 'green', linewidth=1.0, linestyle="dotted")
    ax_fit.plot(day[Fit_StartDay:Fit_EndDay+10], func_SEIRmodel_vec(day[Fit_StartDay:Fit_EndDay+10], dayL_fit, nI0_fit,
                                                                    beta0_fit, beta1_fit-1*ebeta1_fit, lambdaE_fit, lambdaI_fit), 'green', linewidth=1.0, linestyle="dotted")
    ax_fit.fill_between(day[Fit_StartDay:Fit_EndDay+10], func_SEIRmodel_vec(day[Fit_StartDay:Fit_EndDay+10], dayL_fit, nI0_fit, beta0_fit, beta1_fit+1*ebeta1_fit, lambdaE_fit,
                                                                            lambdaI_fit), func_SEIRmodel_vec(day[Fit_StartDay:Fit_EndDay+10], dayL_fit, nI0_fit, beta0_fit, beta1_fit-1*ebeta1_fit, lambdaE_fit, lambdaI_fit), color='green', alpha=0.3)

    ax_fit.set_title(f"Fit of outbreak in {case['Name']}")
    d_fit = {'SEIR-model': "",
             'Prob Chi2': f"{Prob:.5f}",
             r'$\beta_0$      ': f"{minuit_SEIR.values['beta0']:.4f}+-{minuit_SEIR.errors['beta0']:.4f}",
             r'$\beta_1$      ': f"{minuit_SEIR.values['beta1']:.4f}+-{minuit_SEIR.errors['beta1']:.4f}",
             "Lockdown": f"{data_posi.iloc[int(dayL_fit)].name}, ({int(dayL_fit)})"}
    add_text_to_ax(0.05, 0.72, nice_string_output(
        d_fit, decimals=0), ax=ax_fit, fontsize=9)

    if (case['doFitSIR'] or case['doFitSEIR']):
        plt.tight_layout()
        plt.show(block=False)
        if (save_plots):
            plt.savefig(f'figures/{case["Name"]}_modelFit.pdf')

    fig_res, ax_res = plt.subplots(2, figsize=(9, 9))
    ax_res[0].errorbar(day[Fit_StartDay:Fit_EndDay+10], (nPos_aarhus[Fit_StartDay:Fit_EndDay+10]-func_SEIRmodel_vec(day[Fit_StartDay:Fit_EndDay+10], dayL_fit, nI0_fit,
                                                                                                                    beta0_fit, beta1_fit+1*ebeta1_fit, lambdaE_fit, lambdaI_fit)), enPos_aarhus[Fit_StartDay:Fit_EndDay+10],
                       fmt='.', linewidth=2, color="red", label='Residuals)')
    ax_res[0].plot(day[Fit_StartDay:Fit_EndDay+10],
                   np.zeros(len(day[Fit_StartDay:Fit_EndDay+10])), 'k--')
    ax_res[0].set(xlabel='Day after day 0', ylabel="fit residual",
                  title=f"Residuals for {case['Name']}")
    ax_res[0].legend()
    simulation = True
    if simulation:
        days_pos = day[int(np.rint(dayL_fit)):]
        val = minuit_SEIR.args
        from tqdm import tqdm

        def simul(days, vals, cov, N=int(1e5)):
            ys = []
            mu = []
            sig_up = []
            sig_down = []
            sig2_up, sig2_down = [], []
            means = [vals[0], vals[1], vals[2], vals[3]]
            vals = np.random.multivariate_normal(means, cov, size=N).T
            day_L = vals[0]
            n_I0 = vals[1]
            beta_0 = vals[2]
            beta_1 = vals[3]
            print('Simulating MC')
            for i in tqdm(range(N)):
                ys.append(func_SEIRmodel(
                    days, day_L[i], n_I0[i], beta_0[i], beta_1[i], lambdaE_fit, lambdaI_fit))
            ys = np.array(ys)
            ys = ys.T
            for day in ys:
                mu.append(np.mean(day))
                per = np.quantile(day, [0.5+0.34, 0.5-0.34])
                per2 = np.quantile(day, [0.5+0.49, 0.5-0.49])
                sig_up.append(per[0])
                sig_down.append(per[1])
                sig2_up.append(per2[0])
                sig2_down.append(per2[1])
            return np.array(mu), np.array(sig_up), np.array(sig_down), np.array(sig2_up), np.array(sig2_down)
        # playtime tomorrow
        multi = False
        if not multi:
            mu, sigup, sigdown, sig2up, sig2down = simul(days=days_pos, vals=minuit_SEIR.args,
                                                         cov=minuit_SEIR.matrix())
        else:
            import multiprocessing as mp
            import os
            mus, sigs = [], []
            ps = []

            def p(x, txt='Hello'):
                print(txt+x)
                return(x)
            for _ in range(os.cpu_count()):
                # p = mp.Process(target=simul, args=[days_pos)
                p = mp.Process(target=p, args=['Ciara'])
                p.start()
                ps.append(p)
            for process in ps:
                process.join()
        ax_res[1].errorbar(day[Fit_StartDay:Fit_EndDay], nPos_aarhus[Fit_StartDay:Fit_EndDay], yerr=enPos_aarhus[Fit_StartDay:Fit_EndDay],
                           fmt='.', linewidth=2, color="red", alpha=0.8, label='Newly infected (scaled to 3000 tests)')
        ax_res[1].errorbar(day[Fit_EndDay:], nPos_aarhus[Fit_EndDay:], yerr=enPos_aarhus[Fit_EndDay:],
                           fmt='.', linewidth=2, color="purple", alpha=0.8, label='Points for evaluating prediction')
        diff = Fit_EndDay-int(np.rint(dayL_fit))
        ax_res[1].plot(days_pos[:diff+1], mu[:diff+1],
                       'g-', label='Central values')
        ax_res[1].plot(days_pos[diff:], mu[diff:],
                       color='green', linestyle='dotted')
        ax_res[1].plot(days_pos, sigup, 'g--',  label='One sigma deviation')
        ax_res[1].plot(days_pos, sigdown, "g--")
        ax_res[1].fill_between(days_pos, sigup, sigdown,
                               color='green', alpha=0.4)
        ax_res[1].plot(days_pos, sig2up, 'g--',  label='Two sigma deviation')
        ax_res[1].plot(days_pos, sig2down, "g--")
        ax_res[1].fill_between(days_pos, sig2up, sig2down,
                               color='darkgreen', alpha=0.2)
        ax_res[1].set(xlabel='Day after March 1st',
                      ylabel="Scaled cases", title="MC errors")
        ax_res[1].vlines(dayL_fit, 0, max(nPos_aarhus[150:]),
                         linestyles='dashed', alpha=0.6, label='Fitted lockdown')
        ax_res[1].vlines(Fit_EndDay-0.5, 0, max(nPos_aarhus[150:]),
                         linestyles='solid', label='Extrapolating from')
        ax_res[1].legend(loc='best', fontsize=9)
        delay = diff
        case_pos = nPos_aarhus[int(np.rint(dayL_fit))+delay:]
        pos_err = np.sqrt(enPos_aarhus[int(np.rint(
            dayL_fit))+delay:]**2-case['SystError']**2)
        mu, sigup, sigdown = mu[delay:], sigup[delay:], sigdown[delay:]
        chi2list = 0
        diflist = case_pos-mu
        k = 0
        for dif in diflist:
            if dif > 0:
                chi2list += (mu[k]-case_pos[k])**2 / \
                    ((pos_err[k]**2+sigup[k]**2))
            if dif < 0:
                chi2list += (mu[k]-case_pos[k])**2 / \
                    ((pos_err[k]**2+sigdown[k]**2))
            k += 1
        chi2 = chi2list
        Ndof = len(case_pos)-len(minuit_SEIR.np_errors())
        Prob = stats.chi2.sf(chi2, Ndof)
        d_res = {'Chi2': f"{chi2:3.1f}",
                 'Ndof': f"{Ndof}",
                 'Prob': f"{Prob:1.6f}"}
        add_text_to_ax(0.05, 0.95, nice_string_output(
            d_res, decimals=0), ax=ax_res[1], fontsize=9)
    fig_res.tight_layout()
    fig_res.show()
# Plot if any fits are performed:
# -------------------------------

# ----------------------------------------------------------------- #
# Fit to data with SIR and SIER model:
# ----------------------------------------------------------------- #

try:
    __IPYTHON__
except:
    input('Press Enter to Exit!')
