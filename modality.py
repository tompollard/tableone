
# ###################################### #
#                                        #
# Author: Kerstin Johnsson               #
# License: MIT License                   #
# Available from:                        #
# https://github.com/kjohnsson/modality  #
#                                        #
# ###################################### #

import numpy as np 
from scipy.special import beta as betafun
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os
import pickle

def generate_data(peaks=2, n=[1000,1000], mu=[1.0,5.0], 
    std=[1.0,1.0]):
    # generate distributions then append
    dists = []
    for i in range(peaks):
        tmp = np.random.normal(loc=mu[i], scale=std[i], size=n[i])
        dists.append(tmp)
    data = np.concatenate(dists)
    return data

def cum_distr(data, w=None):
    if w is None:
        w = np.ones(len(data))*1./len(data)
    eps = 1e-10
    data_ord = np.argsort(data)
    data_sort = data[data_ord]
    w_sort = w[data_ord]
    data_sort, indices = unique(data_sort, return_index=True, eps=eps, is_sorted=True)
    if len(indices) < len(data_ord):
        w_unique = np.zeros(len(indices))
        for i in range(len(indices)-1):
            w_unique[i] = np.sum(w_sort[indices[i]:indices[i+1]])
        w_unique[-1] = np.sum(w_sort[indices[-1]:])
        w_sort = w_unique
    wcum = np.cumsum(w_sort)
    wcum /= wcum[-1]

    N = len(data_sort)
    x = np.empty(2*N)
    x[2*np.arange(N)] = data_sort
    x[2*np.arange(N)+1] = data_sort
    y = np.empty(2*N)
    y[0] = 0
    y[2*np.arange(N)+1] = wcum
    y[2*np.arange(N-1)+2] = wcum[:-1]
    return x, y

def unique(data, return_index, eps, is_sorted=True):
    if not is_sorted:
        ord = np.argsort(data)
        rank = np.argsort(ord)
        data_sort = data[ord]
    else:
        data_sort = data
    isunique_sort = np.ones(len(data_sort), dtype='bool')
    j = 0
    for i in range(1, len(data_sort)):
        if data_sort[i] - data_sort[j] < eps:
            isunique_sort[i] = False
        else:
            j = i
    if not is_sorted:
        isunique = isunique_sort[rank]
        data_unique = data[isunique]
    else:
        data_unique = data[isunique_sort]

    if not return_index:
        return data_unique

    if not is_sorted:
        ind_unique = np.nonzero(isunique)[0]
    else:
        ind_unique = np.nonzero(isunique_sort)[0]
    return data_unique, ind_unique

def dip_from_cdf(xF, yF, plotting=False, verbose=False, eps=1e-12):
    dip, _ = dip_and_closest_unimodal_from_cdf(xF, yF, plotting, verbose, eps)
    return dip

def dip_pval_tabinterpol(dip, N):
    '''
        dip     -   dip value computed from dip_from_cdf
        N       -   number of observations
    '''

    if qDiptab_df is None:
        raise DataError("Tabulated p-values not available. See installation instructions.")

    if np.isnan(N) or N < 10:
        return np.nan

    diptable = np.array(qDiptab_df)
    ps = np.array(qDiptab_df.columns).astype(float)
    Ns = np.array(qDiptab_df.index)

    if N >= Ns[-1]:
        dip = transform_dip_to_other_nbr_pts(dip, N, Ns[-1]-0.1)
        N = Ns[-1]-0.1

    iNlow = np.nonzero(Ns < N)[0][-1]
    qN = (N-Ns[iNlow])/(Ns[iNlow+1]-Ns[iNlow])
    dip_sqrtN = np.sqrt(N)*dip
    dip_interpol_sqrtN = (
        np.sqrt(Ns[iNlow])*diptable[iNlow, :] + qN*(
            np.sqrt(Ns[iNlow+1])*diptable[iNlow+1, :]-np.sqrt(Ns[iNlow])*diptable[iNlow, :]))

    if not (dip_interpol_sqrtN < dip_sqrtN).any():
        return 1

    iplow = np.nonzero(dip_interpol_sqrtN < dip_sqrtN)[0][-1]
    if iplow == len(dip_interpol_sqrtN) - 1:
        return 0

    qp = (dip_sqrtN-dip_interpol_sqrtN[iplow])/(dip_interpol_sqrtN[iplow+1]-dip_interpol_sqrtN[iplow])
    p_interpol = ps[iplow] + qp*(ps[iplow+1]-ps[iplow])

    return 1 - p_interpol

def transform_dip_to_other_nbr_pts(dip_n, n, m):
    dip_m = np.sqrt(n/m)*dip_n
    return dip_m

def dip_and_closest_unimodal_from_cdf(xF, yF, plotting=False, verbose=False, eps=1e-12):
    '''
        Dip computed as distance between empirical distribution function (EDF) and
        cumulative distribution function for the unimodal distribution with
        smallest such distance. The optimal unimodal distribution is found by
        the algorithm presented in

            Hartigan (1985): Computation of the dip statistic to test for
            unimodaliy. Applied Statistics, vol. 34, no. 3

        If the plotting option is enabled the optimal unimodal distribution
        function is plotted along with (xF, yF-dip) and (xF, yF+dip)

        xF  -   x-coordinates for EDF
        yF  -   y-coordinates for EDF

    '''

    ## TODO! Preprocess xF and yF so that yF increasing and xF does
    ## not have more than two copies of each x-value.

    if (xF[1:]-xF[:-1] < -eps).any():
        raise ValueError('Need sorted x-values to compute dip')
    if (yF[1:]-yF[:-1] < -eps).any():
        raise ValueError('Need sorted y-values to compute dip')

    if plotting:
        Nplot = 5
        bfig = plt.figure(figsize=(12, 3))
        i = 1  # plot index

    D = 0  # lower bound for dip*2

    # [L, U] is interval where we still need to find unimodal function,
    # the modal interval
    L = 0
    U = len(xF) - 1

    # iGfin are the indices of xF where the optimal unimodal distribution is greatest
    # convex minorant to (xF, yF+dip)
    # iHfin are the indices of xF where the optimal unimodal distribution is least
    # concave majorant to (xF, yF-dip)
    iGfin = L
    iHfin = U

    while 1:

        iGG = greatest_convex_minorant_sorted(xF[L:(U+1)], yF[L:(U+1)])
        iHH = least_concave_majorant_sorted(xF[L:(U+1)], yF[L:(U+1)])
        iG = np.arange(L, U+1)[iGG]
        iH = np.arange(L, U+1)[iHH]

        # Interpolate. First and last point are in both and does not need
        # interpolation. Might cause trouble if included due to possiblity
        # of infinity slope at beginning or end of interval.
        if iG[0] != iH[0] or iG[-1] != iH[-1]:
            raise ValueError('Convex minorant and concave majorant should start and end at same points.')
        hipl = np.interp(xF[iG[1:-1]], xF[iH], yF[iH])
        gipl = np.interp(xF[iH[1:-1]], xF[iG], yF[iG])
        hipl = np.hstack([yF[iH[0]], hipl, yF[iH[-1]]])
        gipl = np.hstack([yF[iG[0]], gipl, yF[iG[-1]]])
        #hipl = lin_interpol_sorted(xF[iG], xF[iH], yF[iH])
        #gipl = lin_interpol_sorted(xF[iH], xF[iG], yF[iG])

        # Find largest difference between GCM and LCM.
        gdiff = hipl - yF[iG]
        hdiff = yF[iH] - gipl
        imaxdiffg = np.argmax(gdiff)
        imaxdiffh = np.argmax(hdiff)
        d = max(gdiff[imaxdiffg], hdiff[imaxdiffh])

        # Plot current GCM and LCM.
        if plotting:
            if i > Nplot:
                bfig = plt.figure(figsize=(12, 3))
                i = 1
            bax = bfig.add_subplot(1, Nplot, i)
            bax.plot(xF, yF, color='red')
            bax.plot(xF, yF-d/2, color='black')
            bax.plot(xF, yF+d/2, color='black')
            bax.plot(xF[iG], yF[iG]+d/2, color='blue')
            bax.plot(xF[iH], yF[iH]-d/2, color='blue')

        if d <= D:
            if verbose:
                print "Difference in modal interval smaller than current dip"
            break

        # Find new modal interval so that largest difference is at endpoint
        # and set d to largest distance between current GCM and LCM.
        if gdiff[imaxdiffg] > hdiff[imaxdiffh]:
            L0 = iG[imaxdiffg]
            U0 = iH[iH >= L0][0]
        else:
            U0 = iH[imaxdiffh]
            L0 = iG[iG <= U0][-1]
        # Add points outside the modal interval to the final GCM and LCM.
        iGfin = np.hstack([iGfin, iG[(iG <= L0)*(iG > L)]])
        iHfin = np.hstack([iH[(iH >= U0)*(iH < U)], iHfin])

        # Plot new modal interval
        if plotting:
            ymin, ymax = bax.get_ylim()
            bax.axvline(xF[L0], ymin, ymax, color='orange')
            bax.axvline(xF[U0], ymin, ymax, color='red')
            bax.set_xlim(xF[L]-.1*(xF[U]-xF[L]), xF[U]+.1*(xF[U]-xF[L]))

        # Compute new lower bound for dip*2
        # i.e. largest difference outside modal interval
        gipl = np.interp(xF[L:(L0+1)], xF[iG], yF[iG])
        D = max(D, np.amax(yF[L:(L0+1)] - gipl))
        hipl = np.interp(xF[U0:(U+1)], xF[iH], yF[iH])
        D = max(D, np.amax(hipl - yF[U0:(U+1)]))

        if xF[U0]-xF[L0] < eps:
            if verbose:
                print "Modal interval zero length"
            break

        if plotting:
            mxpt = np.argmax(yF[L:(L0+1)] - gipl)
            bax.plot([xF[L:][mxpt], xF[L:][mxpt]], [yF[L:][mxpt]+d/2, gipl[mxpt]+d/2], '+', color='red')
            mxpt = np.argmax(hipl - yF[U0:(U+1)])
            bax.plot([xF[U0:][mxpt], xF[U0:][mxpt]], [yF[U0:][mxpt]-d/2, hipl[mxpt]-d/2], '+', color='red')
            i += 1

        # Change modal interval
        L = L0
        U = U0

        if d <= D:
            if verbose:
                print "Difference in modal interval smaller than new dip"
            break

    if plotting:

        # Add modal interval to figure
        bax.axvline(xF[L0], ymin, ymax, color='green', linestyle='dashed')
        bax.axvline(xF[U0], ymin, ymax, color='green', linestyle='dashed')

        ## Plot unimodal function (not distribution function)
        bfig = plt.figure()
        bax = bfig.add_subplot(1, 1, 1)
        bax.plot(xF, yF, color='red')
        bax.plot(xF, yF-D/2, color='black')
        bax.plot(xF, yF+D/2, color='black')

    # Find string position in modal interval
    iM = np.arange(iGfin[-1], iHfin[0]+1)
    yM_lower = yF[iM]-D/2
    yM_lower[0] = yF[iM[0]]+D/2
    iMM_concave = least_concave_majorant_sorted(xF[iM], yM_lower)
    iM_concave = iM[iMM_concave]
    #bax.plot(xF[iM], yM_lower, color='orange')
    #bax.plot(xF[iM_concave], yM_lower[iMM_concave], color='red')
    lcm_ipl = np.interp(xF[iM], xF[iM_concave], yM_lower[iMM_concave])
    try:
        mode = iM[np.nonzero(lcm_ipl > yF[iM]+D/2)[0][-1]]
        #bax.axvline(xF[mode], color='green', linestyle='dashed')
    except IndexError:
        iM_convex = np.zeros(0, dtype='i')
    else:
        after_mode = iM_concave > mode
        iM_concave = iM_concave[after_mode]
        iMM_concave = iMM_concave[after_mode]
        iM = iM[iM <= mode]
        iM_convex = iM[greatest_convex_minorant_sorted(xF[iM], yF[iM])]

    if plotting:

        bax.plot(xF[np.hstack([iGfin, iM_convex, iM_concave, iHfin])],
                 np.hstack([yF[iGfin] + D/2, yF[iM_convex] + D/2,
                            yM_lower[iMM_concave], yF[iHfin] - D/2]), color='blue')
        #bax.plot(xF[iM], yM_lower, color='orange')

        ## Plot unimodal distribution function
        bfig = plt.figure()
        bax = bfig.add_subplot(1, 1, 1)
        bax.plot(xF, yF, color='red')
        bax.plot(xF, yF-D/2, color='black')
        bax.plot(xF, yF+D/2, color='black')

    # Find string position in modal interval
    iM = np.arange(iGfin[-1], iHfin[0]+1)
    yM_lower = yF[iM]-D/2
    yM_lower[0] = yF[iM[0]]+D/2
    iMM_concave = least_concave_majorant_sorted(xF[iM], yM_lower)
    iM_concave = iM[iMM_concave]
    #bax.plot(xF[iM], yM_lower, color='orange')
    #bax.plot(xF[iM_concave], yM_lower[iMM_concave], color='red')
    lcm_ipl = np.interp(xF[iM], xF[iM_concave], yM_lower[iMM_concave])
    try:
        mode = iM[np.nonzero(lcm_ipl > yF[iM]+D/2)[0][-1]]
        #bax.axvline(xF[mode], color='green', linestyle='dashed')
    except IndexError:
        iM_convex = np.zeros(0, dtype='i')
    else:
        after_mode = iM_concave > mode
        iM_concave = iM_concave[after_mode]
        iMM_concave = iMM_concave[after_mode]
        iM = iM[iM <= mode]
        iM_convex = iM[greatest_convex_minorant_sorted(xF[iM], yF[iM])]

    # Closest unimodal curve
    xU = xF[np.hstack([iGfin[:-1], iM_convex, iM_concave, iHfin[1:]])]
    yU = np.hstack([yF[iGfin[:-1]] + D/2, yF[iM_convex] + D/2,
                    yM_lower[iMM_concave], yF[iHfin[1:]] - D/2])
    # Add points so unimodal curve goes from 0 to 1
    k_start = (yU[1]-yU[0])/(xU[1]-xU[0])
    xU_start = xU[0] - yU[0]/k_start
    k_end = (yU[-1]-yU[-2])/(xU[-1]-xU[-2])
    xU_end = xU[-1] + (1-yU[-1])/k_end
    xU = np.hstack([xU_start, xU, xU_end])
    yU = np.hstack([0, yU, 1])

    if plotting:
        bax.plot(xU, yU, color='blue')
        #bax.plot(xF[iM], yM_lower, color='orange')
        plt.show()

    return D/2, (xU, yU)

def greatest_convex_minorant_sorted(x, y):
    i = least_concave_majorant_sorted(x, -y)
    return i

def least_concave_majorant_sorted(x, y, eps=1e-12):
    i = [0]
    icurr = 0
    while icurr < len(x) - 1:
        if np.abs(x[icurr+1]-x[icurr]) > eps:
            q = (y[(icurr+1):]-y[icurr])/(x[(icurr+1):]-x[icurr])
            icurr += 1 + np.argmax(q)
            i.append(icurr)
        elif y[icurr+1] > y[icurr] or icurr == len(x)-2:
            icurr += 1
            i.append(icurr)
        elif np.abs(x[icurr+2]-x[icurr]) > eps:
            q = (y[(icurr+2):]-y[icurr])/(x[(icurr+2):]-x[icurr])
            icurr += 2 + np.argmax(q)
            i.append(icurr)
        else:
            print "x[icurr] = {}, x[icurr+1] = {}, x[icurr+2] = {}".format(x[icurr], x[icurr+1], x[icurr+2])
            raise ValueError('Maximum two copies of each x-value allowed')

    return np.array(i)

class KernelDensityDerivative(object):

    def __init__(self, data, deriv_order):

        if deriv_order == 0:
            self.kernel = lambda u: np.exp(-u**2/2)
        elif deriv_order == 2:
            self.kernel = lambda u: (u**2-1)*np.exp(-u**2/2)
        else:
            raise ValueError('Not implemented for derivative of order {}'.format(deriv_order))
        self.deriv_order = deriv_order
        self.h = silverman_bandwidth(data, deriv_order)
        self.datah = data/self.h

    def evaluate(self, x):
        xh = np.array(x).reshape(-1)/self.h
        res = np.zeros(len(xh))
        if len(xh) > len(self.datah):  # loop over data
            for data_ in self.datah:
                res += self.kernel(data_-xh)
        else:  # loop over x
            for i, x_ in enumerate(xh):
                res[i] = np.sum(self.kernel(self.datah-x_))
        return res*1./(np.sqrt(2*np.pi)*self.h**(1+self.deriv_order)*len(self.datah))

    def score_samples(self, x):
        return self.evaluate(x)

    def plot(self, ax=None):
        x = self.h*np.linspace(np.min(self.datah)-5, np.max(self.datah)+5, 200)
        y = self.evaluate(x)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x, y)


def silverman_bandwidth(data, deriv_order=0):
    sigmahat = np.std(data, ddof=1)
    return sigmahat*bandwidth_factor(data.shape[0], deriv_order)


def bandwidth_factor(nbr_data_pts, deriv_order=0):
    '''
        Scale factor for one-dimensional plug-in bandwidth selection.
    '''
    if deriv_order == 0:
        return (3.0*nbr_data_pts/4)**(-1.0/5)

    if deriv_order == 2:
        return (7.0*nbr_data_pts/4)**(-1.0/9)

    raise ValueError('Not implemented for derivative of order {}'.format(deriv_order))

def calibrated_dip_test(data, N_bootstrap=1000):
    xF, yF = cum_distr(data)
    dip = dip_from_cdf(xF, yF)
    n_eval = 512
    f_hat = KernelDensityDerivative(data, 0)
    f_bis_hat = KernelDensityDerivative(data, 2)
    x = np.linspace(np.min(data), np.max(data), n_eval)
    f_hat_eval = f_hat.evaluate(x)
    ind_x0_hat = np.argmax(f_hat_eval)
    d_hat = np.abs(f_bis_hat.evaluate(x[ind_x0_hat]))/f_hat_eval[ind_x0_hat]**3
    ref_distr = select_calibration_distribution(d_hat)
    ref_dips = np.zeros(N_bootstrap)
    for i in range(N_bootstrap):
        samp = ref_distr.sample(len(data))
        xF, yF = cum_distr(samp)
        ref_dips[i] = dip_from_cdf(xF, yF)
    return np.mean(ref_dips > dip)


def select_calibration_distribution(d_hat):
    data_dir = os.path.join('.', 'data')
    print(data_dir)
    with open(os.path.join(data_dir, 'gammaval.pkl'), 'r') as f:
        savedat = pickle.load(f)

    if np.abs(d_hat-np.pi) < 1e-4:
        return RefGaussian()
    if d_hat < 2*np.pi:  # beta distribution
        gamma = lambda beta: 2*(beta-1)*betafun(beta, 1.0/2)**2 - d_hat
        i = np.searchsorted(savedat['gamma_betadistr'], d_hat)
        beta_left = savedat['beta_betadistr'][i-1]
        beta_right = savedat['beta_betadistr'][i]
        beta = brentq(gamma, beta_left, beta_right)
        return RefBeta(beta)

    # student t distribution
    gamma = lambda beta: 2*beta*betafun(beta-1./2, 1./2)**2 - d_hat
    i = np.searchsorted(-savedat['gamma_studentt'], -d_hat)
    beta_left = savedat['beta_studentt'][i-1]
    beta_right = savedat['beta_studentt'][i]
    beta = brentq(gamma, beta_left, beta_right)
    return RefStudentt(beta)

class RefGaussian(object):
    def sample(self, n):
        return np.random.randn(n)

class RefBeta(object):
    def __init__(self, beta):
        self.beta = beta

    def sample(self, n):
        return np.random.beta(self.beta, self.beta, n)

class RefStudentt(object):
    def __init__(self, beta):
        self.beta = beta

    def sample(self, n):
        dof = 2*self.beta-1
        return 1./np.sqrt(dof)*np.random.standard_t(dof, n)