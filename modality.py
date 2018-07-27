
# ###################################### #
#                                        #
# Updated by: Tom Pollard (2018.03.19)   #
# Author: Kerstin Johnsson               #
# License: MIT License                   #
# Available from:                        #
# https://github.com/kjohnsson/modality  #
#                                        #
# ###################################### #

import numpy as np 
from scipy.special import beta as betafun
# import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os
import pandas as pd
np.random.seed(1337)
# import pickle

def generate_data(peaks=2, n=None, mu=None, std=None):
    # Generate parameters if not provided
    if not n:
        n = [5000] * peaks
    if not mu:
        mu = np.random.randint(0,30,peaks)
    if not std:
        std = [1.0] * peaks
    # generate distributions then append
    dists = []
    for i in range(peaks):
        tmp = np.random.normal(loc=mu[i], scale=std[i], size=n[i])
        dists.append(tmp)
    data = np.concatenate(dists)
    return data

def hartigan_diptest(data):
    '''
        P-value according to Hartigan's dip test for unimodality.
        The dip is computed using the function
        dip_and_closest_unimodal_from_cdf. From this the p-value is
        interpolated using a table imported from the R package diptest.

        References:
            Hartigan and Hartigan (1985): The dip test of unimodality.
            The Annals of Statistics. 13(1).

        Input:
            data    -   one-dimensional data set.

        Value:
            p-value for the test.
    '''
    
    try:
      p = pval_hartigan(data[~np.isnan(data)])
    except:
      p = np.nan

    return p

def pval_hartigan(data):
    xF, yF = cum_distr(data)
    dip = dip_from_cdf(xF, yF)
    return dip_pval_tabinterpol(dip, len(data))

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

    # if qDiptab_df is None:
    #     raise DataError("Tabulated p-values not available. See installation instructions.")

    if np.isnan(N) or N < 10:
        return np.nan

    qDiptab_dict = {'0': {4: 0.125,
      5: 0.1,
      6: 0.0833333333333333,
      7: 0.0714285714285714,
      8: 0.0625,
      9: 0.0555555555555556,
      10: 0.05,
      15: 0.0341378172277919,
      20: 0.033718563622065004,
      30: 0.0262674485075642,
      50: 0.0218544781364545,
      100: 0.0164852597438403,
      200: 0.0111236388849688,
      500: 0.007554885975761959,
      1000: 0.00541658127872122,
      2000: 0.0039043999745055702,
      5000: 0.00245657785440433,
      10000: 0.00174954269199566,
      20000: 0.00119458814106091,
      40000: 0.000852415648011777,
      72000: 0.000644400053256997},
     '0.01': {4: 0.125,
      5: 0.1,
      6: 0.0833333333333333,
      7: 0.0714285714285714,
      8: 0.0625,
      9: 0.0613018090298924,
      10: 0.0610132555623269,
      15: 0.0546284208048975,
      20: 0.0474333740698401,
      30: 0.0395871890405749,
      50: 0.0314400501999916,
      100: 0.022831985803043,
      200: 0.0165017735429825,
      500: 0.0106403461127515,
      1000: 0.0076028674530018705,
      2000: 0.0054166418179658294,
      5000: 0.0034480928223332603,
      10000: 0.00244595133885302,
      20000: 0.00173435346896287,
      40000: 0.00122883479310665,
      72000: 0.000916872204484283},
     '0.02': {4: 0.125,
      5: 0.1,
      6: 0.0833333333333333,
      7: 0.0714285714285714,
      8: 0.0656911994503283,
      9: 0.0658615858179315,
      10: 0.0651627333214016,
      15: 0.0572191260231815,
      20: 0.0490891387627092,
      30: 0.0414574606741673,
      50: 0.0329008160470834,
      100: 0.0238917486442849,
      200: 0.0172594157992489,
      500: 0.0111255573208294,
      1000: 0.00794987834644799,
      2000: 0.0056617138625232296,
      5000: 0.00360473943713036,
      10000: 0.00255710802275612,
      20000: 0.0018119443458468102,
      40000: 0.0012846930445701802,
      72000: 0.0009579329467655321},
     '0.05': {4: 0.125,
      5: 0.1,
      6: 0.0833333333333333,
      7: 0.0725717816250742,
      8: 0.0738651136071762,
      9: 0.0732651142535317,
      10: 0.0718321619656165,
      15: 0.0610087367689692,
      20: 0.052719998201553,
      30: 0.0444462614069956,
      50: 0.0353023819040016,
      100: 0.0256559537977579,
      200: 0.0185259426032926,
      500: 0.0119353655328931,
      1000: 0.0085216518343594,
      2000: 0.00607120971135229,
      5000: 0.0038632654801084897,
      10000: 0.00273990955227265,
      20000: 0.00194259470485893,
      40000: 0.0013761765052555301,
      72000: 0.00102641863872347},
     '0.1': {4: 0.125,
      5: 0.1,
      6: 0.0833333333333333,
      7: 0.0817315478539489,
      8: 0.0820045917762512,
      9: 0.0803941629593475,
      10: 0.077966212182459,
      15: 0.0642657137330444,
      20: 0.0567795509056742,
      30: 0.0473998525042686,
      50: 0.0377279973102482,
      100: 0.0273987414570948,
      200: 0.0197917612637521,
      500: 0.0127411306411808,
      1000: 0.00909775605533253,
      2000: 0.0064762535755248,
      5000: 0.00412089506752692,
      10000: 0.0029225480567908,
      20000: 0.00207173719623868,
      40000: 0.0014675150200632301,
      72000: 0.0010949515421800199},
     '0.2': {4: 0.125,
      5: 0.1,
      6: 0.0924514470941933,
      7: 0.0940590181922527,
      8: 0.0922700601131892,
      9: 0.0890432420913848,
      10: 0.0852835359834564,
      15: 0.0692234107989591,
      20: 0.0620134674468181,
      30: 0.0516677370374349,
      50: 0.0410699984399582,
      100: 0.0298109370830153,
      200: 0.0215233745778454,
      500: 0.0138524542751814,
      1000: 0.00988924521014078,
      2000: 0.00703573098590029,
      5000: 0.00447640050137479,
      10000: 0.00317374638422465,
      20000: 0.00224993202086955,
      40000: 0.00159376453672466,
      72000: 0.00118904090369415},
     '0.3': {4: 0.125,
      5: 0.1,
      6: 0.103913431059949,
      7: 0.10324449080087102,
      8: 0.0996737189599363,
      9: 0.0950811420297928,
      10: 0.0903204173707099,
      15: 0.0745462114365167,
      20: 0.0660163872069048,
      30: 0.0551037519001622,
      50: 0.0437704598622665,
      100: 0.0317771496530253,
      200: 0.0229259769870428,
      500: 0.0147536004288476,
      1000: 0.0105309297090482,
      2000: 0.007494212545892991,
      5000: 0.00476555693102276,
      10000: 0.00338072258533527,
      20000: 0.00239520831473419,
      40000: 0.00169668445506151,
      72000: 0.00126575197699874},
     '0.4': {4: 0.125,
      5: 0.10872059357632902,
      6: 0.113885220640212,
      7: 0.110964599995697,
      8: 0.10573353180273701,
      9: 0.0999380897811046,
      10: 0.0943334983745117,
      15: 0.0792030878981762,
      20: 0.0696506075066401,
      30: 0.058265005347492994,
      50: 0.0462925642671299,
      100: 0.0336073821590387,
      200: 0.024243848341112,
      500: 0.0155963185751048,
      1000: 0.0111322726797384,
      2000: 0.007920878896017329,
      5000: 0.005037040297500721,
      10000: 0.0035724387653598205,
      20000: 0.00253036792824665,
      40000: 0.0017925341833790601,
      72000: 0.00133750966361506},
     '0.5': {4: 0.125,
      5: 0.12156379802641401,
      6: 0.123071187137781,
      7: 0.11780784650433501,
      8: 0.11103512984770501,
      9: 0.10415356007586801,
      10: 0.0977817630384725,
      15: 0.083621033469191,
      20: 0.0733437740592714,
      30: 0.0614510857304343,
      50: 0.048851155289608,
      100: 0.0354621760592113,
      200: 0.025584358256487003,
      500: 0.0164519238025286,
      1000: 0.0117439009052552,
      2000: 0.008355737247680059,
      5000: 0.0053123924740821294,
      10000: 0.00376734715752209,
      20000: 0.00266863168718114,
      40000: 0.00189061261635977,
      72000: 0.00141049709228472},
     '0.6': {4: 0.125,
      5: 0.134318918697053,
      6: 0.13186973390253,
      7: 0.124216086833531,
      8: 0.11592005574998801,
      9: 0.10800780236193198,
      10: 0.102180866696628,
      15: 0.0881198482202905,
      20: 0.0776460662880254,
      30: 0.0649164408053978,
      50: 0.0516145897865757,
      100: 0.0374805844550272,
      200: 0.0270252129816288,
      500: 0.017383057902553,
      1000: 0.012405033293814,
      2000: 0.00882439333812351,
      5000: 0.00560929919359959,
      10000: 0.00397885007249132,
      20000: 0.0028181999035216,
      40000: 0.00199645471886179,
      72000: 0.00148936709298802},
     '0.7': {4: 0.13255954878268902,
      5: 0.14729879897625198,
      6: 0.140564796497941,
      7: 0.130409013968317,
      8: 0.120561479262465,
      9: 0.112512617124951,
      10: 0.10996094814295099,
      15: 0.093124666680253,
      20: 0.0824558407118372,
      30: 0.0689178762425442,
      50: 0.0548121932066019,
      100: 0.0398046179116599,
      200: 0.0286920262150517,
      500: 0.0184503949887735,
      1000: 0.0131684179320803,
      2000: 0.009367858207170609,
      5000: 0.00595352728377949,
      10000: 0.00422430013176233,
      20000: 0.00299137548142077,
      40000: 0.00211929748381704,
      72000: 0.00158027541945626},
     '0.8': {4: 0.15749736904023498,
      5: 0.161085025702604,
      6: 0.14941924112913002,
      7: 0.136639642123068,
      8: 0.125558759034845,
      9: 0.12291503348081699,
      10: 0.11884476721158699,
      15: 0.0996694393390689,
      20: 0.08834462700173701,
      30: 0.0739249074078291,
      50: 0.0588230482851366,
      100: 0.0427283846799166,
      200: 0.0308006766341406,
      500: 0.0198162679782071,
      1000: 0.0141377942603047,
      2000: 0.01005604603884,
      5000: 0.00639092280563517,
      10000: 0.00453437508148542,
      20000: 0.00321024899920135,
      40000: 0.0022745769870358102,
      72000: 0.00169651643860074},
     '0.9': {4: 0.18740187880755899,
      5: 0.176811998476076,
      6: 0.159137064572627,
      7: 0.144240669035124,
      8: 0.141841067033899,
      9: 0.136412639387084,
      10: 0.130462149644819,
      15: 0.11008749690090598,
      20: 0.0972346018122903,
      30: 0.0814791379390127,
      50: 0.0649136324046767,
      100: 0.047152783315718,
      200: 0.0339967814293504,
      500: 0.0218781313182203,
      1000: 0.0156148055023058,
      2000: 0.0111019116837591,
      5000: 0.00705566126234625,
      10000: 0.00500178808402368,
      20000: 0.00354362220314155,
      40000: 0.00250999080890397,
      72000: 0.0018730618472582602},
     '0.95': {4: 0.20726978858735998,
      5: 0.18639179602794398,
      6: 0.164769608513302,
      7: 0.159903395678336,
      8: 0.153978303998561,
      9: 0.14660378495401902,
      10: 0.139611395137099,
      15: 0.118760769203664,
      20: 0.105130218270636,
      30: 0.0881689143126666,
      50: 0.0702737877191269,
      100: 0.0511279442868827,
      200: 0.0368418413878307,
      500: 0.0237294742633411,
      1000: 0.0169343970067564,
      2000: 0.0120380990328341,
      5000: 0.0076506368153935,
      10000: 0.00542372242836395,
      20000: 0.00384330190244679,
      40000: 0.00272375073486223,
      72000: 0.00203178401610555},
     '0.98': {4: 0.22375580462922195,
      5: 0.19361253363045,
      6: 0.17917654739278197,
      7: 0.17519655327122302,
      8: 0.16597856724751,
      9: 0.157084065653166,
      10: 0.150961728882481,
      15: 0.128890475210055,
      20: 0.11430970428125302,
      30: 0.0960564383013644,
      50: 0.0767095886079179,
      100: 0.0558022052195208,
      200: 0.0402729850316397,
      500: 0.025919578977657003,
      1000: 0.018513067368104,
      2000: 0.0131721010552576,
      5000: 0.00836821687047215,
      10000: 0.00592656681022859,
      20000: 0.00420258799378253,
      40000: 0.00298072958568387,
      72000: 0.00222356097506054},
     '0.99': {4: 0.231796258864192,
      5: 0.19650913979884502,
      6: 0.191862827995563,
      7: 0.184118659121501,
      8: 0.172988528276759,
      9: 0.164164643657217,
      10: 0.159684158858235,
      15: 0.13598356863636,
      20: 0.120624043335821,
      30: 0.101478558893837,
      50: 0.0811998415355918,
      100: 0.059024132304226,
      200: 0.0426864799777448,
      500: 0.0274518022761997,
      1000: 0.0196080260483234,
      2000: 0.0139655122281969,
      5000: 0.00886357892854914,
      10000: 0.00628034732880374,
      20000: 0.00445774902155711,
      40000: 0.00315942194040388,
      72000: 0.00235782814777627},
     '0.995': {4: 0.23726374382677898,
      5: 0.198159967287576,
      6: 0.20210197104296804,
      7: 0.19101439617430602,
      8: 0.179010413496374,
      9: 0.172821674582338,
      10: 0.16719524735674,
      15: 0.14245248368127697,
      20: 0.126552378036739,
      30: 0.10650487144103,
      50: 0.0852854646662134,
      100: 0.0620425065165146,
      200: 0.044958959158761,
      500: 0.0288986369564301,
      1000: 0.0206489568587364,
      2000: 0.0146889122204488,
      5000: 0.00934162787186159,
      10000: 0.00661030641550873,
      20000: 0.00469461513212743,
      40000: 0.0033273652798148,
      72000: 0.00248343580127067},
     '0.998': {4: 0.241992892688593,
      5: 0.19924427936243302,
      6: 0.213015781111186,
      7: 0.198216795232182,
      8: 0.186504388711178,
      9: 0.182555283567818,
      10: 0.175419540856082,
      15: 0.15017281653074202,
      20: 0.13360135382395,
      30: 0.112724636524262,
      50: 0.0904847827490294,
      100: 0.0658016011466099,
      200: 0.0477643873749449,
      500: 0.0306813505050163,
      1000: 0.0219285176765082,
      2000: 0.0156076779647454,
      5000: 0.009932186363240291,
      10000: 0.00702254699967648,
      20000: 0.004994160691291679,
      40000: 0.00353988965698579,
      72000: 0.00264210826339498},
     '0.999': {4: 0.244369839049632,
      5: 0.199617527406166,
      6: 0.219518627282415,
      7: 0.20234101074826102,
      8: 0.19448404115794,
      9: 0.188658833121906,
      10: 0.180611195797351,
      15: 0.15545613369632802,
      20: 0.138569903791767,
      30: 0.117164140184417,
      50: 0.0940930106666244,
      100: 0.0684479731118028,
      200: 0.0497198001867437,
      500: 0.0320170996823189,
      1000: 0.0228689168972669,
      2000: 0.0162685615996248,
      5000: 0.0103498795291629,
      10000: 0.0073182262815645795,
      20000: 0.00520917757743218,
      40000: 0.00369400045486625,
      72000: 0.0027524322157581},
     '0.9995': {4: 0.245966625504691,
      5: 0.19980094149902802,
      6: 0.22433904739444602,
      7: 0.205377566346832,
      8: 0.200864297005026,
      9: 0.19408912076824603,
      10: 0.18528641605039603,
      15: 0.160896499106958,
      20: 0.14336916123968,
      30: 0.12142585990898701,
      50: 0.0974904344916743,
      100: 0.0709169443994193,
      200: 0.0516114611801451,
      500: 0.0332452747332959,
      1000: 0.023738710122235003,
      2000: 0.0168874937789415,
      5000: 0.0107780907076862,
      10000: 0.0076065423418208,
      20000: 0.005403962359243721,
      40000: 0.00383345715372182,
      72000: 0.0028608570740143},
     '0.9998': {4: 0.24743959723326198,
      5: 0.19991708183427104,
      6: 0.22944933215424101,
      7: 0.208306562526874,
      8: 0.20884999705022897,
      9: 0.19915700809389003,
      10: 0.19120308390504398,
      15: 0.16697940794624802,
      20: 0.148940116394883,
      30: 0.126733051889401,
      50: 0.10228420428399698,
      100: 0.0741183486081263,
      200: 0.0540543978864652,
      500: 0.0348335698576168,
      1000: 0.0248334158891432,
      2000: 0.0176505093388153,
      5000: 0.0113184316868283,
      10000: 0.00795640367207482,
      20000: 0.00564540201704594,
      40000: 0.0040079346963469605,
      72000: 0.00298695044508003},
     '0.9999': {4: 0.24823065965663801,
      5: 0.19995902909307503,
      6: 0.232714530449602,
      7: 0.209866047852379,
      8: 0.212556040406219,
      9: 0.20288159843655804,
      10: 0.19580515933918397,
      15: 0.17111793515551002,
      20: 0.152832538183622,
      30: 0.131198578897542,
      50: 0.104680624334611,
      100: 0.0762579402903838,
      200: 0.0558704526182638,
      500: 0.0359832389317461,
      1000: 0.0256126573433596,
      2000: 0.0181944265400504,
      5000: 0.0117329446468571,
      10000: 0.0082270524584354,
      20000: 0.00580460792299214,
      40000: 0.00414892737222885,
      72000: 0.00309340092038059},
     '0.99995': {4: 0.248754269146416,
      5: 0.19997839537608197,
      6: 0.236548128358969,
      7: 0.21096757693345103,
      8: 0.21714917413729898,
      9: 0.205979795735129,
      10: 0.20029398089673,
      15: 0.17590050570443203,
      20: 0.15601016361897102,
      30: 0.133691739483444,
      50: 0.107496694235039,
      100: 0.0785735967934979,
      200: 0.0573877056330228,
      500: 0.0369051995840645,
      1000: 0.0265491336936829,
      2000: 0.0186226037818523,
      5000: 0.0119995948968375,
      10000: 0.00852240989786251,
      20000: 0.00599774739593151,
      40000: 0.0042839159079761,
      72000: 0.00319932767198801},
     '0.99998': {4: 0.24930203997425898,
      5: 0.199993151405815,
      6: 0.2390887911995,
      7: 0.212233348558702,
      8: 0.22170007640450304,
      9: 0.21054115498898,
      10: 0.20565108964621898,
      15: 0.18185667601316602,
      20: 0.16131922583934502,
      30: 0.137831637950694,
      50: 0.11140887547015,
      100: 0.0813458356889133,
      200: 0.0593365901653878,
      500: 0.0387221159256424,
      1000: 0.027578430100535997,
      2000: 0.0193001796565433,
      5000: 0.0124410052027886,
      10000: 0.00892863905540303,
      20000: 0.00633099254378114,
      40000: 0.0044187010443287895,
      72000: 0.00332688234611187},
     '0.99999': {4: 0.24945965232322498,
      5: 0.199995525025673,
      6: 0.24010356643629502,
      7: 0.21266103831250602,
      8: 0.225000835357532,
      9: 0.21180033095039003,
      10: 0.209682048785853,
      15: 0.185743454151004,
      20: 0.165568255916749,
      30: 0.14155750962435099,
      50: 0.113536607717411,
      100: 0.0832963013755522,
      200: 0.0607646310473911,
      500: 0.039930259057650005,
      1000: 0.0284430733108,
      2000: 0.0196241518040617,
      5000: 0.0129467396733128,
      10000: 0.009138539330002129,
      20000: 0.00656987109386762,
      40000: 0.00450818604569179,
      72000: 0.00339316094477355},
     '1': {4: 0.24974836247845,
      5: 0.199999835639211,
      6: 0.24467288361776798,
      7: 0.21353618608817,
      8: 0.23377291968768302,
      9: 0.21537991431762502,
      10: 0.221530282182963,
      15: 0.19224056333056197,
      20: 0.175834459522789,
      30: 0.163833046059817,
      50: 0.11788671686531199,
      100: 0.0926780423096737,
      200: 0.0705309107882395,
      500: 0.0431448163617178,
      1000: 0.0313640941982108,
      2000: 0.0213081254074584,
      5000: 0.014396063834027,
      10000: 0.00952234579566773,
      20000: 0.006858294480462271,
      40000: 0.00513477467565583,
      72000: 0.00376331697005859}}

    qDiptab_df = pd.DataFrame(qDiptab_dict)

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

    # if plotting:
    #     Nplot = 5
    #     bfig = plt.figure(figsize=(12, 3))
    #     i = 1  # plot index

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

        # # Plot current GCM and LCM.
        # if plotting:
        #     if i > Nplot:
        #         bfig = plt.figure(figsize=(12, 3))
        #         i = 1
        #     bax = bfig.add_subplot(1, Nplot, i)
        #     bax.plot(xF, yF, color='red')
        #     bax.plot(xF, yF-d/2, color='black')
        #     bax.plot(xF, yF+d/2, color='black')
        #     bax.plot(xF[iG], yF[iG]+d/2, color='blue')
        #     bax.plot(xF[iH], yF[iH]-d/2, color='blue')

        # if d <= D:
        #     if verbose:
        #         print("Difference in modal interval smaller than current dip")
        #     break

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

        # # Plot new modal interval
        # if plotting:
        #     ymin, ymax = bax.get_ylim()
        #     bax.axvline(xF[L0], ymin, ymax, color='orange')
        #     bax.axvline(xF[U0], ymin, ymax, color='red')
        #     bax.set_xlim(xF[L]-.1*(xF[U]-xF[L]), xF[U]+.1*(xF[U]-xF[L]))

        # Compute new lower bound for dip*2
        # i.e. largest difference outside modal interval
        gipl = np.interp(xF[L:(L0+1)], xF[iG], yF[iG])
        D = max(D, np.amax(yF[L:(L0+1)] - gipl))
        hipl = np.interp(xF[U0:(U+1)], xF[iH], yF[iH])
        D = max(D, np.amax(hipl - yF[U0:(U+1)]))

        if xF[U0]-xF[L0] < eps:
            if verbose:
                print("Modal interval zero length")
            break

        # if plotting:
        #     mxpt = np.argmax(yF[L:(L0+1)] - gipl)
        #     bax.plot([xF[L:][mxpt], xF[L:][mxpt]], [yF[L:][mxpt]+d/2, 
        #       gipl[mxpt]+d/2], '+', color='red')
        #     mxpt = np.argmax(hipl - yF[U0:(U+1)])
        #     bax.plot([xF[U0:][mxpt], xF[U0:][mxpt]], [yF[U0:][mxpt]-d/2, 
        #       hipl[mxpt]-d/2], '+', color='red')
        #     i += 1

        # Change modal interval
        L = L0
        U = U0

        if d <= D:
            if verbose:
                print("Difference in modal interval smaller than new dip")
            break

    # if plotting:

    #     # Add modal interval to figure
    #     bax.axvline(xF[L0], ymin, ymax, color='green', linestyle='dashed')
    #     bax.axvline(xF[U0], ymin, ymax, color='green', linestyle='dashed')

    #     ## Plot unimodal function (not distribution function)
    #     bfig = plt.figure()
    #     bax = bfig.add_subplot(1, 1, 1)
    #     bax.plot(xF, yF, color='red')
    #     bax.plot(xF, yF-D/2, color='black')
    #     bax.plot(xF, yF+D/2, color='black')

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

    # if plotting:

    #     bax.plot(xF[np.hstack([iGfin, iM_convex, iM_concave, iHfin])],
    #              np.hstack([yF[iGfin] + D/2, yF[iM_convex] + D/2,
    #                         yM_lower[iMM_concave], yF[iHfin] - D/2]), color='blue')
    #     #bax.plot(xF[iM], yM_lower, color='orange')

    #     ## Plot unimodal distribution function
    #     bfig = plt.figure()
    #     bax = bfig.add_subplot(1, 1, 1)
    #     bax.plot(xF, yF, color='red')
    #     bax.plot(xF, yF-D/2, color='black')
    #     bax.plot(xF, yF+D/2, color='black')

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
    k_start = (yU[1]-yU[0])/(xU[1]-xU[0]+1e-5)
    xU_start = xU[0] - yU[0]/(k_start+1e-5)
    k_end = (yU[-1]-yU[-2])/(xU[-1]-xU[-2]+1e-5)
    xU_end = xU[-1] + (1-yU[-1])/(k_end+1e-5)
    xU = np.hstack([xU_start, xU, xU_end])
    yU = np.hstack([0, yU, 1])

    # if plotting:
    #     bax.plot(xU, yU, color='blue')
    #     #bax.plot(xF[iM], yM_lower, color='orange')
    #     plt.show()

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
            print("x[icurr] = {}, x[icurr+1] = {}, x[icurr+2] = {}".format(x[icurr], 
              x[icurr+1], x[icurr+2]))
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

    # def plot(self, ax=None):
    #     x = self.h*np.linspace(np.min(self.datah)-5, np.max(self.datah)+5, 200)
    #     y = self.evaluate(x)
    #     if ax is None:
    #         fig, ax = plt.subplots()
    #     ax.plot(x, y)


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
    # data_dir = os.path.join('.', 'data')
    # print(data_dir)
    # with open(os.path.join(data_dir, 'gammaval.pkl'), 'r') as f:
    #     savedat = pickle.load(f)
    savedat = {'beta_betadistr': np.array([1.0,
                2.718281828459045,
                7.38905609893065,
                20.085536923187668,
                54.598150033144236,
                148.4131591025766,
                403.4287934927351,
                1096.6331584284585,
                2980.9579870417283,
                8103.083927575384,
                22026.465794806718,
                59874.14171519782,
                162754.79141900392,
                442413.3920089205,
                1202604.2841647768,
                3269017.3724721107,
                8886110.520507872,
                24154952.7535753,
                65659969.13733051,
                178482300.96318725]),
                'beta_studentt': np.array([0.5,
                1.3591409142295225,
                3.694528049465325,
                10.042768461593834,
                27.299075016572118,
                74.2065795512883,
                201.71439674636756,
                548.3165792142293,
                1490.4789935208642,
                4051.541963787692,
                11013.232897403359,
                29937.07085759891,
                81377.39570950196,
                221206.69600446025,
                601302.1420823884,
                1634508.6862360553,
                4443055.260253936,
                12077476.37678765,
                32829984.568665255,
                89241150.48159362]),
                'gamma_betadistr': np.array([0.0,
                4.3521604788918555,
                5.619663288128619,
                6.045132289787511,
                6.196412312629769,
                6.251371005194619,
                6.271496014102775,
                6.2788870215785195,
                6.281604322090273,
                6.282603731161307,
                6.282971362173459,
                6.283106602190213,
                6.283156350612787,
                6.283174653445515,
                6.2831813886918635,
                6.283183865648734,
                6.283184776870057,
                6.283185112089616,
                6.283185235410011,
                6.2831852807770385]),
                'gamma_studentt': np.array([np.inf,
                13.130440672051542,
                7.855693794235218,
                6.787835735957803,
                6.46039623388715,
                6.3473005818376755,
                6.306629302123698,
                6.291790708027913,
                6.286348471156239,
                6.284348620590986,
                6.283613218820035,
                6.283342721628752,
                6.283243215161844,
                6.28320661564662,
                6.283193150917383,
                6.283188190242287,
                6.283186367798792,
                6.28318569735954,
                6.283185450718775,
                6.283185359984703])}

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
