import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import configparser as cp
from colossus.halo.mass_defs import changeMassDefinition
from colossus.cosmology.cosmology import setCosmology
from colossus.halo.concentration import concentration
from scipy.special import erfc


def mass_conversion(m200m, redshift, cosmology, mass_is_log=True):
    """Convert m200m to m500c

    Parameters
    ----------
    m200m : array_like
        Halo mass(es) calculated in a radius such that the mean density is 200 times
        the mean density of the universe.
    redshift : float
        Redshift of the halos used to calculate the concentration and perform
        the mass conversion
    cosmology : dict
        Cosmology parameters being used
    mass_is_log : bool
        Flag to tell the script if it should intake and output ln(M) or M

    Returns
    -------
    output : array_like
        Halo mass(es) calculated in a radius such that the mean density is 500 times
        the critical density of the universe.

    Notes
    -----
    We assume that every halo is at the same redshift.
    """
    setCosmology('myCosmo', cosmology)

    if mass_is_log:
        m200m = np.exp(m200m)

    m500c = changeMassDefinition(m200m, concentration(m200m, '200m', redshift),
                                 redshift, '200m', '500c')[0]

    if mass_is_log:
        m500c = np.log(m500c)

    return m500c


# =============================================================================================================
# Load Matteo's lookuptable for Var(lamobs|lamtrue)
# Updated 3/1/2018
# =============================================================================================================
def load_richness_errorbar_lookuptable(addpath=''):
    lamlist, varlist = np.loadtxt('%sinputs/p_lam_obs_z0.2.txt' % addpath, unpack=True, skiprows=1)
    return lamlist, varlist


# =============================================================================================================
# Load the full projections model
# Updated 5/30/2018
# =============================================================================================================
def load_original_projection_model(cfgin, addpath=''):
    lam_true = np.array([1., 3., 5., 7., 9., 12., 15.55555534, 20., 24., 26.11111069, 30.,
                         36.66666412, 40., 47.22222137, 57.77777863, 68.33332825, 78.8888855,
                         89.44444275, 100., 120., 140., 160.])
    z_bins = np.linspace(0.10, 0.30, 5)

    # Find index to for zhmf
    idx = (z_bins == cfgin['SurveyInfo'].getfloat('zhmf'))

    # Load each of the lookup tables and find for appropriate z
    fit_lssmock = np.loadtxt('{}projection_model/prj_params_v9_41_lssmock.txt'.format(addpath))
    tau_prjmock_fit = np.reshape(fit_lssmock[0,:], (len(z_bins), len(lam_true)))[idx][0]
    mu_prjmock_fit = np.reshape(fit_lssmock[1,:], (len(z_bins), len(lam_true)))[idx][0]
    sig_prjmock_fit = np.reshape(fit_lssmock[2,:], (len(z_bins), len(lam_true)))[idx][0]
    fmask_prjmock_fit = np.reshape(fit_lssmock[3,:], (len(z_bins), len(lam_true)))[idx][0]
    fprj_prjmock_fit = np.reshape(fit_lssmock[4,:], (len(z_bins), len(lam_true)))[idx][0]

    projection_model = {'tau': tau_prjmock_fit, 'mu': mu_prjmock_fit, 'sigma': sig_prjmock_fit,
                        'fmsk': fmask_prjmock_fit, 'fprj': fprj_prjmock_fit, 'lam': lam_true}
    return projection_model


def load_projection_model(cfgin, addpath=''):
    lam, fprj, fmsk, mu, sigma, tau = np.loadtxt('{}projection_model/extended_projection_model.dat'.format(addpath), unpack=True)
    projection_model = {'tau': tau, 'mu': mu, 'sigma': sigma, 'fprj': fprj, 'fmsk': fmsk, 'lam': lam}
    return projection_model


def load_projection_model_interpolations(cfgin, addpath=''):
    lss_model = load_projection_model(cfgin, addpath)

    fprj = interp1d(lss_model['lam'], lss_model['fprj'], kind='cubic')
    fmsk = interp1d(lss_model['lam'], lss_model['fmsk'], kind='cubic')
    mu = interp1d(lss_model['lam'], lss_model['mu'], kind='cubic')
    sigma = interp1d(lss_model['lam'], lss_model['sigma'], kind='cubic')
    tau = interp1d(lss_model['lam'], lss_model['tau'], kind='cubic')

    projection_model = {'tau': tau, 'mu': mu, 'sigma': sigma, 'fprj': fprj, 'fmsk': fmsk}
    return projection_model


def calc_plamobslamtrue_on_grid(cfgin, addpath=''):
    lamobs_grid = np.linspace(2., 502., 101)
    lamtrue_grid = np.linspace(1, 501., 101)
    model = load_projection_model_interpolations(cfgin, addpath=addpath)

    table = []
    for lamtrue in lamtrue_grid:
        pmodel = calc_proj_model(model, lamtrue)
        for lamobs in lamobs_grid:
            table.append(p_lamobs_lamtrue(lamobs, pmodel))

    return lamobs_grid, table


def calc_proj_model(model, lamtrue):
    fprj = model['fprj'](lamtrue)
    fmsk = model['fmsk'](lamtrue)
    tau = model['tau'](lamtrue)
    sigma = model['sigma'](lamtrue)
    mu = model['mu'](lamtrue)

    projection_model = {'fprj': fprj, 'fmsk': fmsk, 'mu': mu, 'sigma': sigma,
                        'tau': tau, 'lamtrue': lamtrue}
    return projection_model


def p_lamobs_lamtrue(lamobs, model):
    # Model params
    fprj, fmsk, lamtrue = model['fprj'], model['fmsk'], model['lamtrue']
    mu, sigma, tau = model['mu'], model['sigma'], model['tau']

    # Some stuff to make the calculation easier
    sig2 = sigma*sigma
    A = np.exp(0.5*tau*(2.*mu + tau*sig2 - 2.*lamobs))
    root2siginv = 1./(np.sqrt(2.)*sigma)

    # The 4 terms in the model
    t1 = (1.-fmsk)*(1.-fprj)*np.exp(-(lamobs - mu)*(lamobs - mu)/(2*sig2))/np.sqrt(2.*np.pi*sig2)
    t2 = 0.5*((1.-fmsk)*fprj*tau + fmsk*fprj/lamtrue)*A*erfc((mu + tau*sig2 - lamobs)*root2siginv)
    t3 = (fmsk*0.5/lamtrue)*(erfc((mu - lamobs - lamtrue)*root2siginv) 
                             - erfc((mu - lamobs)*root2siginv))
    t4 = (fmsk*fprj*0.5/lamtrue)*np.exp(-1.*tau*lamtrue)*A*erfc((mu + tau*sig2 - lamtrue 
                                                                 - lamobs)*root2siginv)
    return t1+t2+t3-t4


# =============================================================================================================
# Build a class to hold all of the constants for the survey and cosmology info
# Updated 1/15/2018
# =============================================================================================================
class SurveyInfo(object):
    def __init__(self, cfg_in):
        self.area = cfg_in['SurveyInfo'].getfloat('area')
        self.zMin = cfg_in['SurveyInfo'].getfloat('zlo')
        self.zMax = cfg_in['SurveyInfo'].getfloat('zhi')
        self.OmM = cfg_in['Cosmology'].getfloat('omm')
        self.OmLam = cfg_in['Cosmology'].getfloat('oml')
        self.H0 = cfg_in['Cosmology'].getfloat('h0')

        self.Omb = cfg_in['Cosmology'].getfloat('omb')
        self.sigma8 = cfg_in['Cosmology'].getfloat('sigma8')
        self.ns = cfg_in['Cosmology'].getfloat('ns')

        self.c = 300000.

        self.calc_comoving_volume()

    def calc_comoving_volume(self):
        Vs, solid_angle = calcComovingVolumeMpc3(self)
        self.Vs = Vs
        self.solid_angle = solid_angle

    def reset_survey_area(self, area):
        self.area = area
        self.calc_comoving_volume()


# =============================================================================================================
# Class to hold the cluster information
# Updated: 1/15/2018
# =============================================================================================================
class Cluster(object):
    def __init__(self, richness, richnessError, mgas, mgasError, rescales):
        self.lam = richness/rescales['rich']
        self.lamE = richnessError/rescales['rich']
        self.mgas = mgas/rescales['mgas']
        self.mgasE = mgasError/rescales['mgas']

    def include_mass(self, lnm200m, lnm500c):
        self.lnm200m = lnm200m
        self.lnm500c = lnm500c

    def __lt__(self, other):
        return self.lam < other.lam

    def __repr__(self):
        return "%f %f %f %f\n" % (self.lam, self.lamE, self.mgas, self.mgasE)

    def printout(self):
        return [1.0, self.lam, self.lamE, self.mgas, self.mgasE]

    def printout_extra(self):
        return [1.0, self.lam, self.lamE, self.mgas, self.mgasE, self.lnm200m, self.lnm500c]


# =============================================================================================================
# Realization Class
# Holds all of the information of a sample of clusters
# Can make a list of these objects if multiple realizations are involved
# List with one element for real life
# Updated: 1/15/2018
# =============================================================================================================
class Realization:
    def __init__(self, cllist, lam_n, lam_n_obs):
        self.lam_n = lam_n
        self.lam_n_obs = lam_n_obs
        self.cllist = cllist


# =============================================================================================================
# Take in the HMF Block from Matteo, apply mass cuts and 
# format so that hmf[i] gives the hmf at the i-th redshift
# Updated 1/15/2018
# =============================================================================================================
def read_hmf_files(addpath, cfg_in, generation=0, gencut=-1.):
    log10M_cut = cfg_in['General'].getfloat('log10Mcut')
    zHMF = cfg_in['SurveyInfo'].getfloat('zhmf')
    cosmology = cfg_in['Cosmology']['cosmology']
    path = "{}inputs/generateHMF/hmf_{}_cosmology/mass_function/".format(addpath, cosmology)

    # Find the index of the HMF we want to work with
    zlist = np.loadtxt(path+'z.txt', skiprows=1)
    zind = np.where(abs(zlist-zHMF) < 0.00003)[0][0]
    print("Loading HMF at redshift %f" % zlist[zind])

    # Extract the appropriate HMF
    with open(path+'dndmdz.txt', 'r') as fin:
        lines = fin.readlines()[1:]
        hmf = [float(row.split(' ')[zind]) for row in lines]

    # Load masses and make mass cut
    mlist = np.loadtxt(path+'mass.txt', skiprows=1)
    if generation: 
        if gencut < 0.:
            raise ValueError("ERROR: Need to specify generation mass cut")
        mass_cut = np.where((mlist >= 10.**gencut) & (mlist <= 10.**16))
    else:
        mass_cut = np.where((mlist >= 10.**log10M_cut) & (mlist <= 10.**16))

    # Recast into dn/(dz dlnm)
    lnmlist = np.asarray([np.log(m) for m in mlist])
    dndlnmdz = np.asarray([m*dndmdz for m, dndmdz in zip(mlist, hmf)])

    return lnmlist[mass_cut], dndlnmdz[mass_cut]
    


# =============================================================================================================
# Functions relating to volume calculation
# Updated 1/15/2018
# =============================================================================================================
def comovIntegrand(z, survey):
    OmM, OmL, H0 = survey.OmM, survey.OmLam, survey.H0
    OmK = 1.0-OmL-OmM
    c = survey.c
    return c/(H0*np.sqrt(OmM*pow(1.+z, 3) + OmK*pow(1.+z, 2) + OmL))

def comov(z, survey):
    return quad(comovIntegrand, 0.0, z, args=(survey), epsabs=1.49e-10, epsrel=1.49e-10)[0]

def volumeIntegrand(z, survey):
    return (comov(z, survey))**2*comovIntegrand(z, survey)

def calcComovingVolumeMpc3(survey):
    solid_angle = survey.area*np.pi**2/(180.**2)
    Vs = solid_angle*quad(volumeIntegrand, survey.zMin, survey.zMax, args=(survey), epsabs=1.49e-10, epsrel=1.49e-10)[0]
    return Vs, solid_angle


# =============================================================================================================
# Functions to read in the clusters based on if I am using a mock or real data
# Updated 1/15/2018
# =============================================================================================================
def load_clusters(cfg_in, note, rescales, altpath=''):
    simtype = cfg_in['General']['simtype']
    catalog = cfg_in['General']['synth_catalog']
    Nclusters = cfg_in['SurveyInfo'].getint('Nclusters')

    if simtype == 'synth10':
        real_list = load_synth10(catalog, note, rescales)
        return real_list, 0.0

    # Build the filename based on the type of simulation
    if simtype == 'real':
        #cluster_fname = 'inputs/richestClustersInRedmapper_MantzXray.dat'
        cluster_fname = 'inputs/richestClustersInRedmapper_MantzXray_MatteoErrors.dat'
    elif len(altpath) > 5:
        cluster_fname = altpath
    elif simtype == 'synth':
        cluster_fname = './catalogs/mock/%s/richest%i_%i.dat' % (catalog, Nclusters, note)
        
    # Actually load the clusters
    cllist, lnpivot, lamN, lamN_obs = load_clusters_from_file(cluster_fname, rescales)
    real_list = [Realization(cllist, lamN, lamN_obs)]

    return real_list, lnpivot


# 1/15/2018
def load_clusters_from_file(fname, rescales):
    # Load the clusters
    lamlist, lamElist, mglist, mgElist = np.loadtxt(fname, skiprows=1, usecols=[1,2,3,4], unpack=True)
    cllist = [Cluster(lamlist[i], lamElist[i], mglist[i], mgElist[i], rescales) for i in range(len(lamlist))]

    # Find the median pivot
    lnpivot = np.log(np.median(mglist)/(0.3))

    # Find the least rich cluster
    lamN, idx = min((lamN, idx) for (idx, lamN) in enumerate(lamlist))
    lamN_obs = lamElist[idx]

    print('%i Clusters loaded.' % (len(cllist)))
    return cllist, lnpivot, lamN, lamN_obs


# 1/15/2018
def load_synth10(catalog, note, rescales):
    real_list = []
    for kk in range(10):
        cluster_fname = './catalogs/mock/%s/richest30_%i.dat' % (catalog, note*10+k)
        print("Read cluster", cluster_fname)
        cllist, lnpivot, lamN, lamN_obs = load_clusters_from_file(cluster_fname, rescales)
        real_list.append(Realization(cllist, lamN, lamN_obs))
    return real_list

# =============================================================================================================



# =============================================================================================================
# Class to hold the prior information
# Update: 1/15/2018
# =============================================================================================================
class PriorContainer(object):
    def __init__(self, cfg_in): 
        #fname = '~/cosmosis/modules/xray-likelihood/priors/'+cfg_in['MCMC']['priorfile']
        fname = '/home/matthewkirby/cosmosis/modules/xray-likelihood/priors/'+cfg_in['MCMC']['priorfile']
        priorsin = cp.ConfigParser()
        priorsin.read(fname)

        self.mg0 = priorsin['MgasPriors'].getfloat('mg0')
        self.mg0_uncert = priorsin['MgasPriors'].getfloat('mg0e')
        self.alphamg = priorsin['MgasPriors'].getfloat('alphamg')
        self.alphamg_uncert = priorsin['MgasPriors'].getfloat('alphamge')

        self.amp_mr_rel = priorsin['RichnessPriors'].getfloat('as')
        self.amp_uncert_mr_rel = priorsin['RichnessPriors'].getfloat('ase')
        self.slope_mr_rel = priorsin['RichnessPriors'].getfloat('gamma')
        self.slope_uncert_mr_rel = priorsin['RichnessPriors'].getfloat('gammae')

        self.quant = priorsin['RichnessPriors'].getfloat('quant')
        self.quant_uncert = priorsin['RichnessPriors'].getfloat('quante')
        self.alphalam = priorsin['RichnessPriors'].getfloat('alphalam')
        self.alphalam_uncert = priorsin['RichnessPriors'].getfloat('alphalame')

    def __repr__(self):
        return "mg0 %f +- %f\nalphamg %f +- %f\nAs %f +- %f\ngamma %f +- %f" % \
            (self.mg0, self.mg0_uncert, self.alphamg, self.alphamg_uncert, self.amp_mr_rel,
             self.amp_uncert_mr_rel, self.slope_mr_rel, self.slope_uncert_mr_rel)


# =============================================================================================================
# Write the names of the output files
# Update: 1/15/2018
# =============================================================================================================
def build_output_files(cfg_in, note):
    simtype = cfg_in['General']['simtype']
    prior = cfg_in['MCMC']['prior']
    if simtype == 'synth' or simtype == 'synth10':
        catalog = cfg_in['General']['synth_catalog']
        outChain = './outputs/%s/chains/mcmc%i_%sprior_richest%i.out' % (catalog, note, prior, cfg_in['SurveyInfo'].getint('nclusters'))
        outLike = './outputs/%s/chains/lnL%i_%spriors.out' % (catalog, note, prior)

    elif simtype == 'real':
        outChain = './outputs/real_runs/chains/mcmc_%spriors.out' % (prior)
        outLike = './outputs/real_runs/chains/lnL_%spriors.out' % (prior)

    else:
        print("Simulation type %s is an invalid type." % (simtype))

    return outChain, outLike



# =============================================================================================================
# Finalize the pivots that will be used
# Updated: 1/15/2018
# =============================================================================================================
def select_pivots(cfg_in, lnpivot):
    lam_pivot = cfg_in['General'].getfloat('rich_pivot')
    mgas_pivot = cfg_in['General'].getfloat('mgas_pivot')
    pivots = {'rich': 0.0, 'mgas': 0.0}

    if abs(lam_pivot+1.) <= 0.003:
        pivots['rich'] = lnpivot
    else:
        pivots['rich'] = np.log(lam_pivot)
    if abs(mgas_pivot+1.) <= 0.003:
        pivots['mgas'] = lnpivot
    else:
        pivots['mgas'] = np.log(mgas_pivot)

    return pivots


# =============================================================================================================
# Load truth from file
# Updated: 1/31/2018
# =============================================================================================================
def load_truth(fname):
    model = cp.ConfigParser()
    model.read(fname)
    params = [ model['Model'].getfloat('r'),
               model['Model'].getfloat('mgas0'),
               model['Model'].getfloat('alphamg'),
               model['Model'].getfloat('s0mg'),
               model['Model'].getfloat('lam0'),
               model['Model'].getfloat('alphalam'),
               model['Model'].getfloat('s0lam'),
               model['Model'].getfloat('dhmf')]


    return params



# =============================================================================================================
# Find the max lamobs that is used to generate the mocks as a function of lamtrue
# =============================================================================================================
def find_mock_lamobs_max(simtype, addpath=''):
    '''Load the CDFs that describe the projection model'''
    lamtrue_grid = np.loadtxt('{}projection_model/cdfs/lamtrue.dat'.format(addpath))
    lamobs_grid = np.loadtxt('{}projection_model/cdfs/lamobs.dat'.format(addpath))
    cdf_grid = np.loadtxt('{}projection_model/cdfs/cdflist.dat'.format(addpath))
    maxlamobs = []

    if simtype == 'real':
        return [lamtrue_grid, np.ones(len(lamtrue_grid))*1999.]

    print("FINDING MAX LAMOBS FOR INTEGRATION BOUNDS. MOCK ONLY!!")
    for i in range(len(lamtrue_grid)):
        cdf = cdf_grid[i]
        idxs = np.array(np.where(cdf < 1.0)[0])
        cdf_cut = cdf[idxs]
        lamobs_grid_cut = lamobs_grid[idxs]
        maxlamobs.append(lamobs_grid_cut[-1])
        
    return [lamtrue_grid, maxlamobs]


