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


def load_simple_projection_model(addpath=''):
    """Read in the simple projection model

    The model is Gaussian and this function reads in the variance in the observed
    richness given the true richness.

    Parameters
    ----------
    addpath : str
        The function looks in a specific directory for the file. This parameter
        prepends path onto the default location.

    Returns
    -------
    lamlist : array_like
        A grid in true richness
    varlist : array_like
        The variance in observed richness given true richness on the grid given
        by lamlist
    """
    lamlist, varlist = np.loadtxt('{}inputs/p_lam_obs_z0.2.txt'.format(addpath),
                                  unpack=True, skiprows=1)
    return lamlist, varlist


def load_full_projection_model(addpath=''):
    """Read in the full projection model

    Parameters
    ----------
    addpath : str
        The function looks in a specific directory for the file. This parameter
        prepends path onto the default location.

    Returns
    -------
    projection_model : dict
        Dictionary containing lists to the 5 projection model parameters and true richness
        on a grid
    """
    input_model = np.loadtxt('{}projection_model/extended_projection_model.dat'.format(addpath))
    lam, fprj, fmsk, mu, sigma, tau = input_model
    return {'tau': tau, 'mu': mu, 'sigma': sigma, 'fprj': fprj, 'fmsk': fmsk, 'lam': lam}


def load_projection_model_interpolations(addpath=''):
    """Interpolate the full projection model

    Parameters
    ----------
    addpath : str
        The function looks in a specific directory for the file. This parameter
        prepends path onto the default location.

    Returns
    -------
    projection_model : dict
        Dictionary containing cubic splines for each of the 5 projection model parameters
    """
    lss_model = load_full_projection_model(addpath)

    fprj = interp1d(lss_model['lam'], lss_model['fprj'], kind='cubic')
    fmsk = interp1d(lss_model['lam'], lss_model['fmsk'], kind='cubic')
    mu = interp1d(lss_model['lam'], lss_model['mu'], kind='cubic')
    sigma = interp1d(lss_model['lam'], lss_model['sigma'], kind='cubic')
    tau = interp1d(lss_model['lam'], lss_model['tau'], kind='cubic')

    projection_model = {'tau': tau, 'mu': mu, 'sigma': sigma, 'fprj': fprj, 'fmsk': fmsk}
    return projection_model


def evaluate_full_projection_params(model, true_richness):
    """Given a true richness, determine the values of the projection model parameters

    Parameters
    ----------
    model : dict
        Dictionary of cubic splines for each of the projection model parameters
    true_richness : float
        The true richess that you wish to evaluate the projection model at

    Returns
    -------
    output : dict
        Dictionary of the model parameters evaluated at the requested true richness.
    """
    fprj = model['fprj'](true_richness)
    fmsk = model['fmsk'](true_richness)
    tau = model['tau'](true_richness)
    sigma = model['sigma'](true_richness)
    mu = model['mu'](true_richness)

    return {'fprj': fprj, 'fmsk': fmsk, 'mu': mu, 'sigma': sigma,
            'tau': tau, 'lamtrue': true_richness}


def evaluate_full_projection_model(lamobs, model):
    """Calculate P(observed_richness|true_richness)

    Parameters
    ----------
    lamobs : float
        Observed richness
    model : dict
        Model parameters and true richness at the true richness

    Returns
    -------
    output : float
        P(observed_richness|true_richness)
    """
    # Model params
    fprj, fmsk, lamtrue = model['fprj'], model['fmsk'], model['lamtrue']
    mu, sigma, tau = model['mu'], model['sigma'], model['tau']

    # Some stuff to make the calculation easier
    sig2 = sigma*sigma
    aa = np.exp(0.5*tau*(2.*mu + tau*sig2 - 2.*lamobs))
    root2siginv = 1./(np.sqrt(2.)*sigma)

    # The 4 terms in the model
    t1 = (1.-fmsk)*(1.-fprj)*np.exp(-(lamobs - mu)*(lamobs - mu)/(2*sig2))/np.sqrt(2.*np.pi*sig2)
    t2 = 0.5*((1.-fmsk)*fprj*tau + fmsk*fprj/lamtrue)*aa*erfc((mu + tau*sig2 - lamobs)*root2siginv)
    t3 = (fmsk*0.5/lamtrue)*(erfc((mu - lamobs - lamtrue)*root2siginv) 
                             - erfc((mu - lamobs)*root2siginv))
    t4 = (fmsk*fprj*0.5/lamtrue)*np.exp(-1.*tau*lamtrue)*aa*erfc((mu + tau*sig2 - lamtrue
                                                                 - lamobs)*root2siginv)
    return t1+t2+t3-t4


class SurveyInfo(object):
    """Hold all of the survey details

    Attributes
    ----------
    area : float
        Survey area in sq deg
    zmin : float
        Minimum survey redshift
    zmax : float
        Maximum survey redshift
    cosmo : dict
        Cosmological model
    c : float
        The speed of light
    survey_volume : float
        3d volume of survey in Mpc^3
    solid_angle : float
        Survey area in steradians
    """
    def __init__(self, cfgin, cosmology):
        self.area = cfgin['SurveyInfo'].getfloat('area')
        self.zmin = cfgin['SurveyInfo'].getfloat('zlo')
        self.zmax = cfgin['SurveyInfo'].getfloat('zhi')
        self.cosmo = cosmology
        self.c = 300000.
        self._calc_comoving_volume()

    def _calc_comoving_volume(self):
        survey_volume, solid_angle = calcComovingVolumeMpc3(self)
        self.survey_volume = survey_volume
        self.solid_angle = solid_angle

    def reset_survey_area(self, area):
        self.area = area
        self._calc_comoving_volume()


class Cluster(object):
    """Hold all of the details for a single galaxy cluster

    Attributes
    ----------
    lam : float
        Richness
    lamE : float
        Uncertainty in richness
    mgas : float
        Gas mass
    mgasE : float
        Uncertainty in gas mass
    lnm200m : float
        ln(M200m)
    lnm500c : float
        ln(M500c)
    """
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
    Vs = solid_angle*quad(volumeIntegrand, survey.zmin, survey.zmax, args=(survey), epsabs=1.49e-10, epsrel=1.49e-10)[0]
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


