import numpy as np
import os
from scipy.interpolate import interp1d
import configparser as cp
from colossus.halo.mass_defs import changeMassDefinition
from colossus.cosmology.cosmology import setCosmology
from colossus.halo.concentration import concentration
from scipy.special import erfc
from astropy.cosmology import FlatLambdaCDM


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
        self.solid_angle = self.area*np.pi**2/(180.**2)
        self._calc_comoving_volume()

    def _calc_comoving_volume(self):
        cosmo = FlatLambdaCDM(H0=self.cosmo['H0'], Om0=self.cosmo['Om0'])
        fullsky_volume = cosmo.comoving_volume(self.zmax)-cosmo.comoving_volume(self.zmin)
        survey_volume = fullsky_volume.value*(self.solid_angle/(4.*np.pi))
        self.survey_volume = survey_volume

    def reset_survey_area(self, area):
        self.area = area
        self.solid_angle = self.area*np.pi**2/(180.**2)
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
    def __init__(self, richness, richness_error, mgas, mgas_error, rescales):
        self.lam = richness/rescales['rich']
        self.lamE = richness_error/rescales['rich']
        self.mgas = mgas/rescales['mgas']
        self.mgasE = mgas_error/rescales['mgas']

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


class Realization:
    """Hold a single survey realization.

    The intended use here is to allow the analysis to be run on several survey
    realizations at once.

    Attributes
    ----------
    cllist : array_like
        List of Cluster objects
    lam_n : float
        The richness of the N richest object for our selection function
        analysis
    lam_n_obs : float
        Observational uncertainty in lam_n
    """
    def __init__(self, cllist, lam_n, lam_n_obs):
        self.lam_n = lam_n
        self.lam_n_obs = lam_n_obs
        self.cllist = cllist


def read_hmf_files(addpath, cfgin, gencut=-1.):
    """Load the halo mass function

    My halo mass function comes from a cosmosis modules and is precalculated.
    This could probably be rewritten to calculate it on the fly using pyccl or
    some other tool.

    Parameters
    ----------
    addpath : str
        If not storing this in the default location, prepend onto the path
    cfgin : ConfigParser
        Config details for the run
    gencut : float, optional
        Use a different mass cut if I am generating a mock
    """
    log10m_cut = cfgin['General'].getfloat('log10Mcut')
    zhmf = cfgin['SurveyInfo'].getfloat('zhmf')
    cosmology = cfgin['Cosmology']['cosmology']
    path = "{}inputs/generateHMF/hmf_{}_cosmology/mass_function/".format(addpath, cosmology)

    # Find the index of the HMF we want to work with
    zlist = np.loadtxt(path+'z.txt', skiprows=1)
    zind = np.where(np.abs(zlist-zhmf) < 0.00003)[0][0]
    print("Loading HMF at redshift %f" % zlist[zind])

    # Extract the appropriate HMF
    with open(path+'dndmdz.txt', 'r') as fin:
        lines = fin.readlines()[1:]
        hmf = [float(row.split(' ')[zind]) for row in lines]

    # Load masses and make mass cut
    mlist = np.loadtxt(path+'mass.txt', skiprows=1)
    if gencut > 0:
        mass_cut = np.where((mlist >= 10.**gencut) & (mlist <= 10.**16))
    else:
        mass_cut = np.where((mlist >= 10.**log10m_cut) & (mlist <= 10.**16))

    # Recast into dn/(dz dlnm)
    lnmlist = np.asarray([np.log(m) for m in mlist])
    dndlnmdz = np.asarray([m*dndmdz for m, dndmdz in zip(mlist, hmf)])

    return lnmlist[mass_cut], dndlnmdz[mass_cut]


def load_clusters(cfgin, note, rescales, altpath=''):
    """Load the list of clusters being used for analysis

    Parameters
    ----------
    cfgin : ConfigParser
        Configuration details for the run
    note : int
        Additional details for the input file names. Used to tell realizations apart
    rescales : array_like
        Quantities to rescale the cluster measurements to ~unity
    altpath : str
        If clusters are stored somewhere besides the default path

    Returns
    -------
    real_list : array_like
        Realization objects loaded in
    lnpivot : float
        The median cluster mass (or estimate of such) to use as a default mass pivot
    """
    simtype = cfgin['General']['simtype']
    catalog = cfgin['General']['synth_catalog']
    n_clusters = cfgin['SurveyInfo'].getint('Nclusters')

    if simtype == 'synth10':
        real_list = load_synth10(catalog, note, rescales)
        return real_list, 0.0

    # Build the filename based on the type of simulation
    if simtype == 'real':
        cluster_fname = 'inputs/richestClustersInRedmapper_MantzXray_MatteoErrors.dat'
    elif len(altpath) > 5:
        cluster_fname = altpath
    elif simtype == 'synth':
        cluster_fname = './catalogs/mock/{}/richest{}_{}.dat'.format(catalog, n_clusters, note)
        
    # Actually load the clusters
    cllist, lnpivot, lam_n, lam_n_obs = load_clusters_from_file(cluster_fname, rescales)
    real_list = [Realization(cllist, lam_n, lam_n_obs)]

    return real_list, lnpivot


def load_clusters_from_file(fname, rescales):
    """Load the clusters from a file into Cluster objects

    Parameters
    ----------
    fname : str
        Path to the cluster catalog
    rescales : array_like
        Quantities to rescale the cluster measurements to ~unity

    Returns
    -------
    cllist : array_like
        Like of Cluster objects
    lnpivot : float
        Default pivot, the median cluster mass
    lam_n : float
        Richness of the least rich cluster
    lam_n_obs : float
        Uncertainty in lam_n
    """
    # Load the clusters
    inputs = np.loadtxt(fname, skiprows=1, usecols=[1, 2, 3, 4])
    lamobs, lamobs_uncert, mgas, mgas_uncert = inputs
    cllist = [Cluster(lamobs[i], lamobs_uncert[i], mgas[i], mgas_uncert[i], rescales)
              for i in range(len(lamobs))]

    # Find the median pivot
    lnpivot = np.log(np.median(mgas)/0.3)

    # Find the least rich cluster
    lam_n, idx = min((lamN, idx) for (idx, lamN) in enumerate(lamobs))
    lam_n_obs = lamobs_uncert[idx]

    print('%i Clusters loaded.' % (len(cllist)))
    return cllist, lnpivot, lam_n, lam_n_obs


def load_synth10(catalog, note, rescales):
    """Load 10 realizations"""
    real_list = []
    for kk in range(10):
        cluster_fname = './catalogs/mock/%s/richest30_%i.dat' % (catalog, note*10+kk)
        print("Read cluster", cluster_fname)
        cllist, lnpivot, lam_n, lam_n_obs = load_clusters_from_file(cluster_fname, rescales)
        real_list.append(Realization(cllist, lam_n, lam_n_obs))
    return real_list


class PriorContainer(object):
    """Hold all of the prior information"""
    def __init__(self, cfgin):
        path = '/home/matthewkirby/cosmosis/modules/xray-likelihood/priors/'
        fname = cfgin['MCMC']['priorfile']
        priorsin = cp.ConfigParser()
        priorsin.read(os.path.join(path + fname))

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


def build_output_files(cfgin, note):
    """Build the paths to the output files

    Parameters
    ----------
    cfgin : ConfigParser
        Configuration details for the run
    note : int
        Realization specific path details

    Returns
    -------
    out_chain : str
        Output file name for the chain
    out_like : str
        Output file name for the likelihood
    """
    simtype = cfgin['General']['simtype']
    prior = cfgin['MCMC']['prior']
    if simtype == 'synth' or simtype == 'synth10':
        catalog = cfgin['General']['synth_catalog']
        out_chain = './outputs/{}/chains/mcmc{}_{}prior_richest{}.out'.format(
            catalog, note, prior, cfgin['SurveyInfo'].getint('nclusters'))
        out_like = './outputs/{}/chains/lnL{}_{}priors.out'.format(catalog, note, prior)

    elif simtype == 'real':
        out_chain = './outputs/real_runs/chains/mcmc_{}priors.out'.format(prior)
        out_like = './outputs/real_runs/chains/lnL_{}priors.out'.format(prior)

    else:
        raise TypeError("Simulation type {} is an invalid type.".format(simtype))

    return out_chain, out_like


def select_pivots(cfgin, lnpivot):
    """Select the mass pivots that will be used in the analysis

    Parameters
    ----------
    cfgin : ConfigParser
        Configuration details for the run
    lnpivot : float
        Default pivot to use

    Returns
    -------
    pivots : array_like
        The two pivots to use in the analysis
    """
    lam_pivot = cfgin['General'].getfloat('rich_pivot')
    mgas_pivot = cfgin['General'].getfloat('mgas_pivot')
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


def load_truth(fname):
    """Load the true model"""
    model = cp.ConfigParser()
    model.read(fname)
    params = [model['Model'].getfloat('r'),
              model['Model'].getfloat('mgas0'),
              model['Model'].getfloat('alphamg'),
              model['Model'].getfloat('s0mg'),
              model['Model'].getfloat('lam0'),
              model['Model'].getfloat('alphalam'),
              model['Model'].getfloat('s0lam'),
              model['Model'].getfloat('dhmf')]

    return params


def find_mock_lamobs_max(simtype, addpath=''):
    """Find the maximum relevant observed richness as a function of true richness

    This is used as the upper bound on the integrals.
    """
    lamtrue_grid = np.loadtxt('{}projection_model/cdfs/lamtrue.dat'.format(addpath))
    lamobs_grid = np.loadtxt('{}projection_model/cdfs/lamobs.dat'.format(addpath))
    cdf_grid = np.loadtxt('{}projection_model/cdfs/cdflist.dat'.format(addpath))
    maxlamobs = []

    if simtype == 'real':
        return [lamtrue_grid, np.ones(len(lamtrue_grid))*1999.]

    for i in range(len(lamtrue_grid)):
        cdf = cdf_grid[i]
        idxs = np.array(np.where(cdf < 1.0)[0])
        lamobs_grid_cut = lamobs_grid[idxs]
        maxlamobs.append(lamobs_grid_cut[-1])
        
    return [lamtrue_grid, maxlamobs]
