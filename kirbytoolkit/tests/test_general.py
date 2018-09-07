import numpy as np
from configparser import ConfigParser
import kirbytoolkit as ktk


def test_mass_conversion():
    """Test the mass_conversion function from kirbytoolkit"""

    cosmology = {'flat': True, 'H0': 100, 'Om0': 0.301, 'Ob0': 0.048, 'sigma8': 0.798, 'ns': 0.973}

    sample_m200m = np.asarray([.5e14, .7e14, 1.0e14, 2e14, 5e14])
    desired_lnm500c = [30.960801914658372, 31.28419690995847, 31.627409974386172,
                       32.297005227641485, 33.19327330555602]

    # Test for a single mass
    actual = np.log(ktk.mass_conversion(1.0e14, 0.2, cosmology, mass_is_log=False))
    np.testing.assert_almost_equal(actual, np.float64(31.627409974386172), decimal=6)
    # Test for a single lnmass
    actual = ktk.mass_conversion(np.log(1.0e14), 0.2, cosmology, mass_is_log=True)
    np.testing.assert_almost_equal(actual, np.float64(31.627409974386172), decimal=6)
    # Test for an array of masses
    actual = ktk.mass_conversion(sample_m200m, 0.2, cosmology, mass_is_log=False)
    np.testing.assert_allclose(actual, np.exp(desired_lnm500c), rtol=1e-6)
    # Test for an array of lnmasses
    actual = ktk.mass_conversion(np.log(sample_m200m), 0.2, cosmology, mass_is_log=True)
    np.testing.assert_allclose(actual, desired_lnm500c, rtol=1e-6)


def test_comoving_volume():
    """Test the comoving volume calculation in SurveyInfo"""
    cfgin = ConfigParser()
    cfgin.add_section('Survey Info')
    cfgin.set('Survey Info', 'survey_area', '10000')
    cfgin.set('Survey Info', 'zlo', '0.1')
    cfgin.set('Survey Info', 'zhi', '0.3')

    cosmology = {'flat': True, 'H0': 100, 'Om0': 0.301, 'Ob0': 0.048, 'sigma8': 0.798, 'ns': 0.973}

    sinfo = ktk.SurveyInfo(cfgin, cosmology)
    np.testing.assert_allclose(sinfo.survey_volume, np.float64(567534172.), rtol=1e-4)
