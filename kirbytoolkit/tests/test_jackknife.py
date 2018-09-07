import numpy as np
import kirbytoolkit as ktk


def test_jackknife_arr():
    arr = [1, 2, 2, 3, 4]
    jkvar_true = 0.26
    jkvar_code = ktk.jackknife_array(arr)
    np.testing.assert_almost_equal(jkvar_code, jkvar_true, decimal=7)

    # Test a longer array
    arr = np.loadtxt('randomdata.dat')
    jkvar_true = 0.003031329328040*0.003031329328040
    jkvar_code = ktk.jackknife_array(arr)
    np.testing.assert_almost_equal(jkvar_code, jkvar_true, decimal=7)
    
