import pytest
import numpy as np
import re
from gpaw.response.bse import BSE


@pytest.mark.later  # uses nicl2_pw fixture which does not work with new code
@pytest.mark.response
def test_response_bse_parse_bands(in_tmp_dir, gpw_files):

    bse = BSE(gpw_files['mos2_pw'],
              ecut=10,
              valence_bands=4,
              conduction_bands=3,
              eshift=0.8,
              nbands=15)

    # Check consistency with written results

    n_valence_bands = int(bse.gs.nvalence / 2)
    correct_valence_sn = np.atleast_2d(
        range(n_valence_bands - 4, n_valence_bands))
    correct_conduction_sn = np.atleast_2d(
        range(n_valence_bands, n_valence_bands + 3))

    assert np.array_equal(correct_valence_sn, bse.val_sn)
    assert np.array_equal(correct_conduction_sn, bse.con_sn)

    with pytest.raises(NotImplementedError,
                       match='Automatic band generation is currently*'):
        bse2 = BSE(gpw_files['bse_al'],
                   valence_bands=range(4),
                   conduction_bands=5,
                   nbands=4,
                   ecut=10,
                   )
        bse2  # does nothing; this is just here to avoid a linting error

    with pytest.raises(NotImplementedError,
                       match=re.escape('For a spin-polarized calculation, '
                                       'bands must be specified as lists '
                                       'of shape (2,n)')):

        bse3 = BSE(gpw_files['nicl2_pw'],
                   valence_bands=4,
                   conduction_bands=[range(4), range(4)],
                   nbands=4,
                   ecut=10,
                   )
        bse3  # does nothing; this is just here to avoid a linting error
