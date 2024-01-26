'''
Usage:
    python test_errors_prob_rad_cz.py

Test `errors.py` - focus on the "prob rad" functions.
'''
import unittest
import sys
import logging
import numpy as np
from general_qec.qec_helpers import zero, one, superpos
from general_qec.errors import prob_line_rad_CZ
from general_qec.gates import sigma_z
from general_qec.qec_helpers import collapse_dm

LOGGER = logging.getLogger(__name__)


class TestProbRadErrorsCZ(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the `errors` (noise model) module -- the Krauss+"RAD" CZ functions."""

    def test_adjacent_prob_line_rad_cz(self):
        """Test adjacent line-connected cz with krauss and RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.matmul(np.kron(np.identity(2), sigma_z), np.kron(one, one))
        targ_psi = np.kron(one, one)
        control, target, t1, t2, tg, krauss = 0, 1, 1e3, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.allclose(errored_psi, targ_psi))
        # more qubits, more interesting state
        psi = np.kron(np.kron(np.kron(one, superpos), superpos), superpos)
        control, target, t1, t2, tg, krauss = 1, 2, 1e3, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[:8,:8])), 0.)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[8:,:8])), 0.)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[:8,8:])), 0.)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[8:,8:])), 1./8.)
        self.assertAlmostEqual(np.max(errored_psi[:8]), 0+0j)
        self.assertAlmostEqual(np.max(errored_psi[8:]), 1+0j)

    def test_flipped_adjacent_prob_line_rad_cz(self):
        """Test flipped adjacent line-connected CZ with krauss and RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.matmul(np.kron(sigma_z, np.identity(2)), np.kron(one, one))
        targ_psi = np.kron(one, one)
        control, target, t1, t2, tg, krauss = 1, 0, 1e3, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.allclose(errored_psi, targ_psi))
        # more qubits, more interesting state
        psi = np.kron(np.kron(np.kron(one, superpos), superpos), superpos)
        control, target, t1, t2, tg, krauss = 2, 1, 1e3, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[:8,:8])), 0.)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[8:,:8])), 0.)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[:8,8:])), 0.)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[8:,8:])), 1./8.)
        self.assertAlmostEqual(np.max(errored_psi[:8]), 0+0j)
        self.assertAlmostEqual(np.max(errored_psi[8:]), 1+0j)

    def test_nonadjacent_prob_line_rad_cz(self):
        """Test non-adjacent line-connected CZ with krauss and RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.matmul(
            np.kron(np.kron(np.identity(2**2), sigma_z), np.identity(2)),
            np.kron(np.kron(np.kron(one, zero), one), zero)
        )
        targ_psi = np.kron(np.kron(np.kron(one, zero), one), zero)
        control, target, t1, t2, tg, krauss = 0, 2, 1e3, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))

    def test_flipped_nonadjacent_prob_line_rad_cz(self):
        """Test flipped non-adjacent line-connected CZ with krauss and RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.matmul(
            np.kron(np.kron(np.identity(2), sigma_z), np.identity(2**2)),
            np.kron(np.kron(np.kron(zero, one), zero), one)
        )
        targ_psi = np.kron(np.kron(np.kron(zero, one), zero), one)
        control, target, t1, t2, tg, krauss = 3, 1, 1e3, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))

    def test_distant_nonadjacent_prob_line_rad_cz(self):
        """Test distant non-adjacent line-connected CZ with krauss and RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.matmul(
            np.kron(np.kron(np.identity(2**3), sigma_z), np.identity(2)),
            np.kron(np.kron(np.kron(np.kron(zero, one), zero), one), zero)
        )
        targ_psi = np.kron(np.kron(np.kron(np.kron(zero, one), zero), one), zero)
        control, target, t1, t2, tg, krauss = 1, 3, 1e3, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))
        psi = np.matmul(
            np.kron(np.identity(2**4), sigma_z),
            np.kron(np.kron(np.kron(np.kron(one, one), one), zero), one)
        )
        targ_psi = np.kron(np.kron(np.kron(np.kron(one, one), one), zero), one)
        control, target, t1, t2, tg, krauss = 0, 4, 1e3, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))

    def test_low_t1_distant_nonadjacent_prob_line_rad_cz(self):
        """Test low t1 distant non-adjacent line-connected CZ with krauss and RAD errors."""
        # test low t1 -> drive to ground
        psi = np.matmul(
            np.kron(np.identity(2**4), sigma_z),
            np.kron(np.kron(np.kron(np.kron(one, one), one), zero), one)
        )
        targ_psi = np.kron(np.kron(np.kron(np.kron(zero, zero), zero), zero), zero)
        control, target, t1, t2, tg, krauss = 0, 4, 1e-12, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.allclose(errored_psi, targ_psi))

    def test_low_t2_distant_nonadjacent_prob_line_rad_cz(self):
        """Test low t2 distant non-adjacent line-connected CZ with krauss and RAD errors."""
        # test low t2 -> kill off-diagonals
        psi = np.kron(np.kron(np.kron(np.kron(one, superpos), superpos), superpos), zero)
        control, target, t1, t2, tg, krauss = 0, 4, 1e3, 1e-12, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertAlmostEqual(np.sum(errored_rho) - np.trace(errored_rho), 0.0)

    def test_low_t2_large_krauss_distant_nonadjacent_prob_line_rad_cz(self):
        """Test low t2, large krauss distant non-adj. line-connected CZ w/ krauss & RAD errors."""
        # test low t2 -> kill off-diagonals, large krauss -> ~even mix of outcomes
        psi = np.kron(np.kron(np.kron(np.kron(one, superpos), superpos), superpos), zero)
        control, target, t1, t2, tg, krauss = 0, 4, 1e3, 1e-12, 1e-9, [0.5]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertAlmostEqual(np.sum(errored_rho) - np.trace(errored_rho), 0.0)
        # in our 2-qubit error model, no error on qubit 0 in this circuit, so half
        # the diagonals of the density matrix will be 0 and the other half ~1/16 for
        # even mixing wit 5 qubits -> so mean of all should be ~1/32
        self.assertAlmostEqual(np.mean(np.diag(errored_rho)), 1./32.)

    def test_distant_flipped_nonadjacent_prob_line_rad_cz(self):
        """Test distant flipped non-adjacent line-connected CZ with krauss and RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.matmul(
            np.kron(np.kron(np.identity(2), sigma_z), np.identity(2**3)),
            np.kron(np.kron(np.kron(np.kron(zero, one), zero), one), zero)
        )
        targ_psi = np.kron(np.kron(np.kron(np.kron(zero, one), zero), one), zero)
        control, target, t1, t2, tg, krauss = 3, 1, 1e3, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))
        psi = np.matmul(
            np.kron(sigma_z, np.identity(2**4)),
            np.kron(np.kron(np.kron(np.kron(one, zero), one), zero), one)
        )
        targ_psi = np.kron(np.kron(np.kron(np.kron(one, zero), one), zero), one)
        control, target, t1, t2, tg, krauss = 4, 0, 1e3, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))

    def test_low_t1_distant_flipped_nonadjacent_prob_line_rad_cz(self):
        """Test low t1 distant flipped non-adjacent lined CZ with krauss and RAD errors."""
        # test low t1 -> drive to ground
        psi = np.kron(np.kron(np.kron(np.kron(one, one), one), zero), one)
        targ_psi = np.kron(np.kron(np.kron(np.kron(zero, zero), zero), zero), zero)
        control, target, t1, t2, tg, krauss = 4, 0, 1e-12, 1e3, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.allclose(errored_psi, targ_psi))

    def test_low_t2_distant_flipped_nonadjacent_prob_line_rad_cz(self):
        """Test low t2 distant flipped non-adjacent lined CZ with krauss and RAD errors."""
        # test low t2 -> kill off-diagonals
        psi = np.kron(np.kron(np.kron(np.kron(zero, superpos), superpos), superpos), one)
        control, target, t1, t2, tg, krauss = 4, 0, 1e3, 1e-12, 1e-9, [0.]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertAlmostEqual(np.sum(errored_rho) - np.trace(errored_rho), 0.0)

    def test_low_t2_large_krauss_distant_flipped_nonadjacent_prob_line_rad_cz(self):
        """Test low t2, large krauss distant flipped non-adj. lined CZ w/ krauss & RAD errors."""
        # test low t2 -> kill off-diagonals, large krauss -> ~even mix of outcomes
        psi = np.kron(np.kron(np.kron(np.kron(one, superpos), superpos), superpos), zero)
        control, target, t1, t2, tg, krauss = 4, 0, 1e3, 1e-12, 1e-9, [0.5]*len(psi) # pylint: disable=invalid-name
        errored_rho = prob_line_rad_CZ(psi, control, target, t1, t2, tg, krauss)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertAlmostEqual(np.sum(errored_rho) - np.trace(errored_rho), 0.0)
        # in our 2-qubit error model, no error on qubit 4 in this circuit, so half
        # the diagonals of the density matrix will be 0 and the other half ~1/16 for
        # even mixing wit 5 qubits -> so mean of all should be ~1/32
        self.assertAlmostEqual(np.mean(np.diag(errored_rho)), 1./32.)


if __name__ == '__main__':
    unittest.main()
