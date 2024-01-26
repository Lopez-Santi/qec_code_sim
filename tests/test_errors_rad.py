'''
Usage:
    python test_errors_rad.py

Test `errors.py` - focus on the relaxation and dephasing ("rad") functions.
'''
import unittest
import random
import sys
import logging
import numpy as np
from general_qec.qec_helpers import one, zero, superpos
from general_qec.errors import rad_error, line_rad_CNOT
from general_qec.errors import rad_adj_CNOT, rad_non_adj_CNOT
from general_qec.errors import rad_flipped_adj_CNOT, rad_flipped_non_adj_CNOT
from general_qec.errors import rad_adj_CZ, rad_non_adj_CZ, line_rad_CZ
from general_qec.gates import sigma_z
from general_qec.qec_helpers import collapse_dm

LOGGER = logging.getLogger(__name__)


class TestRadErrors(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the `errors` (noise model) module -- the "rad" functions."""

    def test_rad_error(self):
        """Test `rad_error()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        # basic check
        psi = np.kron(zero, zero)
        rho = np.outer(psi, psi.conj().T)
        rho_prime = rad_error(rho, 0.1, 0.1, 1e-9)
        self.assertAlmostEqual(np.trace(rho_prime), 1.0)
        # more intersting state, larger decay
        psi = 1./np.sqrt(2)*np.array([1.0, 0.0, 0.0, 1.0])
        rho = np.outer(psi, psi.conj().T)
        rho_prime = rad_error(rho, 1e-8, 1e-8, 1e-9)
        random.seed(10)  # carefully chosen for collapsed state
        psi_prime = collapse_dm(rho_prime)
        self.assertAlmostEqual(np.trace(rho_prime), 1.0)
        self.assertTrue(np.all(psi_prime == np.array([0., 0., 1., 0.])),
                        msg="Random seed may not have produced the right collapsed state.")
        # more intersting state, just t1 -> drive hard to ground state
        psi = 1./np.sqrt(2)*np.array([1.0, 0.0, 0.0, 1.0])
        rho = np.outer(psi, psi.conj().T)
        rho_prime = rad_error(rho, 1, 1e9, 1e2)
        psi_prime = collapse_dm(rho_prime)
        self.assertAlmostEqual(np.trace(rho_prime), 1.0)
        self.assertTrue(np.all(psi_prime == np.array([1., 0., 0., 0.])),
                        msg="Random seed may not have produced the right collapsed state.")
        # more intersting state, just t2 -> drive off-diagonals to zero hard
        psi = np.kron(superpos, superpos)
        rho = np.outer(psi, psi.conj().T)
        rho_prime = rad_error(rho, 1e9, 1, 1e2)
        random.seed(10)  # carefully chosen for collapsed state
        psi_prime = collapse_dm(rho_prime)
        self.assertAlmostEqual(np.trace(rho_prime), 1.0)
        self.assertAlmostEqual(np.sum(rho_prime) - np.trace(rho_prime), 0.0)
        self.assertTrue(np.all(psi_prime == np.array([0., 0., 1., 0.])),
                        msg="Random seed may not have produced the right collapsed state.")



class TestRadErrorsCNOT(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the `errors` (noise model) module -- the "rad" functions."""

    def test_adjacent_line_rad_cnot(self):
        """Test adjacent line-connected CNOT with RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(one, zero)
        rho = np.outer(psi, psi.conj().T)
        control, target, t1, t2, tg = 0, 1, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = rad_adj_CNOT(rho, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[3][3], 1+0j)
        self.assertTrue(np.all(errored_psi == np.kron(one, one)))
        # more qubits, more interesting state
        psi = np.kron(np.kron(np.kron(zero, superpos), zero), zero)
        rho = np.outer(psi, psi.conj().T)
        control, target, t1, t2, tg = 1, 2, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = rad_adj_CNOT(rho, control, target, t1, t2, tg)
        random.seed(10)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertEqual(errored_psi[6], 1+0j)

    def test_nonadjacent_line_rad_cnot(self):
        """Test non-adjacent line-connected CNOT with RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(np.kron(np.kron(one, zero), zero), zero)
        rho = np.outer(psi, psi.conj().T)
        targ_psi = np.kron(np.kron(np.kron(one, zero), one), zero)
        control, target, t1, t2, tg = 0, 2, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = rad_non_adj_CNOT(rho, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))

    def test_flipped_adjacent_line_rad_cnot(self):
        """Test flipped adjacent line-connected CNOT with RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(zero, one)
        rho = np.outer(psi, psi.conj().T)
        control, target, t1, t2, tg = 1, 0, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = rad_flipped_adj_CNOT(rho, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[3][3], 1+0j)
        self.assertTrue(np.all(errored_psi == np.kron(one, one)))
        # more qubits, more interesting state
        psi = np.kron(np.kron(np.kron(zero, zero), one), zero)
        rho = np.outer(psi, psi.conj().T)
        targ_psi = np.kron(np.kron(np.kron(zero, one), one), zero)
        control, target, t1, t2, tg = 2, 1, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = rad_flipped_adj_CNOT(rho, control, target, t1, t2, tg)
        random.seed(10)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))

    def test_flipped_nonadjacent_line_rad_cnot(self):
        """Test flipped non-adjacent line-connected CNOT with RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(np.kron(np.kron(zero, zero), zero), one)
        rho = np.outer(psi, psi.conj().T)
        targ_psi = np.kron(np.kron(np.kron(zero, one), zero), one)
        control, target, t1, t2, tg = 3, 1, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = rad_flipped_non_adj_CNOT(rho, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))

    def test_distant_nonadjacent_line_rad_cnot(self):
        """Test distant non-adjacent line-connected CNOT with krauss and RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(np.kron(np.kron(np.kron(zero, one), zero), zero), zero)
        targ_psi = np.kron(np.kron(np.kron(np.kron(zero, one), zero), one), zero)
        control, target, t1, t2, tg = 1, 3, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = line_rad_CNOT(psi, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))
        psi = np.kron(np.kron(np.kron(np.kron(one, one), one), zero), zero)
        targ_psi = np.kron(np.kron(np.kron(np.kron(one, one), one), zero), one)
        control, target, t1, t2, tg = 0, 4, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = line_rad_CNOT(psi, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))


class TestRadErrorsCZ(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the `errors` (noise model) module -- the "RAD" CZ functions."""

    def test_adjacent_line_rad_cz(self):
        """Test adjacent line-connected cz with RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.matmul(np.kron(np.identity(2), sigma_z), np.kron(one, one))
        rho = np.outer(psi, psi.conj().T)
        targ_psi = np.kron(one, one)
        control, target, t1, t2, tg = 0, 1, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = rad_adj_CZ(rho, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.allclose(errored_psi, targ_psi))
        # more qubits, more interesting state
        psi = np.kron(np.kron(np.kron(one, superpos), superpos), superpos)
        rho = np.outer(psi, psi.conj().T)
        control, target, t1, t2, tg = 1, 2, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = rad_adj_CZ(rho, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[:8,:8])), 0.)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[8:,:8])), 0.)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[:8,8:])), 0.)
        self.assertAlmostEqual(np.mean(np.abs(errored_rho[8:,8:])), 1./8.)
        self.assertAlmostEqual(np.max(errored_psi[:8]), 0+0j)
        self.assertAlmostEqual(np.max(errored_psi[8:]), 1+0j)

    def test_nonadjacent_line_rad_cz(self):
        """Test non-adjacent line-connected CZ with RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.matmul(
            np.kron(np.kron(np.identity(2**2), sigma_z), np.identity(2)),
            np.kron(np.kron(np.kron(one, zero), one), zero)
        )
        rho = np.outer(psi, psi.conj().T)
        targ_psi = np.kron(np.kron(np.kron(one, zero), one), zero)
        control, target, t1, t2, tg = 0, 2, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = rad_non_adj_CZ(rho, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))

    def test_distant_nonadjacent_line_rad_cz(self):
        """Test distant non-adjacent line-connected CZ with RAD errors."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.matmul(
            np.kron(np.kron(np.identity(2**3), sigma_z), np.identity(2)),
            np.kron(np.kron(np.kron(np.kron(zero, one), zero), one), zero)
        )
        targ_psi = np.kron(np.kron(np.kron(np.kron(zero, one), zero), one), zero)
        control, target, t1, t2, tg = 1, 3, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = line_rad_CZ(psi, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))
        psi = np.matmul(
            np.kron(np.identity(2**4), sigma_z),
            np.kron(np.kron(np.kron(np.kron(one, one), one), zero), one)
        )
        targ_psi = np.kron(np.kron(np.kron(np.kron(one, one), one), zero), one)
        control, target, t1, t2, tg = 0, 4, 1e3, 1e3, 1e-9 # pylint: disable=invalid-name
        errored_rho = line_rad_CZ(psi, control, target, t1, t2, tg)
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(np.trace(errored_rho), 1.0)
        self.assertTrue(np.all(errored_psi == targ_psi))


if __name__ == '__main__':
    unittest.main()
