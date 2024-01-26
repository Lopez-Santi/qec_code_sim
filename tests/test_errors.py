'''
Usage:
    python test_errors.py
'''
import unittest
import random
import sys
import logging
import numpy as np
from general_qec.errors import gate_error
from general_qec.errors import errored_adj_CNOT, errored_non_adj_CNOT
from general_qec.errors import errored_flipped_adj_CNOT, errored_flipped_non_adj_CNOT
from general_qec.errors import errored_adj_CZ, errored_non_adj_CZ
from general_qec.errors import line_errored_CNOT, line_errored_CZ
from general_qec.errors import random_qubit_x_error, random_qubit_z_error
from general_qec.gates import sigma_y, sigma_z
from general_qec.qec_helpers import one, zero, superpos
from general_qec.qec_helpers import collapse_dm

LOGGER = logging.getLogger(__name__)


class TestErrors(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the `errors` (noise model) module - Krauss error functions."""

    def test_random_qubit_x_error(self):
        """Test `random_qubit_x_error()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(11)
        logical_state = zero
        errored_logical_state, error_index = random_qubit_x_error(logical_state)
        self.assertEqual(error_index, 0)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertTrue(np.all(errored_logical_state == one))
        logical_state = np.kron(one, one)
        errored_logical_state, error_index = random_qubit_x_error(logical_state, (1,1))
        self.assertEqual(error_index, 1)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertTrue(np.all(errored_logical_state == np.kron(one, zero)))
        random.seed(11)
        logical_state = np.kron(np.kron(zero, zero), zero)
        errored_logical_state, error_index = random_qubit_x_error(logical_state)
        self.assertEqual(error_index, 2)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertTrue(
            np.all(errored_logical_state == np.kron(zero, np.kron(zero, one)))
        )
        logical_state = np.kron(np.kron(zero, zero), zero)
        errored_logical_state, error_index = random_qubit_x_error(logical_state, (2,2))
        self.assertEqual(error_index, 2)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertTrue(
            np.all(errored_logical_state == np.kron(zero, np.kron(zero, one)))
        )

    def test_random_qubit_z_error(self):
        """Test `random_qubit_x_error()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(11)
        logical_state = superpos
        errored_logical_state, error_index = random_qubit_z_error(logical_state)
        self.assertEqual(error_index, 0)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertEqual(logical_state[0], errored_logical_state[0])
        self.assertEqual(logical_state[1], -1*errored_logical_state[1])
        logical_state = np.kron(superpos, superpos)
        errored_logical_state, error_index = random_qubit_z_error(logical_state, (1,1))
        self.assertEqual(error_index, 1)
        self.assertAlmostEqual(np.sum(errored_logical_state**2), 1)
        self.assertEqual(logical_state[2], errored_logical_state[2])
        self.assertEqual(logical_state[3], -1*errored_logical_state[3])

    def test_gate_error(self):
        """Test `gate_error()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(zero, zero)
        rho = np.outer(psi, psi.conj().T)
        rho_prime = gate_error(rho, 0.1, 0, 2)
        self.assertAlmostEqual(np.trace(rho_prime.real), 1) # pylint: disable=no-member
        psi = np.kron(np.matmul(sigma_y, superpos), np.kron(superpos, superpos))
        rho = np.outer(psi, psi.conj().T)
        rho_prime = gate_error(rho, 0.1, 1, 3)
        self.assertAlmostEqual(np.trace(rho_prime.real), 1) # pylint: disable=no-member
        self.assertAlmostEqual(rho_prime[0][0], 1./8 + 0j)

    def test_adjacent_errored_cnots(self):
        """Test various errored cnot functions (line connectivity)"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(one, zero)
        rho = np.outer(psi, psi.conj().T)
        # apply a zero-error CNOT gate
        errored_rho = errored_adj_CNOT(rho, 0, 1, [0., 0.])
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[3][3], 1+0j)
        self.assertTrue(np.all(errored_psi == np.kron(one, one)))
        # apply a high-error CNOT gate
        errored_rho = errored_adj_CNOT(rho, 0, 1, [0., 0.5])
        random.seed(13)  # seed chosen to find a gate error
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[3][3], 2./3.+0j)
        self.assertTrue(np.all(errored_psi == np.kron(one, zero)),
                        msg="Random seed may have failed to produce expected state.")

    def test_flipped_adjacent_errored_cnots(self):
        """Test various errored cnot functions (line connectivity)"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(zero, one)
        rho = np.outer(psi, psi.conj().T)
        # apply a zero-error CNOT gate
        errored_rho = errored_flipped_adj_CNOT(rho, 1, 0, [0., 0.])
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[3][3], 1+0j)
        self.assertTrue(np.all(errored_psi == np.kron(one, one)))
        # apply a high-error CNOT gate
        rho = np.outer(psi, psi.conj().T)
        errored_rho = errored_flipped_adj_CNOT(rho, 1, 0, [0.5, 0.0])
        random.seed(13)  # seed chosen to find a gate error
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[1][1], 1./3.+0j)
        self.assertTrue(np.all(errored_psi == np.kron(zero, one)),
                        msg="Random seed may have failed to produce expected state.")

    def test_nonadjacent_errored_cnots(self):
        """Test various errored non-adjacent cnot functions (line connectivity)"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.kron(np.kron(one, zero), zero)
        rho = np.outer(psi, psi.conj().T)
        # apply a zero-error CNOT gate
        errored_rho = errored_non_adj_CNOT(rho, 0, 2, [0., 0., 0.])
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[5][5], 1+0j)
        self.assertTrue(np.all(errored_psi == np.kron(np.kron(one, zero), one)))
        # apply a high-error CNOT gate
        random.seed(10)
        errored_rho = errored_non_adj_CNOT(rho, 0, 2, [0., 0., 0.5])
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[4][4], 4./9.+0j)
        self.assertTrue(np.all(errored_psi == np.kron(np.kron(one, zero), one)),
                        msg="Random seed may have failed to produce expected state.")
        psi = np.kron(np.kron(zero, zero), one)
        rho = np.outer(psi, psi.conj().T)
        # apply a zero-error CNOT gate
        errored_rho = errored_flipped_non_adj_CNOT(rho, 2, 0, [0.0, 0., 0.])
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[5][5], 1.+0j)
        self.assertTrue(np.all(errored_psi == np.kron(np.kron(one, zero), one)))
        # apply a high-error CNOT gate
        errored_rho = errored_flipped_non_adj_CNOT(rho, 2, 0, [0.5, 0., 0.])
        random.seed(10)  # seed chosen for no error
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[1][1], 4./9.+0j)
        self.assertTrue(np.all(errored_psi == np.kron(np.kron(one, zero), one)),
                        msg="Random seed may have failed to produce expected state.")

    def test_distant_nonadjacent_errored_cnots(self):
        """Test various distant errored non-adjacent cnot functions (line connectivity)"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        # -- larger non-adj CNOT
        psi = np.kron(np.kron(np.kron(np.kron(zero, one), zero), zero), zero)
        targ_psi = np.kron(np.kron(np.kron(np.kron(zero, one), zero), one), zero)
        # apply a zero-error CNOT gate
        errored_rho = line_errored_CNOT(psi, 1, 3, [0.0]*5)
        errored_psi = collapse_dm(errored_rho)
        self.assertTrue(np.all(errored_psi == targ_psi))
        # -- larger flipped non-adj CNOT
        psi = np.kron(np.kron(np.kron(np.kron(np.kron(zero, zero), zero), zero), one), one)
        targ_psi = np.kron(np.kron(np.kron(np.kron(np.kron(zero, one), zero), zero), one), one)
        errored_rho = line_errored_CNOT(psi, 4, 1, [0.0]*6)
        errored_psi = collapse_dm(errored_rho)
        self.assertTrue(np.all(errored_psi == targ_psi))
        # -- larger flipped non-adj CNOT
        psi = np.kron(np.kron(np.kron(np.kron(np.kron(zero, zero), zero), one), one), one)
        targ_psi = np.kron(np.kron(np.kron(np.kron(np.kron(one, zero), zero), one), one), one)
        errored_rho = line_errored_CNOT(psi, 3, 0, [0.0]*6)
        errored_psi = collapse_dm(errored_rho)
        self.assertTrue(np.all(errored_psi == targ_psi))
        # -- more distant, high error flipped non-adj CNOT
        psi = np.kron(np.kron(np.kron(np.kron(np.kron(zero, zero), zero), one), one), one)
        targ_psi = np.kron(np.kron(np.kron(np.kron(np.kron(one, zero), zero), one), one), one)
        errored_rho = line_errored_CNOT(psi, 5, 0, [0.5]*6)
        errored_psi = collapse_dm(errored_rho)
        diagonals = np.diag(errored_rho)
        # the way the two-qubit gate errors work, we apply errors to every qubit in
        # this line *except* the control - so we can hit 32 states, and the mixing
        # is very even with such high error
        self.assertEqual(len(diagonals[diagonals>0]), 32)
        self.assertAlmostEqual(np.mean(diagonals[diagonals>0]), 1./32.)

    def test_adjacent_line_errored_czs(self):
        """Test various adjacent errored cz functions (line connectivity)"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        psi = np.matmul(np.kron(np.identity(2), sigma_z), np.kron(zero, one))
        rho = np.outer(psi, psi.conj().T)
        # apply a zero-error CZ gate
        errored_rho = errored_adj_CZ(rho, 0, 1, [0., 0.])
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[1][1], 1.+0j)
        self.assertTrue(np.all(errored_psi == np.kron(zero, one)))
        # apply a zero-error CZ gate flipped
        errored_rho = errored_adj_CZ(rho, 1, 0, [0., 0.])
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[1][1], 1.+0j)
        self.assertTrue(np.all(errored_psi == np.kron(zero, one)))
        # apply a high-error CZ gate
        errored_rho = errored_adj_CZ(rho, 0, 1, [0., 0.5])
        random.seed(18)  # seed chosen to find an error
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[1][1], 2./3.+0j)
        self.assertTrue(np.all(errored_psi == np.kron(zero, zero)),
                        msg="Random seed may have failed to produce expected state.")

    def test_nonadjacent_line_errored_czs(self):
        """Test various non-adjacent errored cz functions (line connectivity)"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        # -- non-adjacent cz -> test through wrapper
        psi = np.matmul(
            np.kron(sigma_z, np.identity(2**2)),
            np.kron(np.kron(one, zero), one)
        )
        rho = np.outer(psi, psi.conj().T)
        # apply a zero-error CZ gate
        errored_rho = errored_non_adj_CZ(rho, 0, 2, [0., 0., 0.])
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[5][5], 1.+0j)
        self.assertTrue(np.all(errored_psi == np.kron(np.kron(one, zero), one)))
        # apply a zero-error CZ gate flipped
        errored_rho = errored_non_adj_CZ(rho, 2, 0, [0., 0., 0.])
        errored_psi = collapse_dm(errored_rho)
        self.assertAlmostEqual(errored_rho[5][5], 1.+0j)
        self.assertTrue(np.all(errored_psi == np.kron(np.kron(one, zero), one)))
        # apply a high-error CZ gate
        errored_rho = errored_non_adj_CZ(rho, 0, 2, [0., 0., 0.1])
        errored_psi = collapse_dm(errored_rho)

    def test_distant_nonadjacent_line_errored_czs(self):
        """Test various distant non-adjacent errored cz functions (line connectivity)"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        # -- more distant non-adjacent cz -> test through wrapper
        psi = np.matmul(
            np.kron(sigma_z, np.identity(2**4)),
            np.kron(one, np.kron(np.kron(zero, zero), np.kron(zero, one)))
        )
        errored_rho = line_errored_CZ(psi, 0, 4, [0.1, 0.1, 0.1, 0.1, 0.1])
        random.seed(10)
        errored_psi = collapse_dm(errored_rho)
        self.assertTrue(np.real(errored_rho[29][29]) > 0.0)
        self.assertTrue(np.real(errored_rho[29][29]) < 0.022)
        self.assertEqual(errored_psi[19], 1.0+0.0j)
        # -- more distant non-adjacent cz -> test through wrapper
        base_psi = np.kron(np.kron(np.kron(np.kron(zero, one), zero), one), zero)
        psi = np.matmul(
            np.kron(np.kron(np.identity(2**3), sigma_z), np.identity(2)),
            base_psi
        )
        errored_rho = line_errored_CZ(psi, 1, 3, [0.0]*5)
        errored_psi = collapse_dm(errored_rho)
        self.assertTrue(np.all(errored_psi == base_psi))
        # -- more distant non-adjacent cz -> test through wrapper
        base_psi = np.kron(np.kron(np.kron(np.kron(np.kron(zero, one), zero), zero), one), one)
        psi = np.matmul(
            np.kron(np.kron(np.identity(2), sigma_z), np.identity(2**4)),
            base_psi
        )
        errored_rho = line_errored_CZ(psi, 4, 1, [0.0]*6)
        errored_psi = collapse_dm(errored_rho)
        self.assertTrue(np.all(errored_psi == base_psi))
        # -- more distant non-adjacent cz -> test through wrapper
        base_psi = np.kron(np.kron(np.kron(np.kron(np.kron(one, zero), zero), one), zero), one)
        psi = np.matmul(
            np.kron(sigma_z, np.identity(2**5)),
            base_psi
        )
        errored_rho = line_errored_CZ(psi, 3, 0, [0.0]*6)
        errored_psi = collapse_dm(errored_rho)
        self.assertTrue(np.all(errored_psi == base_psi))
        # -- very distant, high error non-adjacent cz -> test through wrapper
        base_psi = np.kron(np.kron(np.kron(np.kron(np.kron(one, zero), zero), one), zero), one)
        psi = np.matmul(
            np.kron(sigma_z, np.identity(2**5)),
            base_psi
        )
        errored_rho = line_errored_CZ(psi, 5, 0, [0.5]*6)
        errored_psi = collapse_dm(errored_rho)
        diagonals = np.diag(errored_rho)
        # the way the two-qubit gate errors work, we apply errors to every qubit in
        # this line *except* the control - so we can hit 32 states, and the mixing
        # is very even with such high error
        self.assertEqual(len(diagonals[diagonals>0]), 32)
        self.assertAlmostEqual(np.mean(diagonals[diagonals>0]), 1./32.)


if __name__ == '__main__':
    unittest.main()
