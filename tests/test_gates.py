'''
Usage:
    python test_gates.py
'''
import unittest
import sys
import logging
import numpy as np
from general_qec.gates import sigma_x, sigma_y, sigma_z
from general_qec.gates import adj_CNOT, flipped_adj_CNOT, small_non_adj_CNOT
from general_qec.gates import non_adj_CNOT, flipped_non_adj_CNOT, CNOT
from general_qec.gates import cnot, flipped_cnot
from general_qec.gates import CZ
from general_qec.qec_helpers import one, zero

LOGGER = logging.getLogger(__name__)


class TestGates(unittest.TestCase): # pylint: disable=too-many-instance-attributes
    """Tests for the `gates` module."""

    def setUp(self) -> None:
        self.three_qubit000 = np.array([[1], [0], [0], [0], [0], [0], [0], [0]])
        self.three_qubit001 = np.array([[0], [1], [0], [0], [0], [0], [0], [0]])
        self.three_qubit010 = np.array([[0], [0], [1], [0], [0], [0], [0], [0]])
        self.three_qubit011 = np.array([[0], [0], [0], [1], [0], [0], [0], [0]])
        self.three_qubit100 = np.array([[0], [0], [0], [0], [1], [0], [0], [0]])
        self.three_qubit101 = np.array([[0], [0], [0], [0], [0], [1], [0], [0]])
        self.three_qubit110 = np.array([[0], [0], [0], [0], [0], [0], [1], [0]])
        self.three_qubit111 = np.array([[0], [0], [0], [0], [0], [0], [0], [1]])

        return super().setUp()

    def test_commutators(self):
        """Test commutator relationships for Pauli matrices"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertTrue(np.all(
            np.matmul(sigma_x, sigma_y) - np.matmul(sigma_y, sigma_x) == -2j*sigma_z
        ))
        self.assertTrue(np.all(
            np.matmul(sigma_y, sigma_z) - np.matmul(sigma_z, sigma_y) == -2j*sigma_x
        ))
        self.assertTrue(np.all(
            np.matmul(sigma_z, sigma_x) - np.matmul(sigma_x, sigma_z) == -2j*sigma_y
        ))

    def test_cnot(self):
        """Test the various CNOT functions"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertEqual(adj_CNOT(0, 1, 4).shape, (16, 16))
        self.assertEqual(adj_CNOT(2, 3, 4).shape, (16, 16))
        self.assertTrue(np.all(
            np.matmul(adj_CNOT(0, 1, 3), self.three_qubit100) == \
                self.three_qubit110
        ))
        self.assertTrue(np.all(
            np.matmul(adj_CNOT(1, 2, 3), self.three_qubit011) == \
                self.three_qubit010
        ))
        self.assertEqual(flipped_adj_CNOT(1, 0, 4).shape, (16, 16))
        self.assertEqual(flipped_adj_CNOT(3, 2, 4).shape, (16, 16))
        self.assertTrue(np.all(
            np.matmul(flipped_adj_CNOT(1, 0, 3), self.three_qubit010) == \
                self.three_qubit110
        ))
        self.assertTrue(np.all(
            np.matmul(flipped_adj_CNOT(2, 1, 3), self.three_qubit111) == \
                self.three_qubit101
        ))
        self.assertEqual(small_non_adj_CNOT().shape, (8, 8))
        self.assertEqual(non_adj_CNOT(0, 2, 4).shape, (16, 16))
        self.assertEqual(non_adj_CNOT(1, 3, 4).shape, (16, 16))
        self.assertTrue(np.all(
            np.matmul(non_adj_CNOT(0, 2, 3), self.three_qubit110) == \
                self.three_qubit111
        ))
        self.assertTrue(np.all(
            np.matmul(flipped_non_adj_CNOT(2, 0, 3), self.three_qubit001) == \
                self.three_qubit101
        ))
        self.assertTrue(np.all(
            CNOT(0, 1, 2) == cnot
        ))
        self.assertTrue(np.all(
            CNOT(1, 0, 2) == flipped_cnot
        ))
        self.assertTrue(np.all(
            adj_CNOT(2, 3, 4) == CNOT(2, 3, 4)
        ))
        self.assertTrue(np.all(
            flipped_adj_CNOT(2, 1, 3) == CNOT(2, 1, 3)
        ))
        self.assertTrue(np.all(
            non_adj_CNOT(0, 3, 5) == CNOT(0, 3, 5)
        ))
        self.assertTrue(np.all(
            flipped_non_adj_CNOT(4, 0, 5) == CNOT(4, 0, 5)
        ))

    def test_cz(self):
        """Test CZ from `gates` module."""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertEqual(CZ(0, 1, 4).shape, (16, 16))
        self.assertEqual(CZ(3, 0, 4).shape, (16, 16))
        self.assertTrue(np.allclose(
            np.matmul(CZ(0, 1, 3), self.three_qubit100), self.three_qubit100
        ))
        self.assertTrue(np.allclose(
            np.matmul(CZ(0, 2, 3), self.three_qubit101), -1.0 * self.three_qubit101
        ))
        self.assertTrue(np.allclose(
            np.matmul(CZ(2, 0, 3), self.three_qubit101), -1.0 * self.three_qubit101
        ))
        self.assertTrue(np.allclose(
            np.matmul(CZ(0, 1, 3), self.three_qubit111), -1.0 * self.three_qubit111
        ))
        self.assertTrue(np.all(
            CZ(1, 0, 2) == CZ(0, 1, 2)
        ))
        self.assertTrue(np.all(
            CZ(2, 3, 4) == CZ(3, 2, 4)
        ))

    def test_distant_nonadj_cnot(self):
        """Test distant non-adjacent CNOT gates"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        # larger system
        orig_psi = np.kron(np.kron(np.kron(np.kron(np.kron(zero, one), zero), zero), zero), one)
        targ_psi = np.kron(np.kron(np.kron(np.kron(np.kron(one, one), zero), zero), zero), one)
        result = np.matmul(CNOT(5, 0, 6), orig_psi)
        self.assertTrue(np.all(result == targ_psi))
        # larger system
        orig_psi = np.kron(np.kron(np.kron(np.kron(np.kron(zero, one), zero), zero), zero), one)
        targ_psi = np.kron(np.kron(np.kron(np.kron(np.kron(zero, one), zero), zero), one), one)
        result = np.matmul(CNOT(1, 4, 6), orig_psi)
        self.assertTrue(np.all(result == targ_psi))


if __name__ == '__main__':
    unittest.main()
