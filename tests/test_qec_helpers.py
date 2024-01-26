'''
Usage:
    python test_qec_helpers.py
'''
import unittest
import random
import logging
import sys
import numpy as np
from general_qec.qec_helpers import one, zero, superpos
from general_qec.gates import sigma_y, cnot, rx_theta
from general_qec.qec_helpers import ancilla_reset
from general_qec.qec_helpers import collapse_ancilla
from general_qec.qec_helpers import collapse_dm
from general_qec.qec_helpers import remove_small_values
from general_qec.qec_helpers import vector_state_to_bit_state
from general_qec.qec_helpers import CNOT_gate_tot

LOGGER = logging.getLogger(__name__)


class TestHelpers(unittest.TestCase):
    """Tests for the `qec_helpers` module."""

    def test_vector_state_to_bit_state(self):
        """Tests for the `vector_state_to_bit_state()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        test_state1 = np.kron(one, zero)
        log_bit, index_of_element, logical_state = vector_state_to_bit_state(test_state1, 2)
        self.assertEqual(log_bit.shape, (1,))
        self.assertEqual(index_of_element.shape, (1,))
        self.assertEqual(log_bit[0], '10')
        self.assertEqual(index_of_element[0], 2.0)
        self.assertAlmostEqual(logical_state[2], 1.0)
        test_state2 = np.kron(superpos, superpos)
        log_bit, index_of_element, logical_state = vector_state_to_bit_state(test_state2, 2)
        self.assertEqual(log_bit.shape, (4,))
        self.assertEqual(index_of_element.shape, (4,))
        self.assertEqual(log_bit[3], '11')
        self.assertEqual(index_of_element[2], 2.0)
        self.assertAlmostEqual(logical_state[1], 0.5)

    def test_ancilla_functions(self):
        """Tests for various ancilla manipulation functions"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        ## --
        test_state = np.kron(np.kron(one, superpos), one)
        gate = np.kron(np.identity(2**2), rx_theta(np.pi/3))
        test_state = np.dot(gate, test_state)
        gate = np.kron(rx_theta(np.pi/6), np.identity(2**2))
        test_state = np.dot(gate, test_state)
        random.seed(15)  # fix the collapsed state
        collapsed_vector_state = collapse_ancilla(test_state, 1)
        collapsed_bits, _, _ = vector_state_to_bit_state(collapsed_vector_state, 3)
        rho = np.outer(collapsed_vector_state, collapsed_vector_state.conj().T)
        self.assertAlmostEqual(np.sum(np.abs(collapsed_vector_state)**2), 1.0)
        self.assertEqual(collapsed_vector_state.shape, (8,))
        self.assertTrue(
            np.all([np.isclose(collapsed_vector_state[i], -0.353553j) for i in [1, 3]])
        )
        self.assertAlmostEqual(np.trace(rho), 1.0)
        self.assertTrue(set(['001', '011', '101', '111']) == set(collapsed_bits))
        reset_state = ancilla_reset(collapsed_vector_state, 1)
        self.assertEqual(reset_state.shape, (8,))
        self.assertTrue(
            np.all([np.isclose(reset_state[i], -0.353553j) for i in [0, 2]])
        )
        rho = np.outer(collapsed_vector_state, collapsed_vector_state.conj().T)
        self.assertAlmostEqual(np.trace(rho), 1.0)
        # --
        three_qubit = np.kron(np.kron(zero, zero), zero)
        test_state = np.kron(np.kron(three_qubit, superpos), superpos)
        random.seed(11)  # fix the collapsed state
        collapsed_vector_state = collapse_ancilla(test_state, 2)
        self.assertAlmostEqual(np.sum(np.abs(collapsed_vector_state)**2), 1.0)
        self.assertEqual(collapsed_vector_state.shape, (32,))
        self.assertAlmostEqual(collapsed_vector_state[1], 1.0+0.0j)
        rho = np.outer(collapsed_vector_state, collapsed_vector_state.conj().T)
        self.assertAlmostEqual(np.trace(rho), 1.0)
        reset_state = ancilla_reset(collapsed_vector_state, 2)
        self.assertEqual(reset_state.shape, (32,))
        self.assertAlmostEqual(reset_state[0], 1.0+0.0j)
        rho = np.outer(reset_state, reset_state.conj().T)
        self.assertAlmostEqual(np.trace(rho), 1.0)
        # --
        three_qubit = np.kron(np.kron(superpos, superpos), superpos)
        gate = np.kron(np.kron(np.identity(2), sigma_y), np.identity(2))
        three_qubit = np.dot(gate, three_qubit)
        test_state = np.kron(np.kron(three_qubit, superpos), superpos)
        random.seed(12)  # fix the collapsed state -> ancilla to 01
        collapsed_vector_state = collapse_ancilla(test_state, 2)
        self.assertAlmostEqual(np.sum(np.abs(collapsed_vector_state)**2), 1.0)
        _, collapsed_indices, _ = vector_state_to_bit_state(collapsed_vector_state, 5)
        self.assertTrue(set([1, 5, 9, 13, 17, 21, 25, 29]) == set(collapsed_indices))
        rho = np.outer(collapsed_vector_state, collapsed_vector_state.conj().T)
        self.assertAlmostEqual(np.trace(rho), 1.0)
        reset_state = ancilla_reset(collapsed_vector_state, 2)
        _, reset_indices, _ = vector_state_to_bit_state(reset_state, 5)
        self.assertTrue(set([0, 4, 8, 12, 16, 20, 24, 28]) == set(reset_indices))
        rho = np.outer(reset_state, reset_state.conj().T)
        self.assertAlmostEqual(np.trace(rho), 1.0)
        # --
        three_qubit = np.kron(np.kron(superpos, superpos), one)
        gate = np.kron(np.identity(2**2), sigma_y)
        three_qubit = np.dot(gate, three_qubit)
        gate = np.kron(cnot, np.identity(2))
        three_qubit = np.dot(gate, three_qubit)
        test_state = np.kron(np.kron(np.kron(three_qubit, superpos), superpos), one)
        random.seed(13)  # fix the collapsed state -> ancilla to 011
        collapsed_vector_state = collapse_ancilla(test_state, 3)
        collapsed_bits, collapsed_indices, _ = vector_state_to_bit_state(collapsed_vector_state, 6)
        self.assertTrue(set(['000011', '010011', '100011', '110011']) == set(collapsed_bits))
        self.assertTrue(set([3, 19, 35, 51]) == set(collapsed_indices))
        rho = np.outer(collapsed_vector_state, collapsed_vector_state.conj().T)
        self.assertAlmostEqual(np.trace(rho), 1.0)
        reset_state = ancilla_reset(collapsed_vector_state, 3)
        reset_bits, _, _ = vector_state_to_bit_state(reset_state, 6)
        self.assertTrue(set(['000000', '010000', '100000', '110000']) == set(reset_bits))
        rho = np.outer(reset_state, reset_state.conj().T)
        self.assertAlmostEqual(np.trace(rho), 1.0)

    def test_remove_small_values(self):
        """Tests for `remove_small_values()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        x = np.array([1, 1e-16, 1e-16, 1e-16])      # pylint: disable=invalid-name
        y = np.array([1, 0, 0, 0])                  # pylint: disable=invalid-name
        self.assertTrue(np.all(remove_small_values(x) == y))
        self.assertTrue(np.all(remove_small_values(x, tolerance=1e-17) == x))

    def test_cnot_gate_tot(self):
        """Tests for `CNOT_gate_tot()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        self.assertEqual(CNOT_gate_tot(5, 3), 6)
        self.assertEqual(CNOT_gate_tot(3, 7), 14)

    def test_collapse_dm(self):
        """Tests for `collapse_dm()`"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(13)  # fix the collapsed state
        initial_state = np.kron(superpos, superpos)
        initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)
        collapsed_state = collapse_dm(initial_rho)
        self.assertTrue(np.all(collapsed_state == np.array([0, 1, 0, 0])))
        random.seed(13)  # fix the collapsed state
        initial_state = np.kron(np.kron(zero, superpos), one)
        initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)
        collapsed_state = collapse_dm(initial_rho)
        self.assertEqual(collapsed_state[1], 1)
        self.assertEqual(np.sum(collapsed_state), 1)
        random.seed(13)  # fix the collapsed state
        initial_state = np.kron(np.matmul(sigma_y, zero), superpos)
        initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)
        collapsed_state = collapse_dm(initial_rho)
        self.assertEqual(collapsed_state[2], 1)
        self.assertEqual(np.sum(collapsed_state), 1)


if __name__ == '__main__':
    unittest.main()
