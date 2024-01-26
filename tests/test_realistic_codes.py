'''
Usage:
    python test_realistic_codes.py
'''
import unittest
import logging
import sys
import random
import numpy as np
from general_qec.errors import random_qubit_x_error
from general_qec.gates import cnot
from general_qec.qec_helpers import zero, one, superpos
from general_qec.qec_helpers import collapse_dm
from circuit_specific.realistic_three_qubit import initialize_three_qubit_realisitc
from circuit_specific.realistic_three_qubit import three_qubit_realistic

LOGGER = logging.getLogger(__name__)


class TestRealisticThreeQubit(unittest.TestCase):
    """Tests for the `realistic_three_qubit` module."""

    def test_three_qubit_realistic_full(self):
        """Test of `three_qubit_realistic()` with RAD and Krauss errors"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(10)
        initial_psi = one # initialize our psi
        # timing parameters in microseconds
        t1 = 200 * 10**-6 # pylint: disable=invalid-name
        t2 = 150 * 10**-6 # pylint: disable=invalid-name
        tg = 20 * 10**-9  # pylint: disable=invalid-name
        # probability of gate error for each of five qubits
        krauss_probs = [0.001] * 5
        # state preparation and measurement errors
        spam_prob = 0.001
        # initialize the circuit
        initial_rho = initialize_three_qubit_realisitc(
            initial_psi, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        # 5 qubits initialized into |11100>
        initial_state = collapse_dm(initial_rho)
        self.assertTrue(initial_rho.shape, (2**5, 2**5))
        self.assertEqual(
            np.unravel_index(initial_rho.argmax(), initial_rho.shape), (0b11100, 0b11100)
        )
        # apply the 3 qubit circuit to case with no errors
        rho = three_qubit_realistic(
            initial_rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        # errors are low, so most probable state is the same
        self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))
        # throw some errors: qubit 0 -> 01100, qubit 1 -> 10100, qubit 2 -> 11000
        for (qubit, state) in [(0, 0b01100), (1, 0b10100), (2, 0b11000)]:
            errored_state, _ = random_qubit_x_error(initial_state, (qubit, qubit))
            self.assertEqual(np.argmax(errored_state), state)
            rho = np.outer(errored_state, errored_state.conj().T)
            # repair it
            rho = three_qubit_realistic(
                rho, t1=t1, t2=t2, tg=tg,
                qubit_error_probs=krauss_probs, spam_prob=spam_prob
            )
            self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))

    def test_three_qubit_realistic_noerr_superpos(self):
        """Test of `three_qubit_realistic()` with no errors"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        initial_psi = superpos # initialize our psi
        # timing parameters in microseconds
        t1 = None # pylint: disable=invalid-name
        t2 = None # pylint: disable=invalid-name
        tg = None # pylint: disable=invalid-name
        # probability of gate error for each of five qubits
        krauss_probs = [0.0] * 5
        # state preparation and measurement errors
        spam_prob = 0.0
        # initialize the circuit
        initial_rho = initialize_three_qubit_realisitc(
            initial_psi, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        # 5 qubits initialized into |11100>
        initial_state = np.kron(np.kron(np.kron(np.kron(superpos, zero), zero), zero), zero)
        op1 = np.kron(cnot, np.identity(2**3))
        initial_state = np.dot(op1, initial_state)
        op2 = np.kron(np.kron(np.identity(2), cnot), np.identity(2**2))
        initial_state = np.dot(op2, initial_state)
        direct_rho = np.outer(initial_state, initial_state.conj().T)
        self.assertTrue(initial_rho.shape, (2**5, 2**5))
        self.assertTrue(np.allclose(direct_rho, initial_rho))
        # apply the 3 qubit circuit to case with no errors
        rho = three_qubit_realistic(
            initial_rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        self.assertTrue(np.allclose(rho, initial_rho))
        # throw some errors: qubit 0 -> 01100, qubit 1 -> 10100, qubit 2 -> 11000
        for qubit in range(3):
            errored_state, _ = random_qubit_x_error(initial_state, (qubit, qubit))
            rho = np.outer(errored_state, errored_state.conj().T)
            # repair it
            rho = three_qubit_realistic(
                rho, t1=t1, t2=t2, tg=tg,
                qubit_error_probs=krauss_probs, spam_prob=spam_prob
            )
            self.assertTrue(np.allclose(rho, initial_rho))

    def test_three_qubit_realistic_rad(self):
        """Test of `three_qubit_realistic()` with RAD errors"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(10)
        initial_psi = one # initialize our psi
        # timing parameters in microseconds
        t1 = 200 * 10**-6 # pylint: disable=invalid-name
        t2 = 150 * 10**-6 # pylint: disable=invalid-name
        tg = 20 * 10**-9  # pylint: disable=invalid-name
        # probability of gate error for each of five qubits
        krauss_probs = None
        # state preparation and measurement errors
        spam_prob = 0.001
        # initialize the circuit
        rho = initialize_three_qubit_realisitc(
            initial_psi, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        self.assertTrue(rho.shape, (2**5, 2**5))
        self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))
        # apply the 3 qubit circuit
        rho = three_qubit_realistic(
            rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        # errors are low, so most probable state is the same
        self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))
        collapsed_state = collapse_dm(rho)
        # throw an x error on the data qubits
        errored_state, _ = random_qubit_x_error(collapsed_state, (1,1))
        self.assertEqual(np.argmax(errored_state), 0b10100)
        rho = np.outer(errored_state, errored_state.conj().T)
        # repair it
        rho = three_qubit_realistic(
            rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))

    def test_three_qubit_realistic_krauss(self):
        """Test of `three_qubit_realistic()` with Krauss errors"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(10)
        initial_psi = one # initialize our psi
        # timing parameters in microseconds
        t1 = None # pylint: disable=invalid-name
        t2 = None # pylint: disable=invalid-name
        tg = None # pylint: disable=invalid-name
        # probability of gate error for each of five qubits
        krauss_probs = [0.001] * 5
        # state preparation and measurement errors
        spam_prob = 0.001
        # initialize the circuit
        rho = initialize_three_qubit_realisitc(
            initial_psi, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        self.assertTrue(rho.shape, (2**5, 2**5))
        self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))
        # apply the 3 qubit circuit
        rho = three_qubit_realistic(
            rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        # errors are low, so most probable state is the same
        self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))
        collapsed_state = collapse_dm(rho)
        # throw an x error on the data qubits
        errored_state, _ = random_qubit_x_error(collapsed_state, (1,1))
        self.assertEqual(np.argmax(errored_state), 0b10100)
        rho = np.outer(errored_state, errored_state.conj().T)
        # repair it
        rho = three_qubit_realistic(
            rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))

    def test_three_qubit_realistic_error_free(self):
        """Test of `three_qubit_realistic()` with Krauss errors"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(10)
        initial_psi = one # initialize our psi
        # timing parameters in microseconds
        t1 = None # pylint: disable=invalid-name
        t2 = None # pylint: disable=invalid-name
        tg = None # pylint: disable=invalid-name
        # probability of gate error for each of five qubits
        krauss_probs = None
        # state preparation and measurement errors
        spam_prob = None
        # initialize the circuit
        rho = initialize_three_qubit_realisitc(
            initial_psi, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        self.assertTrue(rho.shape, (2**5, 2**5))
        self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))
        # apply the 3 qubit circuit
        rho = three_qubit_realistic(
            rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        # errors are low, so most probable state is the same
        self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))
        collapsed_state = collapse_dm(rho)
        # throw an x error on the data qubits
        errored_state, _ = random_qubit_x_error(collapsed_state, (1,1))
        self.assertEqual(np.argmax(errored_state), 0b10100)
        rho = np.outer(errored_state, errored_state.conj().T)
        # repair it
        rho = three_qubit_realistic(
            rho, t1=t1, t2=t2, tg=tg,
            qubit_error_probs=krauss_probs, spam_prob=spam_prob
        )
        self.assertEqual(np.unravel_index(rho.argmax(), rho.shape), (0b11100, 0b11100))

if __name__ == '__main__':
    unittest.main()
