'''
Usage:
    python test_steane_helpers_13q.py
'''
import unittest
import random
import logging
import sys
import numpy as np
from general_qec.qec_helpers import one, zero
from general_qec.qec_helpers import ancilla_reset
from general_qec.qec_helpers import bit_flip_error
from general_qec.qec_helpers import phase_flip_error
from circuit_specific.steane_helpers import initialize_larger_steane_code
from circuit_specific.steane_helpers import simultaneous_steane_code
from circuit_specific.steane_helpers import steane_dataq_logical_zero
from circuit_specific.steane_helpers import steane_dataq_logical_one

LOGGER = logging.getLogger(__name__)

ANCILLA_6ZERO = \
    np.kron(zero,
    np.kron(zero,
    np.kron(zero,
    np.kron(zero,
    np.kron(zero, zero)
))))

ZERO_STATE7Q = \
    np.kron(zero,
    np.kron(zero,
    np.kron(zero,
    np.kron(zero,
    np.kron(zero,
    np.kron(zero, zero)
)))))
ONE_STATE7Q = \
    np.kron(one,
    np.kron(one,
    np.kron(one,
    np.kron(one,
    np.kron(one,
    np.kron(one, one)
)))))
SUPERPOS_STATE7Q = (1. / np.sqrt(2.0)) * (ZERO_STATE7Q + ONE_STATE7Q)


class Test13QSteaneCode(unittest.TestCase):
    """Tests for Steane code functions for the 7+6 qubit system."""

    def test_13q_phase_and_bit_flip_error_correction(self):
        """Test 13-qubit Steane initialization, phase, and bit flip error corrections"""
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        n_qubits = 7  # number of data qubits in our system
        n_ancilla = 6 # for "simultaneous" Steane
        n_qtotal = n_qubits + n_ancilla
        # -
        random.seed(10)
        initialized_zero_state = initialize_larger_steane_code(ZERO_STATE7Q)
        self.assertEqual(initialized_zero_state.shape, (2**n_qtotal,))
        initialized_zero_state = ancilla_reset(initialized_zero_state, n_ancilla)
        self.assertTrue(
            np.allclose(initialized_zero_state, np.kron(steane_dataq_logical_zero(), ANCILLA_6ZERO))
        )
        # too slow for multiple tests - phase flip
        phase_error_state, error_index = phase_flip_error(initialized_zero_state, n_qubits)
        self.assertEqual(error_index, 5)
        corrected_state = simultaneous_steane_code(phase_error_state)
        corrected_state = ancilla_reset(corrected_state, n_ancilla)
        self.assertTrue(
            np.allclose(initialized_zero_state, corrected_state) or
            np.allclose(initialized_zero_state, -1.0 * corrected_state)
        )
        # -
        random.seed(11)
        initialized_one_state = initialize_larger_steane_code(ONE_STATE7Q)
        self.assertEqual(initialized_one_state.shape, (2**n_qtotal,))
        initialized_one_state = ancilla_reset(initialized_one_state, n_ancilla)
        self.assertTrue(
            np.allclose(initialized_one_state, np.kron(steane_dataq_logical_one(), ANCILLA_6ZERO))
        )
        # too slow for multiple tests - bit flip
        bit_error_state, error_index = bit_flip_error(initialized_one_state, n_qubits)
        self.assertEqual(error_index, 6)
        corrected_state = simultaneous_steane_code(bit_error_state)
        corrected_state = ancilla_reset(corrected_state, n_ancilla)
        self.assertTrue(
            np.allclose(initialized_one_state, corrected_state) or
            np.allclose(initialized_one_state, -1.0 * corrected_state)
        )


if __name__ == '__main__':
    unittest.main()
