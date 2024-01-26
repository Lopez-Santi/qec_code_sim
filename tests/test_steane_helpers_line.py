'''
Usage:
    python test_steane_helpers_line.py
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
from circuit_specific.steane_helpers import steane_dataq_logical_zero
from circuit_specific.steane_helpers import steane_dataq_logical_one
from circuit_specific.steane_helpers import steane_dataq_logical_superpos
from circuit_specific.steane_helpers import initialize_steane_line_conn
from circuit_specific.steane_helpers import steane_line_conn_phase_correction
from circuit_specific.steane_helpers import steane_line_conn_bit_correction

LOGGER = logging.getLogger(__name__)

ANCILLA_3ZERO = \
    np.kron(zero,
    np.kron(zero, zero)
)

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


class TestSteaneLineConnectivityCode(unittest.TestCase):
    """Tests for Steane code functions for the 7+3 qubit system with line connectivity."""

    def setUp(self) -> None:
        self.n_qubits = 7  # number of data qubits in our system
        self.n_ancilla = 3 # for "simultaneous" Steane
        self.n_qtotal = self.n_qubits + self.n_ancilla

    def test_line_phase_and_bit_flip_error_correction_zero_state(self):
        """
        Test Line Connected Steane initialization, phase, and bit flip error corrections for
        logical zero
        """
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(11)
        # -
        initialized_zero_state = initialize_steane_line_conn(ZERO_STATE7Q)
        initialized_zero_state = ancilla_reset(initialized_zero_state, self.n_ancilla)
        self.assertTrue(
            np.allclose(initialized_zero_state, np.kron(steane_dataq_logical_zero(), ANCILLA_3ZERO))
        )
        # try 5 random phase flips (equal chance of any or no qubits)
        for _ in range(5):
            phase_error_state = phase_flip_error(initialized_zero_state, self.n_qubits)[0]
            corrected_state = steane_line_conn_phase_correction(phase_error_state)
            corrected_state = ancilla_reset(corrected_state, self.n_ancilla)
            self.assertTrue(np.allclose(initialized_zero_state, corrected_state))
        # try 5 random bit flips (equal chance of any or no qubits)
        for _ in range(5):
            bit_error_state = bit_flip_error(initialized_zero_state, self.n_qubits)[0]
            corrected_state = steane_line_conn_bit_correction(bit_error_state)
            corrected_state = ancilla_reset(corrected_state, self.n_ancilla)
            self.assertTrue(np.allclose(initialized_zero_state, corrected_state))

    def test_line_phase_and_bit_flip_error_correction_one_state(self):
        """
        Test Line Connected Steane initialization, phase, and bit flip error corrections for
        logical one
        """
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(11)
        # -
        initialized_one_state = initialize_steane_line_conn(ONE_STATE7Q)
        initialized_one_state = ancilla_reset(initialized_one_state, self.n_ancilla)
        self.assertTrue(
            np.allclose(initialized_one_state, np.kron(steane_dataq_logical_one(), ANCILLA_3ZERO))
        )
        # try 5 random phase flips (equal chance of any or no qubits)
        for _ in range(5):
            phase_error_state = phase_flip_error(initialized_one_state, self.n_qubits)[0]
            corrected_state = steane_line_conn_phase_correction(phase_error_state)
            corrected_state = ancilla_reset(corrected_state, self.n_ancilla)
            self.assertTrue(np.allclose(initialized_one_state, corrected_state))
        # try 5 random bit flips (equal chance of any or no qubits)
        for _ in range(5):
            bit_error_state = bit_flip_error(initialized_one_state, self.n_qubits)[0]
            corrected_state = steane_line_conn_bit_correction(bit_error_state)
            corrected_state = ancilla_reset(corrected_state, self.n_ancilla)
            self.assertTrue(np.allclose(initialized_one_state, corrected_state))

    def test_line_phase_and_bit_flip_error_correction_superpos_state(self):
        """
        Test Line Connected Steane initialization, phase, and bit flip error corrections for
        logical superpos
        """
        LOGGER.info(sys._getframe().f_code.co_name) # pylint: disable=protected-access
        random.seed(12)
        # -
        initialized_superpos_state = initialize_steane_line_conn(SUPERPOS_STATE7Q)
        initialized_superpos_state = ancilla_reset(initialized_superpos_state, self.n_ancilla)
        self.assertTrue(
            np.allclose(initialized_superpos_state,
                        np.kron(steane_dataq_logical_superpos(), ANCILLA_3ZERO))
        )
        # do a phase flip AND a bit flip
        phase_error_state, phase_index = phase_flip_error(initialized_superpos_state, self.n_qubits)
        bit_phase_error_state, bit_index = bit_flip_error(phase_error_state, self.n_qubits)
        # could alternatively do a while loop to avoid random seed portability issues
        self.assertTrue(phase_index != bit_index,
                        msg="Check random seed; need different qubit indices here.")
        corrected_state = steane_line_conn_phase_correction(bit_phase_error_state)
        corrected_state = ancilla_reset(corrected_state, self.n_ancilla)
        corrected_state = steane_line_conn_bit_correction(corrected_state)
        corrected_state = ancilla_reset(corrected_state, self.n_ancilla)
        self.assertTrue(np.allclose(initialized_superpos_state, corrected_state))


if __name__ == '__main__':
    unittest.main()
