"""
This file will contain functions that are useful when implementing logical t1
testing for the 9 qubit code from error models.
"""
import random

import numpy as np
from general_qec.qec_helpers import *
from general_qec.gates import *
from general_qec.errors import *
from general_qec.errors import *

# Masurement operators for individual qubits
zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
one_meas = np.kron(one, one[np.newaxis].conj().T)


def initialize_nine_qubit_realisitc(            # pylint: disable=too-many-arguments
        initial_psi, t1=None, t2=None, tg=None,  # pylint: disable=invalid-name
        qubit_error_probs=None, spam_prob=None
    ):
    """
    Initialize a 9 qubit logical state "realistically" and return the density
    matrix and state.

    * initial_psi: a valid single qubit state vector
    """
    # Initialize the 9 qubit logical state and 2 qubit ancilla pair

    initial_ancilla_state = zero

    # Here we can initialize the 9 qubit logical state
    initial_state = np.kron(initial_psi, np.kron(initial_ancilla_state, np.kron(initial_ancilla_state, np.kron(
            initial_ancilla_state, np.kron(initial_ancilla_state, np.kron(initial_ancilla_state, np.kron(
                initial_ancilla_state,np.kron(initial_ancilla_state, np.kron(initial_ancilla_state, np.kron(
                    initial_ancilla_state, initial_ancilla_state))))))))))


    n = 11 # pylint: disable=invalid-name

    # Find the density matrix of our logical system
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    if spam_prob is not None:
        for index in range(n):
            initial_rho = spam_error(initial_rho, spam_prob, index)

    # 3 Hadamard Gates (qubits 0, 3, 6)
    h_gates = np.kron(hadamard, np.kron(np.identity(2**2), np.kron(hadamard, np.kron(
        np.identity(2**2), np.kron(hadamard, np.identity(2**4)))))
    )

    # Apply the Initialization Circuit (CNOT gates and Hadamard gates) to the kronecker
    # product of the current 9 qubit density matrix
    if (qubit_error_probs is not None) and \
            ((t1 is not None) and (t2 is not None) and (tg is not None)):
        # first CNOT gate
        rho = prob_line_rad_CNOT(
            initial_rho, 0, 3, t1, t2, tg, qubit_error_probs, form = 'rho'
        )
        # second CNOT gate
        rho = prob_line_rad_CNOT(
            rho, 1, 6, t1, t2, tg, qubit_error_probs, form = 'rho'
        )
        rho = np.dot(h_gates, np.dot(rho, h_gates.conj().T))
        # apply depolarizing error
        rho = gate_error(rho, qubit_error_probs[0], 0, n)
        rho = gate_error(rho, qubit_error_probs[3], 3, n)
        rho = gate_error(rho, qubit_error_probs[6], 6, n)
        # apply rad error
        rho = rad_error(rho, t1, t2, tg)
        # CNOT Gates between qubits in same block (just like 3-qubit code)
        # Block 1
        # first CNOT gate
        rho = prob_line_rad_CNOT(
            initial_rho, 0, 1, t1, t2, tg, qubit_error_probs, form = 'rho'
        )
        # second CNOT gate
        rho = prob_line_rad_CNOT(
            rho, 1, 2, t1, t2, tg, qubit_error_probs, form = 'rho'
        )
        # Block 2
        # first CNOT gate
        rho = prob_line_rad_CNOT(
            initial_rho, 3, 4, t1, t2, tg, qubit_error_probs, form = 'rho'
        )
        # second CNOT gate
        rho = prob_line_rad_CNOT(
            rho, 4, 5, t1, t2, tg, qubit_error_probs, form = 'rho'
        )
        # Block 3
        # first CNOT gate
        rho = prob_line_rad_CNOT(
            initial_rho, 6, 7, t1, t2, tg, qubit_error_probs, form = 'rho'
        )
        # second CNOT gate
        rho = prob_line_rad_CNOT(
            rho, 7, 8, t1, t2, tg, qubit_error_probs, form = 'rho'
        )
    elif (qubit_error_probs is None) and \
            ((t1 is not None) and (t2 is not None) and (tg is not None)):
        # first CNOT gate
        rho = line_rad_CNOT(
            initial_rho, 0, 3, t1, t2, tg, form = 'rho'
        )
        # second CNOT gate
        rho = line_rad_CNOT(
            rho, 1, 6, t1, t2, tg, form = 'rho'
        )
        rho = np.dot(h_gates, np.dot(rho, h_gates.conj().T))
        # apply rad error
        rho = rad_error(rho, t1, t2, tg)
        # CNOT Gates between qubits in same block (just like 3-qubit code)
        # Block 1
        # first CNOT gate
        rho = line_rad_CNOT(
            initial_rho, 0, 1, t1, t2, tg, form = 'rho'
        )
        # second CNOT gate
        rho = line_rad_CNOT(
            rho, 1, 2, t1, t2, tg, form = 'rho'
        )
        # Block 2
        # first CNOT gate
        rho = line_rad_CNOT(
            initial_rho, 3, 4, t1, t2, tg, form = 'rho'
        )
        # second CNOT gate
        rho = line_rad_CNOT(
            rho, 4, 5, t1, t2, tg, form = 'rho'
        )
        # Block 3
        # first CNOT gate
        rho = line_rad_CNOT(
            initial_rho, 6, 7, t1, t2, tg, form = 'rho'
        )
        # second CNOT gate
        rho = line_rad_CNOT(
            rho, 7, 8, t1, t2, tg, form = 'rho'
        )
    elif (qubit_error_probs is not None) and \
            (t1 is None and t2 is None and tg is None):
         # first CNOT gate
        rho = line_errored_CNOT(
            initial_rho, 0, 3, qubit_error_probs, form = 'rho'
        )
        # second CNOT gate
        rho = line_errored_CNOT(
            rho, 1, 6, qubit_error_probs, form = 'rho'
        )
        rho = np.dot(h_gates, np.dot(rho, h_gates.conj().T))
        # apply depolarizing error
        rho = gate_error(rho, qubit_error_probs[0], 0, n)
        rho = gate_error(rho, qubit_error_probs[3], 3, n)
        rho = gate_error(rho, qubit_error_probs[6], 6, n)
        # CNOT Gates between qubits in same block (just like 3-qubit code)
        # Block 1
        # first CNOT gate
        rho = line_errored_CNOT(
            initial_rho, 0, 1, qubit_error_probs, form = 'rho'
        )
        # second CNOT gate
        rho = line_errored_CNOT(
            rho, 1, 2, qubit_error_probs, form = 'rho'
        )
        # Block 2
        # first CNOT gate
        rho = line_errored_CNOT(
            initial_rho, 3, 4, qubit_error_probs, form = 'rho'
        )
        # second CNOT gate
        rho = line_errored_CNOT(
            rho, 4, 5, qubit_error_probs, form = 'rho'
        )
        # Block 3
        # first CNOT gate
        rho = line_errored_CNOT(
            initial_rho, 6, 7, qubit_error_probs, form = 'rho'
        )
        # second CNOT gate
        rho = line_errored_CNOT(
            rho, 7, 8, qubit_error_probs, form = 'rho'
        )
    else:
        # Applying first two non-adjacent CNOT gates
        rho = np.dot(CNOT(0, 3, 11), np.dot(rho, CNOT(0, 3, 11).conj().T))
        rho = np.dot(CNOT(0, 6, 9), np.dot(rho, CNOT(0, 6, 11).conj().T))

        # Applying the three Hadamard gates
        rho = np.dot(h_gates, np.dot(rho, h_gates.conj().T))

        # The First set of 3 CNOT gates (applied to adjacent qubits)                                                                      
        rho = np.dot(CNOT(0, 1, 11), np.dot(rho, CNOT(0, 1, 11).conj().T))
        rho = np.dot(CNOT(3, 4, 11), np.dot(rho, CNOT(3, 4, 11).conj().T))
        rho = np.dot(CNOT(6, 7, 11) , np.dot(rho, CNOT(6, 7, 11).conj().T))

        # The First set of 3 CNOT gates (applied to non-adjacent qubits)  
        rho = np.dot(CNOT(0, 2, 11) , np.dot(rho, CNOT(0, 2, 11).conj().T))
        rho = np.dot(CNOT(3, 5, 11) , np.dot(rho, CNOT(3, 5, 11).conj().T))
        rho = np.dot(CNOT(6, 8, 11) , np.dot(rho, CNOT(6, 8, 11).conj().T))

    return rho


def nine_qubit_realistic(
        initial_rho, t1=None, t2=None, tg=None,
        qubit_error_probs=None, spam_prob=None
    ): # pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements
    """
    Implements the 9 qubit circuit with relaxation and dephasing errors, gate
    error probabilities, and spam errors.

    Outputs the logical state with reset ancilla after correction.

    * initial_rho: initial density matrix of your 11 qubit system
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * qubit_error_probs: an array of the probability for errors of each qubit in your system
    * spam_prob: The pobability that you have a state prep or measurement error
    Note: times are in seconds.
    """
    nqubits = 11
    apply_rad_errors = (t1 is not None) and (t2 is not None) and (tg is not None)
    apply_krauss_errors = qubit_error_probs is not None
    apply_spam_errors = spam_prob is not None
    # Define the measurement projection operators
    measure00 = np.kron(np.identity(2**9), np.kron(zero_meas, zero_meas))
    measure01 = np.kron(np.identity(2**9), np.kron(zero_meas, one_meas))
    measure10 = np.kron(np.identity(2**9), np.kron(one_meas, zero_meas))
    measure11 = np.kron(np.identity(2**9), np.kron(one_meas, one_meas))
    all_meas = np.array([measure00, measure01, measure10, measure11])

    # Phase Error Detection and Correction First
    detection_rho = initial_rho

    # Apply Hadamard to all data qubits for the first time
    for qubit in range(9):
        h_gate = np.kron(np.identity(2**(qubit)), np.kron(hadamard, np.identity(2**(11 - qubit - 1))))
        detection_rho = np.dot(h_gate, np.dot(detection_rho, h_gate.conj().T))
        if apply_rad_errors and apply_krauss_errors:
            # apply depolarizing error
            detection_rho = gate_error(detection_rho, qubit_error_probs[qubit], qubit, nqubits)
            # apply rad error
            detection_rho = rad_error(detection_rho, t1, t2, tg)
        elif apply_krauss_errors:
            # apply depolarizing error
            detection_rho = gate_error(detection_rho, qubit_error_probs[qubit], qubit, nqubits)
        elif apply_rad_errors:
            # apply rad error
            detection_rho = rad_error(detection_rho, t1, t2, tg)

    # Apply the CNOT gates needed to change the state of the first syndrome ancilla (qubit index 9)
    for control in range(6):
        if apply_rad_errors and apply_krauss_errors:
            detection_rho = prob_line_rad_CNOT(
                          detection_rho, control, 9, t1, t2, tg, qubit_error_probs, form = 'rho'
            )
        elif apply_krauss_errors:
            detection_rho = line_errored_CNOT(
                          detection_rho, control, 9, qubit_error_probs, form = 'rho'
            )
        elif apply_rad_errors:
            detection_rho = line_rad_CNOT(
                          detection_rho, control, 9, t1, t2, tg, qubit_error_probs, form = 'rho'
            )
        else:
            detection_rho = np.dot(CNOT(control, 9, nqubits), np.dot(
                          detection_rho, CNOT(control, 9, nqubits).conj().T)
            )

    # Apply the CNOT gates needed to change the state of the second syndrome ancilla (qubit index 10)
    for control in range(3, 9):
        if apply_rad_errors and apply_krauss_errors:
            detection_rho = prob_line_rad_CNOT(
                          detection_rho, control, 10, t1, t2, tg, qubit_error_probs, form = 'rho'
            )
        elif apply_krauss_errors:
            detection_rho = line_errored_CNOT(
                          detection_rho, control, 10, qubit_error_probs, form = 'rho'
            )
        elif apply_rad_errors:
            detection_rho = line_rad_CNOT(
                          detection_rho, control, 10, t1, t2, tg, qubit_error_probs, form = 'rho'
            )
        else:
            detection_rho = np.dot(CNOT(control, 10, nqubits), np.dot(
                          detection_rho, CNOT(control, 10, nqubits).conj().T)
            )

    # Apply Hadamard to all data qubits for the second time
    for qubit in range(9):
        h_gate = np.kron(np.identity(2**(qubit)), np.kron(hadamard, np.identity(2**(11 - qubit - 1))))
        detection_rho = np.dot(h_gate, np.dot(detection_rho, h_gate.conj().T))
        if apply_rad_errors and apply_krauss_errors:
            # apply depolarizing error
            detection_rho = gate_error(detection_rho, qubit_error_probs[qubit], qubit, nqubits)
            # apply rad error
            detection_rho = rad_error(detection_rho, t1, t2, tg)
        elif apply_krauss_errors:
            # apply depolarizing error
            detection_rho = gate_error(detection_rho, qubit_error_probs[qubit], qubit, nqubits)
        elif apply_rad_errors:
            # apply rad error
            detection_rho = rad_error(detection_rho, t1, t2, tg)

    # Measure the ancilla qubits
    # -- 1st, apply state measurement error if spam_probs is not empty
    if apply_spam_errors:
        detection_rho = spam_error(detection_rho, spam_prob, 9) # ancilla 0
        detection_rho = spam_error(detection_rho, spam_prob, 10) # ancilla 1

    # -- 2nd, find the probability to measure each case
    # ancilla in 00 -> suggests no errors -> error case00
    # ancilla in 01 -> error on data block 3 -> error case01
    # ancilla in 10 -> error on data block 1 -> error case10
    # ancilla in 11 -> error on data block 2 -> error case11
    error_case01, error_case10, error_case11 = 1, 2, 3
    m00_prob = np.trace(np.dot(measure00.conj().T, np.dot(measure00, detection_rho))).real
    m01_prob = np.trace(np.dot(measure01.conj().T, np.dot(measure01, detection_rho))).real
    m10_prob = np.trace(np.dot(measure10.conj().T, np.dot(measure10, detection_rho))).real
    m11_prob = np.trace(np.dot(measure11.conj().T, np.dot(measure11, detection_rho))).real
    all_probs = np.array([m00_prob, m01_prob, m10_prob, m11_prob])
    assert np.isclose(np.sum(all_probs), 1.0), "Invalid measurement space in nine_qubit_realistic"
    # -- 3rd, measure via probability weighted choice
    error_case = random.choices(
        list(range(len(all_probs))), weights=all_probs, k=1
    )[0]
    # -- 4th, measurement collapse of the density matrix
    detection_rho = np.dot(
        all_meas[error_case],
        np.dot(detection_rho, all_meas[error_case].conj().T)
    ) / (all_probs[error_case])

    # Apply correction gates based on ancilla measurements & reset ancilla qubits
    corrected_rho = detection_rho
    # -- 1st, find error index
    error_block_index = None
    qubit_corrected_index = None
    if error_case == error_case11:
        error_block_index = 2
        qubit_corrected_index = 3
        correction_gate = np.kron(np.identity(2**3), np.kron(sigma_z, np.identity(2**7)))
        corrected_rho = np.dot(
            correction_gate,
            np.dot(corrected_rho, correction_gate.conj().T)
        )
        ancilla_reset_gate = np.kron(np.kron(np.identity(2**9), sigma_x), sigma_x)
        corrected_rho = np.dot(
            ancilla_reset_gate,
            np.dot(corrected_rho, ancilla_reset_gate.conj().T)
        )
        if apply_krauss_errors:
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[9], 9, nqubits)
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[10], 10, nqubits)
    elif error_case == error_case10:
        error_block_index = 1
        qubit_corrected_index = 0
        correction_gate = np.kron(sigma_z, np.identity(2**10))
        corrected_rho = np.dot(
            correction_gate, np.dot(corrected_rho, correction_gate.conj().T)
        )
        ancilla_reset_gate = np.kron(np.kron(np.identity(2**9), sigma_x), np.identity(2))
        corrected_rho = np.dot(
            ancilla_reset_gate,
            np.dot(corrected_rho, ancilla_reset_gate.conj().T)
        )
        if apply_krauss_errors:
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[9], 9, nqubits)
    elif error_case == error_case01:
        error_block_index = 3
        qubit_corrected_index = 7
        correction_gate = np.kron(np.identity(2**6), np.kron(sigma_z, np.identity(2**4)))
        corrected_rho = np.dot(
            correction_gate, np.dot(corrected_rho, correction_gate.conj().T)
        )
        ancilla_reset_gate = np.kron(np.identity(2**10), sigma_x)
        corrected_rho = np.dot(
            ancilla_reset_gate,
            np.dot(corrected_rho, ancilla_reset_gate.conj().T)
        )
        if apply_krauss_errors:
            corrected_rho = gate_error(corrected_rho, qubit_error_probs[10], 10, nqubits)
    # -- 2nd, apply Krauss errors for state correction if appropriate
    if (error_block_index is not None) and apply_krauss_errors:
        corrected_rho = gate_error(
            corrected_rho, qubit_error_probs[qubit_corrected_index], qubit_corrected_index, nqubits
        )
    # -- 3rd, apply RAD errors -> assume data and ancilla qubit operations are in paralllel
    if (error_block_index is not None) and apply_rad_errors:
        corrected_rho = rad_error(corrected_rho, t1, t2, tg)

    # End of phase correction
    #############################

    detection_rho = corrected_rho
    # Now Bit Flip Error Detection and Correction
    # -- loop through the 3 qubit code 3 times (one for each block)
    
    # Define the measurement projection operators
    measure00 = np.kron(np.identity(2**9), np.kron(zero_meas, zero_meas))
    measure01 = np.kron(np.identity(2**9), np.kron(zero_meas, one_meas))
    measure10 = np.kron(np.identity(2**9), np.kron(one_meas, zero_meas))
    measure11 = np.kron(np.identity(2**9), np.kron(one_meas, one_meas))
    all_meas = np.array([measure00, measure01, measure10, measure11])

    # Using the 3 qubit code we can cycle through permutations of each block
    for block in range(3):
        val = 3 * block # constant added to the qubits (0, 1, 2) depending on block

        # Apply the CNOT gates needed to change the state of the syndrome ancilla
        detection_rho = initial_rho
        for (control, target) in [(0 + val, 9), (1 + val, 9), (0 + val, 10), (2 + val, 10)]:
            if apply_rad_errors and apply_krauss_errors:
                detection_rho = prob_line_rad_CNOT(
                    detection_rho, control, target, t1, t2, tg, qubit_error_probs, form='rho')
            elif apply_krauss_errors:
                detection_rho = line_errored_CNOT(
                    detection_rho, control, target, qubit_error_probs, form='rho')
            elif apply_rad_errors:
                detection_rho = line_rad_CNOT(
                    detection_rho, control, target, t1, t2, tg, form='rho')
            else:
                detection_rho = np.dot(CNOT(control, target, 11), np.dot(detection_rho, CNOT(target, control, 11).conj().T))
        
        # Measure the ancilla qubits
        # -- 1st, apply state measurement error if spam_probs is not empty
        if apply_spam_errors:
            detection_rho = spam_error(detection_rho, spam_prob, 9) # ancilla 0
            detection_rho = spam_error(detection_rho, spam_prob, 10) # ancilla 1

        # -- 2nd, find the probability to measure each case (in the block)
        # ancilla in 00 -> suggests no errors -> error case00
        # ancilla in 01 -> error on data qubit 2 -> error case01
        # ancilla in 10 -> error on data qubit 1 -> error case10
        # ancilla in 11 -> error on data qubit 0 -> error case11
        error_case01, error_case10, error_case11 = 1, 2, 3
        m00_prob = np.trace(np.dot(measure00.conj().T, np.dot(measure00, detection_rho))).real
        m01_prob = np.trace(np.dot(measure01.conj().T, np.dot(measure01, detection_rho))).real
        m10_prob = np.trace(np.dot(measure10.conj().T, np.dot(measure10, detection_rho))).real
        m11_prob = np.trace(np.dot(measure11.conj().T, np.dot(measure11, detection_rho))).real
        all_probs = np.array([m00_prob, m01_prob, m10_prob, m11_prob])
        assert np.isclose(np.sum(all_probs), 1.0), "Invalid measurement space in nine_qubit_realistic"
        # -- 3rd, measure via probability weighted choice
        error_case = random.choices(
            list(range(len(all_probs))), weights=all_probs, k=1
        )[0]
        # -- 4th, measurement collapse of the density matrix
        detection_rho = np.dot(
            all_meas[error_case],
            np.dot(detection_rho, all_meas[error_case].conj().T)
        ) / (all_probs[error_case])

        # Apply correction gates based on ancilla measurements & reset ancilla qubits
        corrected_rho = detection_rho
        # -- 1st, find error index
        error_qubit_index = None
        if error_case == error_case11:
            error_qubit_index = 0 + val
            correction_gate = np.kron(np.identity(2**(error_qubit_index)), np.kron(
                            sigma_x, np.identity(2**(11 - error_qubit_index - 1)))
            )
            corrected_rho = np.dot(
                correction_gate,
                np.dot(corrected_rho, correction_gate.conj().T)
            )
            ancilla_reset_gate = np.kron(np.kron(np.identity(2**9), sigma_x), sigma_x)
            corrected_rho = np.dot(
                ancilla_reset_gate,
                np.dot(corrected_rho, ancilla_reset_gate.conj().T)
            )
            if apply_krauss_errors:
                corrected_rho = gate_error(corrected_rho, qubit_error_probs[9], 9, nqubits)
                corrected_rho = gate_error(corrected_rho, qubit_error_probs[10], 10, nqubits)
        elif error_case == error_case10:
            error_qubit_index = 1 + val
            correction_gate = np.kron(np.identity(2**(error_qubit_index)), np.kron(
                            sigma_x, np.identity(2**(11 - error_qubit_index - 1)))
            )
            corrected_rho = np.dot(
                correction_gate, np.dot(corrected_rho, correction_gate.conj().T)
            )
            ancilla_reset_gate = np.kron(np.kron(np.identity(2**9), sigma_x), np.identity(2))
            corrected_rho = np.dot(
                ancilla_reset_gate,
                np.dot(corrected_rho, ancilla_reset_gate.conj().T)
            )
            if apply_krauss_errors:
                corrected_rho = gate_error(corrected_rho, qubit_error_probs[9], 9, nqubits)
        elif error_case == error_case01:
            error_qubit_index = 2 + val
            correction_gate = np.kron(np.identity(2**(error_qubit_index)), np.kron(
                            sigma_x, np.identity(2**(11 - error_qubit_index - 1)))
            )
            corrected_rho = np.dot(
                correction_gate, np.dot(corrected_rho, correction_gate.conj().T)
            )
            ancilla_reset_gate = np.kron(np.identity(2**10), sigma_x)
            corrected_rho = np.dot(
                ancilla_reset_gate,
                np.dot(corrected_rho, ancilla_reset_gate.conj().T)
            )
            if apply_krauss_errors:
                corrected_rho = gate_error(corrected_rho, qubit_error_probs[10], 10, nqubits)
        
        # -- 2nd, apply Krauss errors for state correction if appropriate
        if (error_qubit_index is not None) and apply_krauss_errors:
            corrected_rho = gate_error(
                corrected_rho, qubit_error_probs[error_qubit_index], error_qubit_index, nqubits
            )
        # -- 3rd, apply RAD errors -> assume data and ancilla qubit operations are in paralllel
        if (error_qubit_index is not None) and apply_rad_errors:
            corrected_rho = rad_error(corrected_rho, t1, t2, tg)

    # End of bit correction

    return corrected_rho
