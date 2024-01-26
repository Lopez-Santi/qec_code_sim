# The functions in this file are useful when implementing the nine qubit code

import numpy as np
import random
from general_qec.gates import *
from general_qec.qec_helpers import *

### Initializes the nine qubit logical state using an initial single qubit psi ###
def nine_qubit_initialize_logical_state(initial_psi):
    # initial_psi: initial state of your single qubit that you want to use as your logical state (2 x 1)

    initial_ancilla_state = np.array([1, 0])  # initializing the |0> state of the qubits

    # Initialize the 9 qubit logical state by using thr kronecker product
    initial_logical_state = np.kron(initial_psi, np.kron(initial_ancilla_state, np.kron(initial_ancilla_state, np.kron(
        initial_ancilla_state, np.kron(initial_ancilla_state, np.kron(initial_ancilla_state, np.kron(
            initial_ancilla_state,np.kron(initial_ancilla_state, initial_ancilla_state))))))))


    # Setting up the correct logical qubit state

    # Applying first two non-adjacent CNOT gates
    current_state = np.dot(CNOT(0, 3, 9), initial_logical_state)
    current_state = np.dot(CNOT(0, 6, 9), current_state)

    # Creating the three Hadamard gates
    hadamards = np.kron(hadamard, np.kron(np.identity(2**2), np.kron(hadamard, np.kron(
        np.identity(2**2), np.kron(hadamard, np.identity(2**2))))))

    # Applying the three Hadamard gates
    current_state = np.dot(hadamards, current_state)

    # The First set of 3 CNOT gates (applied to adjacent qubits)
    current_state = np.dot(CNOT(0, 1, 9), current_state)
    current_state = np.dot(CNOT(3, 4, 9), current_state)
    current_state = np.dot(CNOT(6, 7, 9) , current_state)

    # The First set of 3 CNOT gates (applied to non-adjacent qubits)
    current_state = np.dot(CNOT(0, 2, 9) , current_state)
    current_state = np.dot(CNOT(3, 5, 9) , current_state)
    current_state = np.dot(CNOT(6, 8, 9) , current_state)

    # Adding the two ancilla qubits using the kronecker product
    final_logical_state = np.kron(current_state, np.kron(np.array([1, 0]), np.array([1,0])))

    return final_logical_state


### - - - - - - Error Detection and Correction - - - - - - ###

### Detects where the z rotation error occured from the vector from of the 11 qubit system ###
def nine_qubit_phase_correction(logical_state):
    # logical_state: the full vector state representation of your 11 qubit system (9 data, 2 ancilla)

     # First Nine Hadamard gates
    hadamards = np.kron(hadamard, np.kron(hadamard, np.kron(hadamard, np.kron(
       hadamard, np.kron(hadamard, np.kron(hadamard, np.kron(
       hadamard, np.kron(hadamard, np.kron(hadamard, np.identity(2**2))))))))))

    current_state = np.dot(hadamards, logical_state)


    # First Set of 6 CNOTS (We use two sets to detect sign differences)
    current_state = np.dot(CNOT(0, 9, 11), current_state)
    current_state = np.dot(CNOT(1, 9, 11), current_state)
    current_state = np.dot(CNOT(2, 9, 11), current_state)
    current_state = np.dot(CNOT(3, 9, 11), current_state)
    current_state = np.dot(CNOT(4, 9, 11), current_state)
    current_state = np.dot(CNOT(5, 9, 11), current_state)


    # Second Set of 6 CNOTS
    current_state = np.dot(CNOT(3, 10, 11), current_state)
    current_state = np.dot(CNOT(4, 10, 11), current_state)
    current_state = np.dot(CNOT(5, 10, 11), current_state)
    current_state = np.dot(CNOT(6, 10, 11), current_state)
    current_state = np.dot(CNOT(7, 10, 11), current_state)
    current_state = np.dot(CNOT(8, 10, 11), current_state)


    # Hadamards applied to each qubit on the right
    final_state = np.dot(hadamards, current_state)
    final_state[np.abs(final_state) < 1e-15] = 0


    # Measure ancilla qubits
    final_state = collapse_ancilla(final_state, 2)

    ancilla_bits = vector_state_to_bit_state(final_state, 11)[0][0][9:]

    a1 = ancilla_bits[0]
    a2 = ancilla_bits[1]

    if (a1 == '0') and (a2 == '0'):
        corrected_state = final_state
    elif (a1 == '1') and (a2 == '0'):
        operation = np.kron(sigma_z, np.identity(2**10))
        corrected_state = np.dot(operation, final_state)
    elif (a1 == '0') and (a2 == '1'):
        operation = np.kron(np.identity(2**6),np.kron(sigma_z, np.identity(2**4)))
        corrected_state = np.dot(operation, final_state)
    elif (a1 == '1') and (a2 == '1'):
        operation = np.kron(np.identity(2**3), np.kron(sigma_z, np.identity(2**7)))
        corrected_state = np.dot(operation, final_state)

    corrected_state = ancilla_reset(corrected_state, 2)
    return corrected_state

### Detects where the x rotation errors occured from the vector from of the 11 qubit system ###

# We will treat this error detection in three blocks
# Each of the three (imaginary) blocks of the 9 Qubit Code,
# correspond to same behavior of the three qubit system.

# First Block
def first_block(logical_state):
    # logical_state: the full vector state representation of your 11 qubit system (9 data, 2 ancilla)

    current_state = np.dot(CNOT(0, 9, 11), logical_state)
    current_state = np.dot(CNOT(1, 9, 11), current_state)
    current_state = np.dot(CNOT(0, 10, 11), current_state)
    current_state = np.dot(CNOT(2, 10, 11), current_state)

    # Measure ancilla qubits
    final_state = collapse_ancilla(current_state, 2)
    ancilla_bits = vector_state_to_bit_state(final_state, 11)[0][0][9:]

    a1 = ancilla_bits[0]
    a2 = ancilla_bits[1]

    if (a1 == '0') and (a2 == '0'):
        corrected_state = final_state
    elif (a1 == '1') and (a2 == '1'):
        operation = np.kron(sigma_x, np.identity(2**10))
        corrected_state = np.dot(operation, final_state)
    elif (a1 == '1') and (a2 == '0'):
        operation = np.kron(np.identity(2), np.kron(sigma_x, np.identity(2**9)))
        corrected_state = np.dot(operation, final_state)
    elif (a1 == '0') and (a2 == '1'):
        operation = np.kron(np.identity(2**2), np.kron(sigma_x, np.identity(2**8)))
        corrected_state = np.dot(operation, final_state)

    corrected_state = ancilla_reset(corrected_state, 2)

    return corrected_state

# Second Block
def second_block(logical_state):
    # logical_state: the full vector state representation of your 11 qubit system (9 data, 2 ancilla)

    current_state = np.dot(CNOT(3, 9, 11), logical_state)
    current_state = np.dot(CNOT(4, 9, 11), current_state)
    current_state = np.dot(CNOT(3, 10, 11), current_state)
    current_state = np.dot(CNOT(5, 10, 11), current_state)

    # Measure ancilla qubits
    final_state = collapse_ancilla(current_state, 2)
    ancilla_bits = vector_state_to_bit_state(final_state, 11)[0][0][9:]

    a1 = ancilla_bits[0]
    a2 = ancilla_bits[1]

    if (a1 == '0') and (a2 == '0'):
        corrected_state = final_state
    elif (a1 == '1') and (a2 == '1'):
        operation = np.kron(np.identity(2**3),np.kron(sigma_x, np.identity(2**7)))
        corrected_state = np.dot(operation, final_state)
    elif (a1 == '1') and (a2 == '0'):
        operation = np.kron(np.identity(2**4),np.kron(sigma_x, np.identity(2**6)))
        corrected_state = np.dot(operation, final_state)
    elif (a1 == '0') and (a2 == '1'):
        operation = np.kron(np.identity(2**5),np.kron(sigma_x, np.identity(2**5)))
        corrected_state = np.dot(operation, final_state)

    corrected_state = ancilla_reset(corrected_state, 2)
    return corrected_state

# Third Block
def third_block(logical_state):
    # logical_state: the full vector state representation of your 11 qubit system (9 data, 2 ancilla)

    current_state = np.dot(CNOT(6, 9, 11), logical_state)
    current_state = np.dot(CNOT(7, 9, 11), current_state)
    current_state = np.dot(CNOT(6, 10, 11), current_state)
    current_state = np.dot(CNOT(8, 10, 11), current_state)

    # Measure ancilla qubits
    final_state = collapse_ancilla(current_state, 2)
    ancilla_bits = vector_state_to_bit_state(final_state, 11)[0][0][9:]

    a1 = ancilla_bits[0]
    a2 = ancilla_bits[1]

    if (a1 == '0') and (a2 == '0'):
        corrected_state = final_state
    elif (a1 == '1') and (a2 == '1'):
        operation = np.kron(np.identity(2**6),np.kron(sigma_x, np.identity(2**4)))
        corrected_state = np.dot(operation, final_state)
    elif (a1 == '1') and (a2 == '0'):
        operation = np.kron(np.identity(2**7),np.kron(sigma_x, np.identity(2**3)))
        corrected_state = np.dot(operation, final_state)
    elif (a1 == '0') and (a2 == '1'):
        operation = np.kron(np.identity(2**8),np.kron(sigma_x, np.identity(2**2)))
        corrected_state = np.dot(operation, final_state)

    corrected_state = ancilla_reset(corrected_state, 2)

    return corrected_state

### Corrects the x rotation errors on the vector from of the 11 qubit system ###
def  nine_qubit_bit_correction(logical_state):
    # logical_state: the full vector state representation of your 11 qubit system (9 data, 2 ancilla)

    state= first_block(logical_state)
    state= second_block(state)
    state= third_block(state)
    return state

### The Full 9-qubit code combining bit and phase correction on the 11 qubit state ###
def full_nine_qubit_code(logical_state):
    # logical_state: the full vector state representation of your 11 qubit system (9 data, 2 ancilla)

    # Apply and Detect Phase-Flip Errors
    phase_corrected_state = nine_qubit_phase_correction(logical_state)

    #Apply and Detect Bit-Flip Errors
    bit_corrected_state = nine_qubit_bit_correction(phase_corrected_state)

    return bit_corrected_state
