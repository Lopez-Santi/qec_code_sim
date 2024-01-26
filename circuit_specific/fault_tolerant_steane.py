# This file will contain functions that are useful when implementing fault tolerance quantum circuits

import numpy as np
import random
from general_qec.qec_helpers import *
from general_qec.gates import *
from general_qec.errors import *

### - - - - - - - - - - Fault Tolerant 7 qubit Steane code - - - - - - - - - - ###

### Initializes and encodes the ancilla block with x errors (5 ancillas out of 12 qubits) ###
def initialize_ancilla(logical_state):
    # logical_state: The full vector state representation of your 12 qubit system (7 data, 5 ancilla)

    # Reset the ancillas if needed: (initialize ancilla block to |00000>)
    reset_state = ancilla_reset(logical_state, 5)

    # set the range where errors can occur
    qubit_range = [7, 11]

    # create the hadamard gate
    h = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(hadamard, np.identity(2))))
    # apply hadamard
    reset_state = np.dot(h, reset_state)
    
    # apply x error
    reset_state = random_qubit_x_error(reset_state, qubit_range)[0]

    # create the CNOT gates
    cnot_gates = np.dot(CNOT(7, 11, 12), np.dot(CNOT(8, 7 ,12), np.dot(CNOT(9, 8, 12), np.dot(
        CNOT(10, 11, 12), CNOT(10, 9, 12)))))
    # apply  CNOT gates
    final_state = np.dot(cnot_gates, reset_state)
    
    return final_state

### Loops over the initialization of the ancilla block until M (10th index) measures 0 (12 qubits) ###
def fault_tolerant_ancilla(logical_state):
    # logical_state: The vector state representation of you 12 qubit system
    a = '1'
    while a == '1':
        state = initialize_ancilla(logical_state)
        qubit_range = [7, 11]
        
        # apply z error
        state = random_qubit_z_error(state, qubit_range)[0]
    
        # collapse the 12th qubit to measure it
        state = collapse_ancilla(state, 1)
        bits = vector_state_to_bit_state(state, 12)[0]
        a = bits[0][11:]
    
    return state

### Loops over the initialization of the ancilla block until M (10th index) measures 0 (12 qubits) ###
def omit_z_fault_tolerant_ancilla(logical_state):
    # logical_state: The vector state representation of you 12 qubit system
    a = '1'
    while a == '1':
        state = initialize_ancilla(logical_state)
        # collapse the 12th qubit to measure it
        state = collapse_ancilla(state, 1)
        bits = vector_state_to_bit_state(state, 12)[0]
        a = bits[0][11:]
    
    return state


### - - - - - Measurement operators for fault tolerant steane code - - - - - ###

def measure_K1(logical_state, error = None):
    # logical_state: The full vector state representation of your 12 qubit system (7 data, 5 ancilla)
    # error: is there a z error present in the ancilla initialization
    
    if error == 'z':
        # run the ancilla fault tolerant check
        initial_state = fault_tolerant_ancilla(logical_state)
    else:
        initial_state = omit_z_fault_tolerant_ancilla(logical_state)
        
    # Perform Stabilizer operation
    h = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2)))
    K1 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
        CNOT(10, 6, 12), np.dot(CNOT(9, 5, 12), np.dot(CNOT(8, 4, 12), CNOT(7, 3, 12)))))
    

    final_state = np.dot(K1, initial_state)
    final_state = np.dot(h, final_state)
    
    final_state = remove_small_values(final_state)
       
    # Collapse the ancilla qubits
    collapsed_state = collapse_ancilla(final_state, 5)
    
    # measure the 10th index qubit
    measurement = vector_state_to_bit_state(collapsed_state, 12)[0][0][10:11]
    
    return collapsed_state, measurement

def measure_K2(logical_state, error = None):
    # logical_state: The full vector state representation of your 12 qubit system (7 data, 5 ancilla)
    # error: is there a z error present in the ancilla initialization
    
    if error == 'z':
        # run the ancilla fault tolerant check
        initial_state = fault_tolerant_ancilla(logical_state)
    else:
        initial_state = omit_z_fault_tolerant_ancilla(logical_state)
    
    # Perform Stabilizer operation 
    h = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2)))
    K2 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
        CNOT(10, 6, 12), np.dot(CNOT(9, 4, 12), np.dot(CNOT(8, 2, 12), CNOT(7, 0, 12)))))

    final_state = np.dot(K2, initial_state)
    final_state = np.dot(h, final_state)

    final_state = remove_small_values(final_state)

    # Collapse the ancilla qubits
    collapsed_state = collapse_ancilla(final_state, 5)
    
    # measure the 10th index qubit
    measurement = vector_state_to_bit_state(collapsed_state, 12)[0][0][10:11]
    
    return collapsed_state, measurement

def measure_K3(logical_state, error = None):
    # logical_state: The full vector state representation of your 12 qubit system (7 data, 5 ancilla)
    # error: is there a z error present in the ancilla initialization
    
    if error == 'z':
        # run the ancilla fault tolerant check
        initial_state = fault_tolerant_ancilla(logical_state)
    else:
        initial_state = omit_z_fault_tolerant_ancilla(logical_state)

    # Perform Stabilizer operation 
    h = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2)))
    K3 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
        CNOT(10, 6, 12), np.dot(CNOT(9, 5, 12), np.dot(CNOT(8, 2, 12), CNOT(7, 1, 12)))))

    final_state = np.dot(K3, initial_state)
    final_state = np.dot(h, final_state)
    
    final_state = remove_small_values(final_state)

    # Collapse the ancilla qubits
    collapsed_state = collapse_ancilla(final_state, 5)
    
    # measure the 10th index qubit
    measurement = vector_state_to_bit_state(collapsed_state, 12)[0][0][10:11]
        
    return collapsed_state, measurement

def measure_K4(logical_state, error = None):
    # logical_state: The full vector state representation of your 12 qubit system (7 data, 5 ancilla)
    # error: is there a z error present in the ancilla initialization
    
    if error == 'z':
        # run the ancilla fault tolerant check
        initial_state = fault_tolerant_ancilla(logical_state)
    else:
        initial_state = omit_z_fault_tolerant_ancilla(logical_state)

    # Perform Stabilizer operation 
    h = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2)))
    K4 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
        CZ(10, 6, 12), np.dot(CZ(9, 5, 12), np.dot(CZ(8, 4, 12), CZ(7, 3, 12)))))
    

    final_state = np.dot(K4, initial_state)
    final_state = np.dot(h, final_state)
    
    final_state = remove_small_values(final_state)
        
    # Collapse the ancilla qubits
    collapsed_state = collapse_ancilla(final_state, 5)
    
    # measure the 10th index qubit
    measurement = vector_state_to_bit_state(collapsed_state, 12)[0][0][10:11]
    
    return collapsed_state, measurement

def measure_K5(logical_state, error = None):
    # logical_state: The full vector state representation of your 12 qubit system (7 data, 5 ancilla)
    # error: is there a z error present in the ancilla initialization
    
    if error == 'z':
        # run the ancilla fault tolerant check
        initial_state = fault_tolerant_ancilla(logical_state)
    else:
        initial_state = omit_z_fault_tolerant_ancilla(logical_state)

    # Perform Stabilizer operation
    h = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2)))
    K5 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
        CZ(10, 6, 12), np.dot(CZ(9, 4, 12), np.dot(CZ(8, 2, 12), CZ(7, 0, 12)))))

    final_state = np.dot(K5, initial_state)
    final_state = np.dot(h, final_state)
    
    final_state = remove_small_values(final_state)
    
    # Collapse the ancilla qubits
    collapsed_state = collapse_ancilla(final_state, 5)
    
    # measure the 10th index qubit
    measurement = vector_state_to_bit_state(collapsed_state, 12)[0][0][10:11]
    
    return collapsed_state, measurement

def measure_K6(logical_state, error = None):
    # logical_state: The full vector state representation of your 12 qubit system (7 data, 5 ancilla)
    # error: is there a z error present in the ancilla initialization
    
    if error == 'z':
        # run the ancilla fault tolerant check
        initial_state = fault_tolerant_ancilla(logical_state)
    else:
        initial_state = omit_z_fault_tolerant_ancilla(logical_state)

    # Perform Stabilizer operation
    h = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2)))
    K6 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
        CZ(10, 6, 12), np.dot(CZ(9, 5, 12), np.dot(CZ(8, 2, 12), CZ(7, 1, 12)))))

    final_state = np.dot(K6, initial_state)
    final_state = np.dot(h, final_state)
    
    final_state = remove_small_values(final_state)

    # Collapse the ancilla qubits
    collapsed_state = collapse_ancilla(final_state, 5)
    
    # measure the 10th index qubit
    measurement = vector_state_to_bit_state(collapsed_state, 12)[0][0][10:11]
        
    return collapsed_state, measurement


### Apply the steane code fault tolerantly (initializes logical state and corrects for errors)
def fault_tolerant_steane_code(logical_state):
    # logical_state: The full vector state representation of your 12 qubit system (7 data, 5 ancilla)

    state = logical_state

    qubit_range = [8, 12]

    # Choose where the z error occurs
    spot = random.randint(1, 3)

    # - - - Measuring for phase correction - - - #

    # Measurement for K1 the first time
    if spot == 1:
        state, m11 = measure_K1(state, 'z')
    else:
        state, m11 = measure_K1(state)

    print('- First m1 measurement done: ', m11)

    # Measurement for K1 the second time
    if spot == 2:
        state, m12 = measure_K1(state, 'z')
    else:
        state, m12 = measure_K1(state)

    print('- Second m1 measurement done: ', m12)

    # Measurement for K1 the third time only if first two results dont agree
    if m11 == m12:
        m1 = m11
        print('No need for third measurement.')

    else:
        state, m13 = measure_K1(state)
        m1 = m13
        print('- Third m1 measurement done: ', m13)

    print('- - - M1: ', m1)

    # Choose where the z error occurs
    spot = random.randint(1, 3)

    # Measurement for K2 the first time
    if spot == 1:
        state, m21 = measure_K2(state, 'z')
    else:
        state, m21 = measure_K2(state)

    print('- First m2 measurement done: ', m21)

    # Measurement for K2 the second time
    if spot == 2:
        state, m22 = measure_K2(state, 'z')
    else:
        state, m22 = measure_K2(state)

    print('- Second m2 measurement done: ', m22)

    # Measurement for K2 the third time only if first two results dont agree
    if m21 == m22:
        m2 = m21
        print('No need for third measurement.')

    else:
        state, m23 = measure_K2(state)
        m2 = m23
        print('- Third m2 measurement done: ', m23)

    print('- - - M2: ', m2)

    # Choose where the z error occurs
    spot = random.randint(1, 3)

    # Measurement for K3 the first time
    if spot == 1:
        state, m31 = measure_K3(state, 'z')
    else:
        state, m31 = measure_K3(state)

    print('- First m3 measurement done: ', m31)

    # Measurement for K3 the second time
    if spot == 2:
        state, m32 = measure_K3(state, 'z')
    else:
        state, m32 = measure_K3(state)

    print('- Second m3 measurement done: ', m32)

    # Measurement for K3 the third time only if first two results dont agree
    if m31 == m32:
        m3 = m31
        print('No need for third measurement.')

    else:
        state, m33 = measure_K3(state)
        m3 = m33
        print('- Third m3 measurement done: ', m33)

    print('- - - M3: ', m3)

    print('Measurements done...')
    print('M1, M2, M3: ', m1,', ', m2,', ', m3)
    print('State after measurements:')
    print_state_info(state, 12)


    # How many total qubits are in our vector representation
    n = int(np.log(len(state))/np.log(2))

    # Applying the Z gate operation on a qubit depending on the ancilla measuremnts
    m_one = 0
    m_two = 0
    m_three = 0
    if m1 == '1':
        m_one = 1
    if m2 == '1':
        m_two = 1
    if m3 == '1':
        m_three = 1
    # Which qubit do we perform the Z gate on
    index = ((m_one * 2**2) + (m_three * 2**1) + (m_two * 2**0)) - 1

    # if no error occurs we dont need to apply a correction
    if index == -1:
        final_vector_state = state
    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**index), np.kron(sigma_z, np.kron(
            np.identity(int(2**(n-5-index-1))), np.identity(2**5))))

        final_vector_state = np.dot(operation, state)

    # Takes the logical |1> to a +1 eigenstate of the stabilizer operators
    z_bar = np.kron(sigma_z, np.kron(sigma_z, np.kron(sigma_z, np.kron(
        sigma_z, np.kron(sigma_z, np.kron(sigma_z, sigma_z))))))
    z_bar = np.kron(z_bar, np.identity(2**5))

    # corrected_vector_state = remove_small_values(final_vector_state)
    corrected_vector_state = final_vector_state

    corrected_vector_state = np.dot(z_bar, corrected_vector_state)

    print('State after phase correction: ')
    print_state_info(corrected_vector_state, 12)


    # - - - Measuring for bit correction - - - #

    # Choose where the z error occurs
    spot = random.randint(1, 3)

    # Measurement for K4 the first time
    if spot == 1:
        state, m41 = measure_K4(corrected_vector_state, 'z')
    else:
        state, m41 = measure_K4(corrected_vector_state)

    print('- First m4 measurement done: ', m41)

    # Measurement for K4 the second time
    if spot == 2:
        state, m42 = measure_K4(state, 'z')
    else:
        state, m42 = measure_K4(state)

    print('- Second m4 measurement done: ', m42)

    # Measurement for K4 the third time only if first two results dont agree
    if m41 == m42:
        m4 = m41
        print('No need for third measurement.')

    else:
        state, m43 = measure_K4(state)
        m4 = m43
        print('- Third m4 measurement done: ', m43)

    print('- - - M4: ', m4)

    # Choose where the z error occurs
    spot = random.randint(1, 3)

    # Measurement for K5 the first time
    if spot == 1:
        state, m51 = measure_K5(state, 'z')
    else:
        state, m51 = measure_K5(state)

    print('- First m5 measurement done: ', m51)

    # Measurement for K5 the second time
    if spot == 2:
        state, m52 = measure_K5(state, 'z')
    else:
        state, m52 = measure_K5(state)

    print('- Second m5 measurement done: ', m52)

    # Measurement for K5 the third time only if first two results dont agree
    if m51 == m52:
        m5 = m51
        print('No need for third measurement.')

    else:
        state, m53 = measure_K5(state)
        m5 = m53
        print('- Third m5 measurement done: ', m53)

    print('- - - M5: ', m5)

    # Choose where the z error occurs
    spot = random.randint(1, 3)

    # Measurement for K6 the first time
    if spot == 1:
        state, m61 = measure_K6(state, 'z')
    else:
        state, m61 = measure_K6(state)

    print('- First m6 measurement done: ', m61)

    # Measurement for K6 the second time
    if spot == 2:
        state, m62 = measure_K6(state, 'z')
    else:
        state, m62 = measure_K6(state)

    print('- Second m6 measurement done: ', m62)

    # Measurement for K6 the third time only if first two results dont agree
    if m61 == m62:
        m6 = m61
        print('No need for third measurement.')

    else:
        state, m63 = measure_K6(state)
        m6 = m63
        print('- Third m6 measurement done: ', m63)

    print('- - - M6: ', m6)

    print('Measurements done...')
    print('M4, M5, M6: ', m4,', ', m5,', ', m6)
    print('State after measurements:')
    print_state_info(state, 12)

    # How many total qubits are in our vector representation
    n = int(np.log(len(state))/np.log(2))

    # Applying the X gate operation on a qubit depending on the ancilla measuremnts
    m_four = 0
    m_five = 0
    m_six = 0
    if m4 == '1':
        m_four = 1
    if m5 == '1':
        m_five = 1
    if m6 == '1':
        m_six = 1
    # Which qubit do we perform the Z gate on
    index = ((m_four * 2**2) + (m_six * 2**1) + (m_five * 2**0)) - 1

    # if no error occurs we dont need to apply a correction
    if index == -1:
        final_vector_state = state
    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**index), np.kron(sigma_x, np.kron(
            np.identity(int(2**(n-5-index-1))), np.identity(2**5))))

        final_vector_state = np.dot(operation, state)

    corrected_vector_state = final_vector_state
    
    # Takes the logical |1> to a +1 eigenstate of the stabilizer operators
#     corrected_vector_state = np.dot(z_bar, corrected_vector_state)

    print('State after bit correction: ')
    print_state_info(corrected_vector_state, 12)
    
    return corrected_vector_state