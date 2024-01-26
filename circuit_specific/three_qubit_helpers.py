# The functions in this file are useful when implementing the three qubit code

import numpy as np
import random
from general_qec.gates import *
from general_qec.qec_helpers import *

### Initializes the three qubit logical state using an initial single qubit psi ###
def three_qubit_initialize_logical_state(initial_psi):
    # initial_psi: initial state of your single qubit that you want to use as your logical state (2 x 1)
    
    initial_ancilla_state = np.array([1,0]) # initializing the |0> state of the qubits
    
    # Initialize the 3 qubit logical state by using thr kronecker product
    initial_logical_state = np.kron(initial_psi, np.kron(initial_ancilla_state, initial_ancilla_state))

    # Setting up the 2 CNOT gates to initialize the correct logical qubit
    cnot_psi_qzero = np.kron(cnot, np.identity(2))
    cnot_qzero_qone = np.kron(np.identity(2), cnot)
    
    # Apply the CNOT gates to the kronecker product of the current 3 qubit state
    final_logical_state = np.dot(cnot_qzero_qone, np.dot(cnot_psi_qzero, initial_logical_state))
    
    return final_logical_state

### - - - - - - Errors - - - - - - ###

### Applies a random X rotation to one of the three physical qubits in your system (randomly) ###
def three_qubit_random_qubit_x_error(logical_state):
    # logical_state: The logical state of the three qubit system you wish to apply the error to (8 x 1)
    
    # Choose the index of the qubit you want to apply the error to.
    error_index = random.randint(-1,2)
    # Apply the error to the qubit (no error may occur)
    if error_index == 0:
        errored_logical_state = np.dot(np.kron(sigma_x, np.identity(4)), logical_state)
    elif error_index == 1:
        errored_logical_state = np.dot(np.kron(np.kron(np.identity(2), sigma_x), np.identity(2)), logical_state)
    elif error_index == 2:
        errored_logical_state = np.dot(np.kron(np.identity(4), sigma_x), logical_state)
    else:
        errored_logical_state = logical_state
        
    return errored_logical_state, error_index


### Applies an X rotation to one of the three physical qubits in your system (your choice for which qubit is errored) ###
def three_qubit_defined_qubit_x_error(logical_state, error_index):
    # logical_state: The logical state of the three qubit system you wish to apply the error to (8 x 1)
    # error_index: Which qubit you want the error to occur.
    
    # Apply the error to the qubit (no error may occur)
    if error_index == 0:
        errored_logical_state = np.dot(np.kron(sigma_x, np.identity(4)), logical_state)
    elif error_index == 1:
        errored_logical_state = np.dot(np.kron(np.kron(np.identity(2), sigma_x), np.identity(2)), logical_state)
    elif error_index == 2:
        errored_logical_state = np.dot(np.kron(np.identity(4), sigma_x), logical_state)
    else:
        errored_logical_state = logical_state
        
    return errored_logical_state, error_index


### Applies an arbitrary X rotation to 1-3 of the physical qubits in your system ###
def three_qubit_coherent_x_rotation_error(logical_state, epsilon, qubits):
    # logical_state: state of the logical qubit before error occurs
    # epsilon: error constant in a coherent rotation error
    U = np.cos(epsilon) * sigma_I + 1j*np.sin(epsilon) * sigma_x # Create the Unitary error operator 
    
    if qubits == 3:
        E = np.kron(U, np.kron(U, U)) # Create the Error operator that will act on our logical qubit
    
    elif qubits == 2:
        # Choose the index of the qubit you want to apply the error to.
        error_index = random.randint(0,2)
        # Apply the error to two qubits
        if error_index == 0:
            E = np.kron(U, np.kron(U, np.identity(2))) # Create the Error operator that will act on our logical qubit
        elif error_index == 1:
            E = np.kron(U, np.kron(np.identity(2), U)) # Create the Error operator that will act on our logical qubit        
        elif error_index == 2:
            E = np.kron(np.identity(2), np.kron(U, U)) # Create the Error operator that will act on our logical qubit
        
    elif qubits == 1:
        # Choose the index of the qubit you want to apply the error to.
        error_index = random.randint(0,2)
        # Apply the error to two qubits
        if error_index == 0:
            E = np.kron(U, np.kron(np.identity(2), np.identity(2))) # Create the Error operator that will act on our logical qubit
        elif error_index == 1:
            E = np.kron(np.identity(2), np.kron(np.identity(2), U)) # Create the Error operator that will act on our logical qubit        
        elif error_index == 2:
            E = np.kron(np.identity(2), np.kron(U, np.identity(2))) # Create the Error operator that will act on our logical qubit
    
    # Apply error
    errored_state = np.dot(E, logical_state)
    
    return errored_state, E, U


### - - - - - - Error Detection - - - - - - ###

### Applying the ancilla qubits to the three qubit system ###
def three_qubit_apply_ancillas(logical_state):
    # logical_state: the vector state representation of our 3 qubit system (8 x 1)
    
    # Extend our system to add in the 2 syndrome ancilla qubits
    full_system = np.kron(logical_state, np.kron(np.array([1,0]), np.array([1,0]))) 

    # Apply the CNOT gates needed to change the state of the syndrome ancilla 
    final_logical_state = np.dot(non_adj_CNOT(2,4,5), np.dot(non_adj_CNOT(0,4,5), 
                                                     np.dot(non_adj_CNOT(1,3,5), np.dot(non_adj_CNOT(0,3,5), full_system))))
    
    return final_logical_state


### Detects where the x rotation error occured from the vector from of the 5 qubit system ###
def three_qubit_detect_error_location_from_vector(logical_state):
    # logical_state: the logical state of our 3 qubit system with 2 ancillas (32 x 1)
    
    # Initialize error index
    error_index = -1

    if (logical_state[28] != 0) or (logical_state[0] != 0): # No error occured
        error_index = -1
        return error_index, print("No bit flip error occured.")
    elif (logical_state[15] != 0) or (logical_state[19] != 0): # Error on qubit 0
        error_index = 0
    elif (logical_state[22] != 0) or (logical_state[10] != 0): # Error on qubit 1
        error_index = 1
    elif (logical_state[25] != 0) or (logical_state[5] != 0): # Error on qubit 2
        error_index = 2
        
    return error_index, print("Bit flip error occured on qubit", error_index )


### Detects where the x rotation error occured from the bit form of the 5 qubit system ###
def three_qubit_detect_error_location_from_bit_state(logical_bits):
    # logical_bits: set of 5 qubits to detect errors with 2 ancilla within the 5 (00000)
    
    # Initialize error index
    error_index = -1
    
    if ((logical_bits[3] == '1') and (logical_bits[4] == '1')): # Error on qubit 0
        error_index = 0
    elif ((logical_bits[3] == '1') and (logical_bits[4] == '0')): # Error on qubit 1
        error_index = 1
    elif ((logical_bits[3] == '0') and (logical_bits[4] == '1')): # Error on qubit 2
        error_index = 2
    elif(logical_bits[3] and logical_bits[4] == '0'): # No error occured
        return error_index, print("No bit flip error occured.")
    
    return error_index, print("Bit flip error occured on qubit", error_index )


### - - - - - - Error Correction - - - - - - ###

### Correct for errors by applying full X rotation gate to the qubit where the error occured. ###
def three_qubit_correct_x_error(logical_state):
    # logical_state: the logical state of our 3 qubit system with 2 ancillas (32 x 1)

    # Find where the error occured using the error detection function
    qubit_index = three_qubit_detect_error_location_from_vector(logical_state)[0]
    
    if qubit_index == 0: # Error on qubit 0
        corrected_state = np.dot(np.kron(sigma_x, np.identity(16)), logical_state)
    elif qubit_index == 1: # Error on qubit 1
        corrected_state = np.dot(np.kron(np.identity(2), np.kron(sigma_x, np.identity(8))), logical_state)
    elif qubit_index == 2: # Error on qubit 2
        corrected_state = np.dot(np.kron(np.identity(4), np.kron(sigma_x, np.identity(4))), logical_state)
    else: # No error occured
        corrected_state = logical_state
    
    return corrected_state


### Takes the error state and separates it into vectors so that we can see the separate error bit states and correct for them
def separate_and_correct_multiple_errors(error_state, n):
    # error_state: The errored logical state of your system
    # n: total number of qubits in your system
    
    x = 0 # used to keep track of first indice where error_state is non-zero
    for i in range(len(error_state)):
        if error_state[i] != 0: 
            error_position = np.zeros((1,2**n), dtype=complex) # initialize the vector that will hold the single non-zero value
            error_position[:,i] = error_state[i] # insert the non-zero value in the correct spot
            
            # Add the error position vector to an array of all the error places
            if x == 0:
                all_error_states = error_position
            else:
                all_error_states = np.append(all_error_states, error_position , axis=0)
            x+=1

    # find the number of rows and columns in the all error state array so that we can loop over the rows
    num_rows, num_cols = all_error_states.shape

    # Here we take all of the error states and apply the correction depending on where the error occurs
    for i in range(num_rows):
        # correct for the error
        vector_correct_state = three_qubit_correct_x_error(all_error_states[i][:])
        bit_correct_state = vector_state_to_bit_state(vector_correct_state, 5)[0]
        
        # add the corrected states to the array for all the corrected states
        if i == 0:
            corrected_states = [vector_correct_state]
            corrected_bit_states = bit_correct_state
        else:
            corrected_states = np.append(corrected_states, [vector_correct_state], axis = 0)
            corrected_bit_states = np.append(corrected_bit_states, bit_correct_state, axis = 0)
    
    return corrected_states, corrected_bit_states

### - - - - - - Outputting Information - - - - - - ###

### Outputting the information for the three qubit bit flip correcting code
def three_qubit_info(vector_error_state, vector_corrected_state):
    # vector_error_state: the logical state of our errored 3 qubit system with 2 ancillas (32 x 1)
    # vector_corrected_state: the logical state of our corrected 3 qubit system with 2 ancillas (32 x 1)
    
    # Find the 5 bit representation information for the errored vector state
    error_logical_bits, error_index, error_state = vector_state_to_bit_state(vector_error_state, 5)
    
    # Ouput the information for the 5 bit errored state
    if len(error_index) < 2:
        print('Full system error bit state:     ', 
              error_state[error_index[0].astype(int)], error_logical_bits[0])
    else:
        print('Full system error bit state:     ', 
              error_state[error_index[0].astype(int)], error_logical_bits[0], ' + ', 
              error_state[error_index[1].astype(int)], error_logical_bits[1])

    # Find the 5 bit representation information for the errored vector state
    corrected_logical_bits, corrected_index, corrected_state = vector_state_to_bit_state(vector_corrected_state, 5)
    
    # Ouput the information for the 5 bit corrected state
    if len(corrected_index) < 2:
        print('Full system corrected bit state: ', 
              corrected_state[corrected_index[0].astype(int)], corrected_logical_bits[0])
    else:
        print('Full system corrected bit state: ', 
              corrected_state[corrected_index[0].astype(int)], corrected_logical_bits[0], ' + ', 
              corrected_state[corrected_index[1].astype(int)], corrected_logical_bits[1])

    # Find the 3 bit representation information for the errored vector state
    error_logical_bits, error_index, error_state = vector_state_to_bit_state(vector_error_state, 3)
    
    # Ouput the information for the 3 bit errored state
    if len(error_index) < 2:
        print('Error bit state:                 ', 
              error_state[error_index[0].astype(int)], error_logical_bits[0])
    else:
        print('Error bit state:                 ', 
              error_state[error_index[0].astype(int)], error_logical_bits[0], '   + ', 
              error_state[error_index[1].astype(int)], error_logical_bits[1])
    
    # Find the 3 bit representation information for the corrected vector state
    corrected_logical_bits, corrected_index, corrected_state = vector_state_to_bit_state(vector_corrected_state, 3)
    
    # Ouput the information for the 3 bit corrected state
    if len(corrected_index) < 2:
        print('Corrected bit state:             ', 
              corrected_state[corrected_index[0].astype(int)], corrected_logical_bits[0])
    else:
        print('Corrected bit state:             ', 
              corrected_state[corrected_index[0].astype(int)], corrected_logical_bits[0], '   + ', 
              corrected_state[corrected_index[1].astype(int)], corrected_logical_bits[1])







