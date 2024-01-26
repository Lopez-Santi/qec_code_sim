### This file will contain functions that are useful when implementing the fault tolerant steane code with realistic error models

import numpy as np
import random
from general_qec.qec_helpers import *
from general_qec.gates import *
from general_qec.errors import *

### Loops over the initialization of the ancilla block until M (10th index) measures 0 (12 qubits) ###
def realistic_ft_ancilla(initial_rho, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None):
    # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(initial_rho))/np.log(2))
    
    a = '1'
    while a == '1':
        # Reset the ancillas by projecting to the |00><00| basis    
        # apply correct measurement collapse of the density matrix
        M = np.kron(np.identity(2**7), np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas,zero_meas)))))
        ancilla_reset_prob = np.trace(np.dot(M.conj().T, np.dot(M, initial_rho)))
        reset_rho = np.dot(M1, np.dot(initial_rho, M1.conj().T))/(ancilla_reset_prob)
        initial_rho = reset_rho
        # Apply state prep error if spam_probs is not empty
        if spam_prob != None:
            for index in range(n):
                initial_rho = spam_error(initial_rho, spam_prob, index)

        
        # create the hadamard gate
        h = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(hadamard, np.identity(2))))
        # apply hadamard
        current_rho = np.dot(h, initial_rho)
        
        # Apply error gates depending on error parameters 
        if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
            # apply depolarizing error
            current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
            # apply rad error
            current_rho = rad_error(current_rho, t1, t2, tg)
        elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
            # apply rad error
            current_rho = rad_error(current_rho, t1, t2, tg)
        elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
            # apply depolarizing error
            current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
        else:
            current_rho = current_rho

        # create and apply the CNOT gates depending on error parameters
        if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
            current_rho = prob_line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, qubit_error_probs, form = 'rho')
            current_rho = prob_line_rad_CNOT(current_rho, 10, 11, t1, t2, tg, qubit_error_probs, form = 'rho')
            current_rho = prob_line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, qubit_error_probs, form = 'rho')
            current_rho = prob_line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, qubit_error_probs, form = 'rho')
            current_rho = prob_line_rad_CNOT(current_rho, 7, 11, t1, t2, tg, qubit_error_probs, form = 'rho')
        elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
            current_rho = line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, form = 'rho')
            current_rho = line_rad_CNOT(current_rho, 10, 11, t1, t2, tg, form = 'rho')
            current_rho = line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, form = 'rho')
            current_rho = line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, form = 'rho')
            current_rho = line_rad_CNOT(current_rho, 7, 11, t1, t2, tg, form = 'rho')
        elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
            current_rho = line_errored_CNOT(current_rho, 10, 9, qubit_error_probs, form = 'rho')
            current_rho = line_errored_CNOT(current_rho, 10, 11, qubit_error_probs, form = 'rho')
            current_rho = line_errored_CNOT(current_rho, 9, 8, qubit_error_probs, form = 'rho')
            current_rho = line_errored_CNOT(current_rho, 8, 7, qubit_error_probs, form = 'rho')
            current_rho = line_errored_CNOT(current_rho, 7, 11, qubit_error_probs, form = 'rho')
        else:
            cnot_gates = np.dot(CNOT(7, 11, 12), np.dot(CNOT(8, 7 ,12), np.dot(CNOT(9, 8, 12), np.dot(
                CNOT(10, 11, 12), CNOT(10, 9, 12)))))
            current_rho = np.dot(cnot_gates, np.dot(current_rho, cnot_gates.conj().T))
    
         # Apply state measurement error if spam_probs is not empty
        if spam_prob != None:
            current_rho = spam_error(current_rho, spam_prob, 10) # ancilla
        
        
        # Masurement operators for individual qubits
        zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
        one_meas = np.kron(one, one[np.newaxis].conj().T)

        # Define the measurement projection operators
        M0 = np.kron(np.identity(2**9), np.kron(np.identity(2), np.kron(np.identity(2), zero_meas)))
        M1 = np.kron(np.identity(2**9), np.kron(np.identity(2), np.kron(np.identity(2), one_meas)))

        all_meas = np.array([M0, M1])

        # find the probability to measure each case
        m0_prob = np.trace(np.dot(M0.conj().T, np.dot(M0, current_rho)))
        m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, current_rho)))

        all_probs = np.array([m0_prob, m1_prob])

        # find which measurement operator is measured based on their probabilities
        index = random.choices(all_probs, weights=all_probs, k=1)
        index = np.where(all_probs == index)[0][0]

        # apply correct measurement collapse of the density matrix
        rho_prime = np.dot(all_meas[index], np.dot(current_rho, all_meas[index].conj().T))/(all_probs[index])
        # Create our new density matrix after collapsing ancilla qubits

        rho = rho_prime

        if ((t1!=None) and (t2!=None) and (tg!=None)):
            # apply an error for time taken to collapse ancilla
            rho = rad_error(rho_prime, t1, t2, tg)

        ### Just so we can look at the measurement bit
        probs = np.array([])
        for i in range(len(rho)):
            probs = np.append(probs, rho[i,i])

        collapsed_state = collapse_ancilla(np.sqrt(probs), 12)
        bits = vector_state_to_bit_state(collapsed_state, 12)[0]
        a = bits[0][11:]
    
    return rho


### - - - - - Measurement operators for realistic fault tolerant steane code - - - - - ###

def realistic_measure_K1(initial_rho, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None):
    # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(initial_rho))/np.log(2))
    
    initial_rho = realistic_ft_ancilla(initial_rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    current_rho = initial_rho
        
    # apply the control stabilizer gates to current_rho depending on error parameters
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply K1
        current_rho = prob_line_rad_CNOT(current_rho, 7, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 10, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply K1
        current_rho = line_rad_CNOT(current_rho, 7, 3, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 10, 6, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply K1
        current_rho = line_errored_CNOT(current_rho, 7, 3, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 10, 6, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 7, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 8, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 10, 9, qubit_error_probs, form = 'rho')
    else:
        K1 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
            CNOT(10, 6, 12), np.dot(CNOT(9, 5, 12), np.dot(CNOT(8, 4, 12), CNOT(7, 3, 12)))))
        current_rho = np.dot(K1, np.cot(current_rho, K1.conj().T))
    
    # hadamard for the ancilla
    ancilla_hadamard = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2))) 
    # apply the hadamard to the ancilla
    current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
    else:
        current_rho = current_rho
    
    # Apply state measurement error if spam_probs is not empty
    if spam_prob != None:
        current_rho = spam_error(current_rho, spam_prob, 10) # ancilla
        
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Define the measurement projection operators
    M0 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(zero_meas, np.identity(2))))
    M1 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(one_meas, np.identity(2))))
    
    all_meas = np.array([M0, M1])

    # find the probability to measure each case
    m0_prob = np.trace(np.dot(M0.conj().T, np.dot(M0, current_rho)))
    m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, current_rho)))

    all_probs = np.array([m0_prob, m1_prob])

    # find which measurement operator is measured based on their probabilities
    index = random.choices(all_probs, weights=all_probs, k=1)
    index = np.where(all_probs == index)[0][0]

    # apply correct measurement collapse of the density matrix
    rho_prime = np.dot(all_meas[index], np.dot(current_rho, all_meas[index].conj().T))/(all_probs[index])
    # Create our new density matrix after collapsing ancilla qubits
    
    rho = rho_prime
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # apply an error for time taken to collapse ancilla
        rho = rad_error(rho_prime, t1, t2, tg)

    # How many total qubits are in our vector representation
    n = int(np.log(len(rho))/np.log(2))

    ### Just so we can look at the measurement bit
    probs = np.array([])
    for i in range(len(rho)):
        probs = np.append(probs, rho[i,i])

    collapsed_state = collapse_ancilla(np.sqrt(probs), 10)
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
    measurement = bits[10:11]
    
    return rho, measurement

def realistic_measure_K2(initial_rho, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None):
    # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(initial_rho))/np.log(2))
    
    initial_rho = realistic_ft_ancilla(initial_rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    current_rho = initial_rho
        
    # apply the control stabilizer gates to current_rho depending on error parameters
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply K2
        current_rho = prob_line_rad_CNOT(current_rho, 7, 0, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 10, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply K2
        current_rho = line_rad_CNOT(current_rho, 7, 0, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 10, 6, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply K2
        current_rho = line_errored_CNOT(current_rho, 7, 0, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 10, 6, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 7, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 8, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 10, 9, qubit_error_probs, form = 'rho')
    else:
        K2 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
            CNOT(10, 6, 12), np.dot(CNOT(9, 4, 12), np.dot(CNOT(8, 2, 12), CNOT(7, 0, 12)))))
        current_rho = np.dot(K2, np.cot(current_rho, K2.conj().T))
    
    # hadamard for the ancilla
    ancilla_hadamard = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2))) 
    # apply the hadamard to the ancilla
    current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
    else:
        current_rho = current_rho
    
    # Apply state measurement error if spam_probs is not empty
    if spam_prob != None:
        current_rho = spam_error(current_rho, spam_prob, 10) # ancilla
        
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Define the measurement projection operators
    M0 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(zero_meas, np.identity(2))))
    M1 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(one_meas, np.identity(2))))
    
    all_meas = np.array([M0, M1])

    # find the probability to measure each case
    m0_prob = np.trace(np.dot(M0.conj().T, np.dot(M0, current_rho)))
    m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, current_rho)))

    all_probs = np.array([m0_prob, m1_prob])

    # find which measurement operator is measured based on their probabilities
    index = random.choices(all_probs, weights=all_probs, k=1)
    index = np.where(all_probs == index)[0][0]

    # apply correct measurement collapse of the density matrix
    rho_prime = np.dot(all_meas[index], np.dot(current_rho, all_meas[index].conj().T))/(all_probs[index])
    # Create our new density matrix after collapsing ancilla qubits
    
    rho = rho_prime
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # apply an error for time taken to collapse ancilla
        rho = rad_error(rho_prime, t1, t2, tg)

    # How many total qubits are in our vector representation
    n = int(np.log(len(rho))/np.log(2))

    ### Just so we can look at the measurement bit
    probs = np.array([])
    for i in range(len(rho)):
        probs = np.append(probs, rho[i,i])

    collapsed_state = collapse_ancilla(np.sqrt(probs), 10)
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
    measurement = bits[10:11]
    
    return rho, measurement

def realistic_measure_K3(initial_rho, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None):
    # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(initial_rho))/np.log(2))
    
    initial_rho = realistic_ft_ancilla(initial_rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    current_rho = initial_rho
        
    # apply the control stabilizer gates to current_rho depending on error parameters
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply K3
        current_rho = prob_line_rad_CNOT(current_rho, 7, 1, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 10, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply K3
        current_rho = line_rad_CNOT(current_rho, 7, 1, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 10, 6, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply K3
        current_rho = line_errored_CNOT(current_rho, 7, 1, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 10, 6, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 7, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 8, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 10, 9, qubit_error_probs, form = 'rho')
    else:
        K3 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
            CNOT(10, 6, 12), np.dot(CNOT(9, 5, 12), np.dot(CNOT(8, 2, 12), CNOT(7, 1, 12)))))
        current_rho = np.dot(K3, np.cot(current_rho, K3.conj().T))
    
    # hadamard for the ancilla
    ancilla_hadamard = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2))) 
    # apply the hadamard to the ancilla
    current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
    else:
        current_rho = current_rho
    
    # Apply state measurement error if spam_probs is not empty
    if spam_prob != None:
        current_rho = spam_error(current_rho, spam_prob, 10) # ancilla
        
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Define the measurement projection operators
    M0 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(zero_meas, np.identity(2))))
    M1 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(one_meas, np.identity(2))))
    
    all_meas = np.array([M0, M1])

    # find the probability to measure each case
    m0_prob = np.trace(np.dot(M0.conj().T, np.dot(M0, current_rho)))
    m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, current_rho)))

    all_probs = np.array([m0_prob, m1_prob])

    # find which measurement operator is measured based on their probabilities
    index = random.choices(all_probs, weights=all_probs, k=1)
    index = np.where(all_probs == index)[0][0]

    # apply correct measurement collapse of the density matrix
    rho_prime = np.dot(all_meas[index], np.dot(current_rho, all_meas[index].conj().T))/(all_probs[index])
    # Create our new density matrix after collapsing ancilla qubits
    
    rho = rho_prime
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # apply an error for time taken to collapse ancilla
        rho = rad_error(rho_prime, t1, t2, tg)

    # How many total qubits are in our vector representation
    n = int(np.log(len(rho))/np.log(2))

    ### Just so we can look at the measurement bit
    probs = np.array([])
    for i in range(len(rho)):
        probs = np.append(probs, rho[i,i])

    collapsed_state = collapse_ancilla(np.sqrt(probs), 10)
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
    measurement = bits[10:11]
    
    return rho, measurement

def realistic_measure_K4(initial_rho, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None):
    # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(initial_rho))/np.log(2))
    
    initial_rho = realistic_ft_ancilla(initial_rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    current_rho = initial_rho
        
    # apply the control stabilizer gates to current_rho depending on error parameters
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply K1
        current_rho = prob_line_rad_CZ(current_rho, 7, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 8, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 9, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 10, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply K1
        current_rho = line_rad_CZ(current_rho, 7, 3, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 8, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 9, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 10, 6, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply K1
        current_rho = line_errored_CZ(current_rho, 7, 3, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 8, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 9, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 10, 6, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 7, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 8, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 10, 9, qubit_error_probs, form = 'rho')
    else:
        K4 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
            CZ(10, 6, 12), np.dot(CZ(9, 5, 12), np.dot(CZ(8, 4, 12), CZ(7, 3, 12)))))
        current_rho = np.dot(K4, np.cot(current_rho, K4.conj().T))
    
    
    # hadamard for the ancilla
    ancilla_hadamard = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2))) 
    # apply the hadamard to the ancilla
    current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
    else:
        current_rho = current_rho
    
    # Apply state measurement error if spam_probs is not empty
    if spam_prob != None:
        current_rho = spam_error(current_rho, spam_prob, 10) # ancilla
        
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Define the measurement projection operators
    M0 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(zero_meas, np.identity(2))))
    M1 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(one_meas, np.identity(2))))
    
    all_meas = np.array([M0, M1])

    # find the probability to measure each case
    m0_prob = np.trace(np.dot(M0.conj().T, np.dot(M0, current_rho)))
    m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, current_rho)))

    all_probs = np.array([m0_prob, m1_prob])

    # find which measurement operator is measured based on their probabilities
    index = random.choices(all_probs, weights=all_probs, k=1)
    index = np.where(all_probs == index)[0][0]

    # apply correct measurement collapse of the density matrix
    rho_prime = np.dot(all_meas[index], np.dot(current_rho, all_meas[index].conj().T))/(all_probs[index])
    # Create our new density matrix after collapsing ancilla qubits
    
    rho = rho_prime
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # apply an error for time taken to collapse ancilla
        rho = rad_error(rho_prime, t1, t2, tg)

    # How many total qubits are in our vector representation
    n = int(np.log(len(rho))/np.log(2))

    ### Just so we can look at the measurement bit
    probs = np.array([])
    for i in range(len(rho)):
        probs = np.append(probs, rho[i,i])

    collapsed_state = collapse_ancilla(np.sqrt(probs), 10)
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
    measurement = bits[10:11]
    
    return rho, measurement


def realistic_measure_K5(initial_rho, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None):
    # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(initial_rho))/np.log(2))
    
    initial_rho = realistic_ft_ancilla(initial_rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    current_rho = initial_rho
        
    # apply the control stabilizer gates to current_rho depending on error parameters
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply K2
        current_rho = prob_line_rad_CZ(current_rho, 7, 0, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 8, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 9, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 10, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply K2
        current_rho = line_rad_CZ(current_rho, 7, 0, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 8, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 9, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 10, 6, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply K2
        current_rho = line_errored_CZ(current_rho, 7, 0, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 8, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 9, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 10, 6, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 7, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 8, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 10, 9, qubit_error_probs, form = 'rho')
    else:
        K5 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
            CZ(10, 6, 12), np.dot(CZ(9, 4, 12), np.dot(CZ(8, 2, 12), CZ(7, 0, 12)))))
        current_rho = np.dot(K5, np.cot(current_rho, K5.conj().T))
    
    # hadamard for the ancilla
    ancilla_hadamard = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2))) 
    # apply the hadamard to the ancilla
    current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
    else:
        current_rho = current_rho
    
    # Apply state measurement error if spam_probs is not empty
    if spam_prob != None:
        current_rho = spam_error(current_rho, spam_prob, 10) # ancilla
        
        
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Define the measurement projection operators
    M0 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(zero_meas, np.identity(2))))
    M1 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(one_meas, np.identity(2))))
    
    all_meas = np.array([M0, M1])

    # find the probability to measure each case
    m0_prob = np.trace(np.dot(M0.conj().T, np.dot(M0, current_rho)))
    m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, current_rho)))

    all_probs = np.array([m0_prob, m1_prob])

    # find which measurement operator is measured based on their probabilities
    index = random.choices(all_probs, weights=all_probs, k=1)
    index = np.where(all_probs == index)[0][0]

    # apply correct measurement collapse of the density matrix
    rho_prime = np.dot(all_meas[index], np.dot(current_rho, all_meas[index].conj().T))/(all_probs[index])
    # Create our new density matrix after collapsing ancilla qubits
    
    rho = rho_prime
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # apply an error for time taken to collapse ancilla
        rho = rad_error(rho_prime, t1, t2, tg)

    # How many total qubits are in our vector representation
    n = int(np.log(len(rho))/np.log(2))

    ### Just so we can look at the measurement bit
    probs = np.array([])
    for i in range(len(rho)):
        probs = np.append(probs, rho[i,i])

    collapsed_state = collapse_ancilla(np.sqrt(probs), 10)
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
    measurement = bits[10:11]
    
    return rho, measurement


def realistic_measure_K6(initial_rho, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None):
    # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    
    # How many total qubits are in our vector representation
    n = int(np.log(len(initial_rho))/np.log(2))
    
    initial_rho = realistic_ft_ancilla(initial_rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    current_rho = initial_rho
    
    # apply the control stabilizer gates to current_rho depending on error parameters
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply K3
        current_rho = prob_line_rad_CZ(current_rho, 7, 1, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 8, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 9, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 10, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply K3
        current_rho = line_rad_CZ(current_rho, 7, 1, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 8, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 9, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 10, 6, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 7, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 8, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 10, 9, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply K3
        current_rho = line_errored_CZ(current_rho, 7, 1, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 8, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 9, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 10, 6, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 7, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 8, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 10, 9, qubit_error_probs, form = 'rho')
    else:
        K6 = np.dot(np.dot(CNOT(10, 9, 12), np.dot(CNOT(9, 8, 12), CNOT(8, 7, 12))), np.dot(
            CZ(10, 6, 12), np.dot(CZ(9, 5, 12), np.dot(CZ(8, 2, 12), CZ(7, 1, 12)))))
        current_rho = np.dot(K6, np.cot(current_rho, K6.conj().T))
    
    # hadamard for the ancilla
    ancilla_hadamard = np.kron(np.identity(2**10), np.kron(hadamard, np.identity(2))) 
    # apply the hadamard to the ancilla
    current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[10], 10, n)
    else:
        current_rho = current_rho
    
    # Apply state measurement error if spam_probs is not empty
    if spam_prob != None:
        current_rho = spam_error(current_rho, spam_prob, 10) # ancilla
        
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Define the measurement projection operators
    M0 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(zero_meas, np.identity(2))))
    M1 = np.kron(np.identity(2**7), np.kron(np.identity(2**3), np.kron(one_meas, np.identity(2))))
    
    all_meas = np.array([M0, M1])

    # find the probability to measure each case
    m0_prob = np.trace(np.dot(M0.conj().T, np.dot(M0, current_rho)))
    m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, current_rho)))

    all_probs = np.array([m0_prob, m1_prob])

    # find which measurement operator is measured based on their probabilities
    index = random.choices(all_probs, weights=all_probs, k=1)
    index = np.where(all_probs == index)[0][0]

    # apply correct measurement collapse of the density matrix
    rho_prime = np.dot(all_meas[index], np.dot(current_rho, all_meas[index].conj().T))/(all_probs[index])
    # Create our new density matrix after collapsing ancilla qubits
    
    rho = rho_prime
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # apply an error for time taken to collapse ancilla
        rho = rad_error(rho_prime, t1, t2, tg)

    # How many total qubits are in our vector representation
    n = int(np.log(len(rho))/np.log(2))

    ### Just so we can look at the measurement bit
    probs = np.array([])
    for i in range(len(rho)):
        probs = np.append(probs, rho[i,i])

    collapsed_state = collapse_ancilla(np.sqrt(probs), 10)
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
    measurement = bits[10:11]
    
    return rho, measurement



### Apply the steane code fault tolerantly (initializes logical state and corrects for errors)
def realistic_ft_steane(initial_rho, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None):
    # initial_rho: initial density matrix of your 10 qubit system (7 data, 3 ancilla)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # qubit_error_probs: an array of the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # info: Do you want to print out debugging/helpful info?

    # How many total qubits are in our vector representation
    n = int(np.log(len(initial_rho))/np.log(2))

    # Apply state prep error if spam_probs is not empty
    if spam_prob != None:
        for index in range(n):
            initial_rho = spam_error(initial_rho, spam_prob, index)
    
    rho = initial_rho
    # - - - - - - - - - - # Z Error Correction # - - - - - - - - - - #

    # Measurement for K1 the first time
    rho, m11 = realistic_measure_K1(rho, t1, t2, tg, qubit_error_probs, spam_prob)

    # Measurement for K1 the second time
    rho, m12 = realistic_measure_K1(rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    # Measurement for K1 the third time only if first two results dont agree
    if m11 == m12:
        m1 = m11
    else:
        rho, m13 = realistic_measure_K1(rho, t1, t2, tg, qubit_error_probs, spam_prob)
        m1 = m13

    # Measurement for K2 the first time
    rho, m21 = realistic_measure_K2(rho, t1, t2, tg, qubit_error_probs, spam_prob)

    # Measurement for K2 the second time
    rho, m22 = realistic_measure_K2(rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    # Measurement for K2 the third time only if first two results dont agree
    if m21 == m22:
        m2 = m21
    else:
        rho, m23 = realistic_measure_K2(rho, t1, t2, tg, qubit_error_probs, spam_prob)
        m2 = m23

    # Measurement for K3 the first time
    rho, m31 = realistic_measure_K3(rho, t1, t2, tg, qubit_error_probs, spam_prob)

    # Measurement for K3 the second time
    rho, m32 = realistic_measure_K3(rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    # Measurement for K3 the third time only if first two results dont agree
    if m31 == m32:
        m3 = m31
    else:
        rho, m33 = realistic_measure_K3(rho, t1, t2, tg, qubit_error_probs, spam_prob)
        m3 = m33

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
        final_rho = rho

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-5-index-1)), np.identity(2**5))))

        current_rho = np.dot(operation, np.dot(rho, operation.conj().T))
        
        if qubit_error_probs is not None:
            current_rho = gate_error(current_rho, qubit_error_probs[index], index, n) # depolarizing error
        if ((t1!=None) and (t2!=None) and (tg!=None)):
            current_rho = rad_error(current_rho, t1, t2, tg) # apply an error for correction gate time

        final_rho = current_rho
    
    
    rho = final_rho
    
    # - - - - - - - - - - # X Error Correction # - - - - - - - - - - #

    
    # Measurement for K4 the first time
    rho, m41 = realistic_measure_K4(rho, t1, t2, tg, qubit_error_probs, spam_prob)

    # Measurement for K4 the second time
    rho, m42 = realistic_measure_K4(rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    # Measurement for K4 the third time only if first two results dont agree
    if m41 == m42:
        m4 = m41
    else:
        rho, m43 = realistic_measure_K4(rho, t1, t2, tg, qubit_error_probs, spam_prob)
        m4 = m43

    # Measurement for K5 the first time
    rho, m51 = realistic_measure_K5(rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    # Measurement for K5 the second time
    rho, m52 = realistic_measure_K5(rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    # Measurement for K5 the third time only if first two results dont agree
    if m51 == m52:
        m5 = m51
    else:
        rho, m53 = realistic_measure_K5(rho, t1, t2, tg, qubit_error_probs, spam_prob)
        m5 = m53

    # Measurement for K6 the first time
    rho, m61 = realistic_measure_K6(rho, t1, t2, tg, qubit_error_probs, spam_prob)

    # Measurement for K6 the second time
    rho, m62 = realistic_measure_K6(rho, t1, t2, tg, qubit_error_probs, spam_prob)
    
    # Measurement for K6 the third time only if first two results dont agree
    if m61 == m62:
        m6 = m61
    else:
        rho, m63 = realistic_measure_K6(rho, t1, t2, tg, qubit_error_probs, spam_prob)
        m6 = m63

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
        final_rho = rho

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_x, np.kron(
            np.identity(2**(n-5-index-1)), np.identity(2**5))))

        current_rho = np.dot(operation, np.dot(rho, operation.conj().T))
        
        if qubit_error_probs is not None:
            current_rho = gate_error(current_rho, qubit_error_probs[index], index, n) # depolarizing error
        if ((t1!=None) and (t2!=None) and (tg!=None)):
            current_rho = rad_error(current_rho, t1, t2, tg) # apply an error for correction gate time

        final_rho = current_rho
    
    return final_rho