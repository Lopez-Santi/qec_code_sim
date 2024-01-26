# This file will contain functions that are useful when implementing logical t1 testing for the Steane code from error models

import numpy as np
import random
from general_qec.qec_helpers import *
from general_qec.gates import *
from general_qec.errors import *

### implemenent the Steane code with depolarization, rad, and spam errors
def realistic_steane(initial_rho, t1=None, t2=None, tg=None, qubit_error_probs=None, spam_prob=None):
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
    
    # - - - - - - - - - - # Z Error Correction # - - - - - - - - - - #
    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    current_rho = np.dot(ancilla_hadamard, np.dot(initial_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
    else:
        current_rho = current_rho
        
    
    # apply the control stabilizer gates to current_rho depending on error parameters
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply K1 first:
        current_rho = prob_line_rad_CNOT(current_rho, 7, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 7, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 7, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 7, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        # apply K2:
        current_rho = prob_line_rad_CNOT(current_rho, 8, 0, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        # apply K3:
        current_rho = prob_line_rad_CNOT(current_rho, 9, 1, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 9, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply K1 first:
        current_rho = line_rad_CNOT(current_rho, 7, 3, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 7, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 7, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 7, 6, t1, t2, tg, form = 'rho')
        # apply K2:
        current_rho = line_rad_CNOT(current_rho, 8, 0, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 8, 6, t1, t2, tg, form = 'rho')
        # apply K3:
        current_rho = line_rad_CNOT(current_rho, 9, 1, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CNOT(current_rho, 9, 6, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply K1 first:
        current_rho = line_errored_CNOT(current_rho, 7, 3, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 7, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 7, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 7, 6, qubit_error_probs, form = 'rho')
        # apply K2:
        current_rho = line_errored_CNOT(current_rho, 8, 0, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 8, 6, qubit_error_probs, form = 'rho')
        # apply K3:
        current_rho = line_errored_CNOT(current_rho, 9, 1, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CNOT(current_rho, 9, 6, qubit_error_probs, form = 'rho')
    else:
        # Define the Stabilizer Operators as CNOT gates between line adjacent qubits 
        K1 = np.dot(CNOT(7, 3, 10), np.dot(CNOT(7, 4, 10), np.dot(CNOT(7, 5, 10), CNOT(7, 6, 10))))
        K2 = np.dot(CNOT(8, 0, 10), np.dot(CNOT(8, 2, 10), np.dot(CNOT(8, 4, 10), CNOT(8, 6, 10))))
        K3 = np.dot(CNOT(9, 1, 10), np.dot(CNOT(9, 2, 10), np.dot(CNOT(9, 5, 10), CNOT(9, 6, 10))))
        current_rho = np.dot(K1, np.dot(current_rho, K1.conj().T))
        current_rho = np.dot(K2, np.dot(current_rho, K2.conj().T))
        current_rho = np.dot(K3, np.dot(current_rho, K3.conj().T))
    
    
    # apply the second hadamard to the ancillas
    current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
    else:
        current_rho = current_rho
    
    # Apply state measurement error if spam_probs is not empty
    if spam_prob != None:
        current_rho = spam_error(current_rho, spam_prob, 7) # ancilla 0
        current_rho = spam_error(current_rho, spam_prob, 8) # ancilla 1
        current_rho = spam_error(current_rho, spam_prob, 9) # ancilla 2

    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Define the measurement projection operators
    M1 = np.kron(np.identity(2**7), np.kron(zero_meas, np.kron(zero_meas, zero_meas)))
    M2 = np.kron(np.identity(2**7), np.kron(zero_meas, np.kron(zero_meas, one_meas)))
    M3 = np.kron(np.identity(2**7), np.kron(zero_meas, np.kron(one_meas, zero_meas)))
    M4 = np.kron(np.identity(2**7), np.kron(zero_meas, np.kron(one_meas, one_meas)))
    M5 = np.kron(np.identity(2**7), np.kron(one_meas, np.kron(zero_meas, zero_meas)))
    M6 = np.kron(np.identity(2**7), np.kron(one_meas, np.kron(zero_meas, one_meas)))
    M7 = np.kron(np.identity(2**7), np.kron(one_meas, np.kron(one_meas, zero_meas)))
    M8 = np.kron(np.identity(2**7), np.kron(one_meas, np.kron(one_meas, one_meas)))

    all_meas = np.array([M1, M2, M3, M4, M5, M6, M7, M8])

    # find the probability to measure each case
    m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, current_rho)))
    m2_prob = np.trace(np.dot(M2.conj().T, np.dot(M2, current_rho)))
    m3_prob = np.trace(np.dot(M3.conj().T, np.dot(M3, current_rho)))
    m4_prob = np.trace(np.dot(M4.conj().T, np.dot(M4, current_rho)))
    m5_prob = np.trace(np.dot(M5.conj().T, np.dot(M5, current_rho)))
    m6_prob = np.trace(np.dot(M6.conj().T, np.dot(M6, current_rho)))
    m7_prob = np.trace(np.dot(M7.conj().T, np.dot(M7, current_rho)))
    m8_prob = np.trace(np.dot(M8.conj().T, np.dot(M8, current_rho)))

    all_probs = np.array([m1_prob, m2_prob, m3_prob, m4_prob, m5_prob, m6_prob, m7_prob, m8_prob])

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

    ### Just so we can look at the measurement bits
    probs = np.array([])
    for i in range(len(rho)):
        probs = np.append(probs, rho[i,i])

    collapsed_state = collapse_ancilla(np.sqrt(probs), 10)
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]

    # find index
    m_one = 0
    m_two = 0
    m_three = 0
    if bits[7] == '1':
        m_one = 1
    if bits[8] == '1':
        m_two = 1
    if bits[9] == '1':
        m_three = 1

    # Which qubit do we perform the Z gate on
    index = (m_one * 2**2) + (m_three * 2**1) + (m_two * 2**0) - 1

    # if no error occurs we dont need to apply a correction
    if index == -1:
        final_rho = rho

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        current_rho = np.dot(operation, np.dot(rho, operation.conj().T))
        
        if qubit_error_probs is not None:
            current_rho = gate_error(current_rho, qubit_error_probs[index], index, n) # depolarizing error
        if ((t1!=None) and (t2!=None) and (tg!=None)):
            current_rho = rad_error(current_rho, t1, t2, tg) # apply an error for correction gate time

        final_rho = current_rho
    
    # Reset the ancillas by projecting to the |00><00| basis    
    # apply correct measurement collapse of the density matrix
    ancilla_reset_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, final_rho)))
    reset_rho = np.dot(M1, np.dot(final_rho, M1.conj().T))/(ancilla_reset_prob)
    
    initial_rho = reset_rho
    
    # Apply state prep error if spam_probs is not empty
    if spam_prob != None:
        for index in range(n):
            initial_rho = spam_error(initial_rho, spam_prob, index)
    
    # - - - - - - - - - - # X Error Correction # - - - - - - - - - - #

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    current_rho = np.dot(ancilla_hadamard, np.dot(initial_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
    else:
        current_rho = current_rho
        
    
    # apply the control stabilizer gates to current_rho depending on error parameters
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply K1 first:
        current_rho = prob_line_rad_CZ(current_rho, 7, 3, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 7, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 7, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 7, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        # apply K2:
        current_rho = prob_line_rad_CZ(current_rho, 8, 0, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 8, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 8, 4, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CNOT(current_rho, 8, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
        # apply K3:
        current_rho = prob_line_rad_CZ(current_rho, 9, 1, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 9, 2, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 9, 5, t1, t2, tg, qubit_error_probs, form = 'rho')
        current_rho = prob_line_rad_CZ(current_rho, 9, 6, t1, t2, tg, qubit_error_probs, form = 'rho')
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply K1 first:
        current_rho = line_rad_CZ(current_rho, 7, 3, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 7, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 7, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 7, 6, t1, t2, tg, form = 'rho')
        # apply K2:
        current_rho = line_rad_CZ(current_rho, 8, 0, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 8, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 8, 4, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 8, 6, t1, t2, tg, form = 'rho')
        # apply K3:
        current_rho = line_rad_CZ(current_rho, 9, 1, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 9, 2, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 9, 5, t1, t2, tg, form = 'rho')
        current_rho = line_rad_CZ(current_rho, 9, 6, t1, t2, tg, form = 'rho')
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply K1 first:
        current_rho = line_errored_CZ(current_rho, 7, 3, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 7, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 7, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 7, 6, qubit_error_probs, form = 'rho')
        # apply K2:
        current_rho = line_errored_CZ(current_rho, 8, 0, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 8, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 8, 4, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 8, 6, qubit_error_probs, form = 'rho')
        # apply K3:
        current_rho = line_errored_CZ(current_rho, 9, 1, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 9, 2, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 9, 5, qubit_error_probs, form = 'rho')
        current_rho = line_errored_CZ(current_rho, 9, 6, qubit_error_probs, form = 'rho')
    else:
        # Define the Stabilizer Operators as CNOT gates between line adjacent qubits 
        K4 = np.dot(CZ(7, 3, 10), np.dot(CZ(7, 4, 10), np.dot(CZ(7, 5, 10), CZ(7, 6, 10))))
        K5 =np.dot(CZ(8, 0, 10), np.dot(CZ(8, 2, 10), np.dot(CZ(8, 4, 10), CZ(8, 6, 10))))
        K6 =np.dot(CZ(9, 1, 10), np.dot(CZ(9, 2, 10), np.dot(CZ(9, 5, 10), CZ(9, 6, 10)))) 
        
        current_rho = np.dot(K4, np.dot(current_rho, K4.conj().T))
        current_rho = np.dot(K5, np.dot(current_rho, K5.conj().T))
        current_rho = np.dot(K6, np.dot(current_rho, K6.conj().T))
    
    
    # apply the second hadamard to the ancillas
    current_rho = np.dot(ancilla_hadamard, np.dot(current_rho, ancilla_hadamard.conj().T))
    
    # Apply error gates depending on error parameters 
    if ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1!=None) and (t2!=None) and (tg!=None)) and (qubit_error_probs is None):
        # apply rad error
        current_rho = rad_error(current_rho, t1, t2, tg)
    elif ((t1==None) and (t2==None) and (tg==None)) and (qubit_error_probs is not None):
        # apply depolarizing error
        current_rho = gate_error(current_rho, qubit_error_probs[7], 7, n)
        current_rho = gate_error(current_rho, qubit_error_probs[8], 8, n)
        current_rho = gate_error(current_rho, qubit_error_probs[9], 9, n)
    else:
        current_rho = current_rho
    
    # Apply state measurement error if spam_probs is not empty
    if spam_prob != None:
        current_rho = spam_error(current_rho, spam_prob, 7) # ancilla 0
        current_rho = spam_error(current_rho, spam_prob, 8) # ancilla 1
        current_rho = spam_error(current_rho, spam_prob, 9) # ancilla 2
    
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Define the measurement projection operators
    M1 = np.kron(np.identity(2**7), np.kron(zero_meas, np.kron(zero_meas, zero_meas)))
    M2 = np.kron(np.identity(2**7), np.kron(zero_meas, np.kron(zero_meas, one_meas)))
    M3 = np.kron(np.identity(2**7), np.kron(zero_meas, np.kron(one_meas, zero_meas)))
    M4 = np.kron(np.identity(2**7), np.kron(zero_meas, np.kron(one_meas, one_meas)))
    M5 = np.kron(np.identity(2**7), np.kron(one_meas, np.kron(zero_meas, zero_meas)))
    M6 = np.kron(np.identity(2**7), np.kron(one_meas, np.kron(zero_meas, one_meas)))
    M7 = np.kron(np.identity(2**7), np.kron(one_meas, np.kron(one_meas, zero_meas)))
    M8 = np.kron(np.identity(2**7), np.kron(one_meas, np.kron(one_meas, one_meas)))

    all_meas = np.array([M1, M2, M3, M4, M5, M6, M7, M8])

    # find the probability to measure each case
    m1_prob = np.trace(np.dot(M1.conj().T, np.dot(M1, current_rho)))
    m2_prob = np.trace(np.dot(M2.conj().T, np.dot(M2, current_rho)))
    m3_prob = np.trace(np.dot(M3.conj().T, np.dot(M3, current_rho)))
    m4_prob = np.trace(np.dot(M4.conj().T, np.dot(M4, current_rho)))
    m5_prob = np.trace(np.dot(M5.conj().T, np.dot(M5, current_rho)))
    m6_prob = np.trace(np.dot(M6.conj().T, np.dot(M6, current_rho)))
    m7_prob = np.trace(np.dot(M7.conj().T, np.dot(M7, current_rho)))
    m8_prob = np.trace(np.dot(M8.conj().T, np.dot(M8, current_rho)))

    all_probs = np.array([m1_prob, m2_prob, m3_prob, m4_prob, m5_prob, m6_prob, m7_prob, m8_prob])

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

    ### Just so we can look at the measurement bits that we just collapsed to just now
    probs = np.array([])
    for i in range(len(rho)):
        probs = np.append(probs, rho[i,i])

    collapsed_state = collapse_ancilla(np.sqrt(probs), 10)
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]

    # find index
    m_four = 0
    m_five = 0
    m_six = 0
    if bits[7] == '1':
        m_four = 1
    if bits[8] == '1':
        m_five = 1
    if bits[9] == '1':
        m_six = 1

    # Which qubit do we perform the Z gate on
    index = (m_four * 2**2) + (m_six * 2**1) + (m_five * 2**0) - 1

    # if no error occurs we dont need to apply a correction
    if index == -1:
        final_rho = rho

    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_x, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        current_rho = np.dot(operation, np.dot(rho, operation.conj().T))
        
        if qubit_error_probs is not None:
            current_rho = gate_error(current_rho, qubit_error_probs[index], index, n) # depolarizing error
        if ((t1!=None) and (t2!=None) and (tg!=None)):
            current_rho = rad_error(current_rho, t1, t2, tg) # apply an error for correction gate time

        final_rho = current_rho
    
    return final_rho
