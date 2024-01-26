"""
This file contains different universal error operations that can be used on any number of qubits.
"""
import random
import numpy as np
from general_qec.gates import hadamard, cnot, flipped_cnot, cz
from general_qec.gates import sigma_x, sigma_y, sigma_z, sigma_I


def random_single_qubit_error(logical_state, single_qubit_unitary, qubit_range=None):
    """
    Applies a single qubit unitary to a uniformly randomly selected qubit in the
    range - can choose "no error" with the same probability as any given qubit
    if a range is not specified (so, probability of no error falls like 1/n).

    * logical_state: The logical state of the N qubit system you wish to
    apply the error to - e.g. np.kron(one, zero)
    * single_qubit_unitary: the gate applied
    * qubit_range: The indices you want to consider in your error application
    (starts at 0) - e.g., a *tuple* like (0, 2)
    """
    # total number of qubits in your system
    nqubits = int(np.log(len(logical_state))/np.log(2))

    # Choose the index of the qubit you want to apply the error to.
    error_index = -1
    if qubit_range is not None:
        error_index = random.randint(qubit_range[0], qubit_range[1])
    else:
        error_index = random.randint(-1, nqubits-1)

    # Apply the error depending on the index
    errored_logical_state = logical_state
    if error_index > -1:
        error_gate = np.kron(
            np.identity(2**(error_index)),
            np.kron(single_qubit_unitary, np.identity(2**(nqubits-error_index-1)))
        )
        errored_logical_state = np.dot(error_gate, logical_state)

    return errored_logical_state, error_index


def random_qubit_x_error(logical_state, qubit_range=None):
    """
    Applies a Pauli X to a uniformly randomly selected qubit in the
    range - can choose "no error" with the same probability as any given qubit
    if a range is not specified (so, probability of no error falls like 1/n).

    * logical_state: The logical state of the N qubit system you wish to
    apply the error to - e.g. np.kron(one, zero)
    * qubit_range: The indices you want to consider in your error application
    (starts at 0) - e.g., a *tuple* like (0, 2)
    """
    return random_single_qubit_error(logical_state, sigma_x, qubit_range)


def random_qubit_z_error(logical_state, qubit_range=None):
    """
    Applies a Pauli Z to a uniformly randomly selected qubit in the
    range - can choose "no error" with the same probability as any given qubit
    if a range is not specified (so, probability of no error falls like 1/n).

    * logical_state: The logical state of the N qubit system you wish to
    apply the error to - e.g. np.kron(one, zero)
    * qubit_range: The indices you want to consider in your error application
    (starts at 0) - e.g., a *tuple* like (0, 2)
    """
    return random_single_qubit_error(logical_state, sigma_z, qubit_range)


### - Gates which contain probability for errors (line connectivity) - ###

def gate_error(rho, error_prob, index, n):
    """
    Take a density matrix after a perfect operation and apply a Krauss error
    gate based on probability of an error.

    * rho: density matrix of qubit system after perfect gate was applied
    * error_prob: probability for gate operation error
    * index: index of qubit that gate was applied (target qubit in this case)
    * n: total number of qubits in your system
    """
    # qubit error rates:
    KD0 = np.sqrt(1-error_prob) * sigma_I # pylint: disable=invalid-name
    KD1 = np.sqrt(error_prob/3) * sigma_x # pylint: disable=invalid-name
    KD2 = np.sqrt(error_prob/3) * sigma_z # pylint: disable=invalid-name
    KD3 = np.sqrt(error_prob/3) * sigma_y # pylint: disable=invalid-name

    # qubit error gates
    KD0 = np.kron(np.identity(2**(index)), np.kron(KD0, np.identity(2**(n-index-1)))) # pylint: disable=invalid-name
    KD1 = np.kron(np.identity(2**(index)), np.kron(KD1, np.identity(2**(n-index-1)))) # pylint: disable=invalid-name
    KD2 = np.kron(np.identity(2**(index)), np.kron(KD2, np.identity(2**(n-index-1)))) # pylint: disable=invalid-name
    KD3 = np.kron(np.identity(2**(index)), np.kron(KD3, np.identity(2**(n-index-1)))) # pylint: disable=invalid-name

    # apply error gates
    d_rho = \
        np.dot(KD0, np.dot(rho, KD0.conj().T)) + \
        np.dot(KD1, np.dot(rho, KD1.conj().T)) + \
        np.dot(KD2, np.dot(rho, KD2.conj().T)) + \
        np.dot(KD3, np.dot(rho, KD3.conj().T))

    return d_rho


### - - - CNOT Gates - - - ###

def errored_adj_CNOT(rho, control, target, qubit_error_probs): # pylint: disable=invalid-name
    """
    Apply an adjacent CNOT gate between 2 qubits in a system with Krauss
    errors. The Krauss error is applied to the target qubit only.

    * rho: the desnity matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be > than control)
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    """
    t1, t2, tg, form = 1e9, 1e9, 1e-9, 'rho' # pylint: disable=invalid-name
    error_rho = prob_line_rad_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs, form
    )
    return error_rho


def errored_non_adj_CNOT(rho, control, target, qubit_error_probs): # pylint: disable=invalid-name
    """
    Apply a non-adjacent CNOT gate between 2 qubits in a system with line
    connectivity and Krauss errors. The Krauss error is applied to the
    target qubit only, but it may be applied for every CNOT in the swap
    operations.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be > control)
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    """
    t1, t2, tg, form = 1e9, 1e9, 1e-9, 'rho' # pylint: disable=invalid-name
    error_rho = prob_line_rad_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs, form
    )
    return error_rho


def errored_flipped_adj_CNOT(rho, control, target, qubit_error_probs): # pylint: disable=invalid-name
    """
    Apply an adjacent flipped CNOT gate between 2 qubits in a system with line
    connectivity and Krauss errors. The Krauss error is applied to the
    target qubit only.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be > control)
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    """
    t1, t2, tg, form = 1e9, 1e9, 1e-9, 'rho' # pylint: disable=invalid-name
    error_rho = prob_line_rad_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs, form
    )
    return error_rho


def errored_flipped_non_adj_CNOT(rho, control, target, qubit_error_probs): # pylint: disable=invalid-name
    """
    Apply a non-adjacent flipped CNOT gate between 2 qubits in a system with
    line connectivity and Krauss errors. The Krauss error is applied to the
    target qubit only, but it may be applied for every CNOT in the swap
    operations.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be > control)
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    """
    t1, t2, tg, form = 1e9, 1e9, 1e-9, 'rho' # pylint: disable=invalid-name
    error_rho = prob_line_rad_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs, form
    )
    return error_rho


def line_errored_CNOT(state, control, target, qubit_error_probs, form='psi'): # pylint: disable=invalid-name
    """
    Apply an arbitrary CNOT gate between 2 qubits in a system with line
    connectivity and Krauss errors. The Krauss error is applied to the target
    qubit only, but it may be applied for every CNOT in the swap operations.

    * state: the vector state representation or density matrix representation
    of your system (default is state vector, see `form` arg for more)
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    * form: if 'psi' we compute the density matrix on the fly, else we assume
    we were passed density matrix.
    """
    t1, t2, tg = 1e9, 1e9, 1e-9 # pylint: disable=invalid-name
    error_rho = prob_line_rad_CNOT(
        state, control, target, t1, t2, tg, qubit_error_probs, form
    )
    return error_rho


### - - - - - CZ Gates - - - - - ###

def errored_adj_CZ(rho, control, target, qubit_error_probs): # pylint: disable=invalid-name
    """
    Apply an adjacent CZ gate between 2 qubits in a system with Krauss errors.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    """
    t1, t2, tg, form = 1e9, 1e9, 1e-9, 'rho' # pylint: disable=invalid-name
    error_rho = prob_line_rad_CZ(
        rho, control, target, t1, t2, tg, qubit_error_probs, form
    )
    return error_rho


def errored_non_adj_CZ(rho, control, target, qubit_error_probs): # pylint: disable=invalid-name
    """
    Implement a non-adjacent CZ gate between 2 qubits in a system with Krauss
    errors.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be a larger index than control)
    * qubit_error_probs: an array of the probability for errors of each qubit in your system
    """
    t1, t2, tg, form = 1e9, 1e9, 1e-9, 'rho' # pylint: disable=invalid-name
    error_rho = prob_line_rad_CZ(
        rho, control, target, t1, t2, tg, qubit_error_probs, form
    )
    return error_rho


def line_errored_CZ(state, control, target, qubit_error_probs, form='psi'): # pylint: disable=invalid-name
    """
    Implement an errord-CZ gate between 2 qubits with line connectivity and
    Krauss errors. The Krauss error is applied to the target qubit only, but it
    may be applied for every CZ in the swap operations.

    * state: the vector state representation or density matrix representation
    of your system (default is state vector, see `form` arg for more)
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    * form: if 'psi' we compute the density matrix on the fly, else we assume
    we were passed density matrix.
    """
    t1, t2, tg = 1e9, 1e9, 1e-9 # pylint: disable=invalid-name
    final_rho = prob_line_rad_CZ(
        state, control, target, t1, t2, tg, qubit_error_probs, form
    )
    return final_rho


### - Gates with rad (relaxation and dephasing) errors (line connectivity) - ###

def rad_error(rho, t1, t2, tg): # pylint: disable=invalid-name,too-many-locals
    """
    Takes the density matrix after a perfect operation and applies a relaxation
    and dephasing (rad) error gate based on t1, t2, and tg (gate length).

    * rho: density matrix of qubit system after perfect gate was applied
    * t1: the relaxation time of the qubits
    * t2: the dephasing time of the qubits
    * tg: length in time of the logical gate you are applying
    """
    # TODO: we don't yet handle only depolarizing or dephasing... bail if one is None
    if ((t1 is None) or (t2 is None) or (tg is None)):
        return rho
    # total number of qubits in your system
    tot_qubits = int(np.log(len(rho))/np.log(2))

    p_t1 = 1-np.exp(-tg/t1) # find the probability of relaxation
    p_t2 = 1-np.exp(-tg/t2) # find the probability of dephasing

    # decay channel:
    k_0 = np.array([[1, 0], [0, np.sqrt(1-p_t1)]])
    k_1 = np.array([[0, np.sqrt(p_t1)], [0, 0]])

    # apply decay operators
    for i in range(tot_qubits):
        operator_0 = np.kron(np.identity(2**i), np.kron(k_0, np.identity(2**(tot_qubits - i - 1))))
        operator_1 = np.kron(np.identity(2**i), np.kron(k_1, np.identity(2**(tot_qubits - i - 1))))
        rho = np.dot(operator_0, np.dot(rho, operator_0.conj().T)) + np.dot(
            operator_1, np.dot(rho, operator_1.conj().T))

    # dephasing channel:
    k_2 = np.sqrt(1-p_t2) * np.identity(2)
    k_3 = np.array([[np.sqrt(p_t2), 0], [0, 0]])
    k_4 = np.array([[0, 0], [0, np.sqrt(p_t2)]])

    # apply dephasing operators
    for i in range(tot_qubits):
        operator_2 = np.kron(np.identity(2**i), np.kron(k_2, np.identity(2**(tot_qubits - i - 1))))
        operator_3 = np.kron(np.identity(2**i), np.kron(k_3, np.identity(2**(tot_qubits - i - 1))))
        operator_4 = np.kron(np.identity(2**i), np.kron(k_4, np.identity(2**(tot_qubits - i - 1))))

        rho = np.dot(operator_2, np.dot(rho, operator_2.conj().T)) + \
            np.dot(operator_3, np.dot(rho, operator_3.conj().T)) + \
            np.dot(operator_4, np.dot(rho, operator_4.conj().T))

    final_rho = rho

    return final_rho #np.round(final_rho, 9)


### - - - CNOT GATES - - - ###

def rad_adj_CNOT(rho, control, target, t1, t2, tg): # pylint: disable=invalid-name,too-many-arguments
    """
    Apply an adjacent CNOT gate between 2 qubits in a system with line
    connectivity and rad errors.

    * rho: the desnity matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be > control)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    """
    tot_qubits = int(np.log(len(rho))/np.log(2))
    qubit_error_probs = [0.] * tot_qubits
    error_rho = prob_line_rad_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs, 'rho'
    )
    return error_rho


def rad_non_adj_CNOT(rho, control, target, t1, t2, tg): # pylint: disable=invalid-name,too-many-arguments
    """
    Apply a non-adjacent CNOT gate between 2 qubits in a system with line
    connectivity and rad errors.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be a larger index than control)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    """
    tot_qubits = int(np.log(len(rho))/np.log(2))
    qubit_error_probs = [0.] * tot_qubits
    error_rho = prob_line_rad_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs, 'rho'
    )
    return error_rho


def rad_flipped_adj_CNOT(rho, control, target, t1, t2, tg): # pylint: disable=invalid-name,too-many-arguments
    """
    Apply an adjacent flipped CNOT gate between 2 qubits in a system with
    line connectivity and relaxation and dephasing errors.

    # rho: the density matrix representation of your system
    # control: control qubit index (starting from 0)
    # target: target qubit index (starting from 0) (must be > control)
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations
    """
    tot_qubits = int(np.log(len(rho))/np.log(2))
    qubit_error_probs = [0.] * tot_qubits
    error_rho = prob_line_rad_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs, 'rho'
    )
    return error_rho


def rad_flipped_non_adj_CNOT(rho, control, target, t1, t2, tg): # pylint: disable=invalid-name,too-many-arguments
    """
    Apply a non-adjacent flipped CNOT gate between 2 qubits in a system with
    line connectivity and relaxation and dephasing errors.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be > control)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    """
    tot_qubits = int(np.log(len(rho))/np.log(2))
    qubit_error_probs = [0.] * tot_qubits
    error_rho = prob_line_rad_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs, 'rho'
    )
    return error_rho


def line_rad_CNOT(state, control, target, t1, t2, tg, form='psi'): # pylint: disable=invalid-name,too-many-arguments
    """
    Implement a rad CNOT gate between 2 qubits depending on your control and
    target qubit.

    * state: the vector state representation or density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * form: either 'psi' for vector representation or 'rho' for density matrix that user inputs
    """
    tot_qubits = int(np.log(len(state))/np.log(2))
    qubit_error_probs = [0.] * tot_qubits
    error_rho = prob_line_rad_CNOT(
        state, control, target, t1, t2, tg, qubit_error_probs, form
    )
    return error_rho


### - - - CZ GATES - - - ###

def rad_adj_CZ(rho, control, target, t1, t2, tg): # pylint: disable=invalid-name,too-many-arguments
    """
    Implement a RAD CZ gate between 2 adjacent qubits in a system.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    """
    tot_qubits = int(np.log(len(rho))/np.log(2))
    qubit_error_probs = [0.] * tot_qubits
    final_rho = prob_line_rad_CZ(
        rho, control, target, t1, t2, tg, qubit_error_probs, 'rho'
    )
    return final_rho


def rad_non_adj_CZ(rho, control, target, t1, t2, tg): # pylint: disable=invalid-name,too-many-arguments
    """
    Implement a non-adjacent RAD CZ gate between 2 qubits in a system.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    """
    tot_qubits = int(np.log(len(rho))/np.log(2))
    qubit_error_probs = [0.] * tot_qubits
    final_rho = prob_line_rad_CZ(
        rho, control, target, t1, t2, tg, qubit_error_probs, 'rho'
    )
    return final_rho


def line_rad_CZ(state, control, target, t1, t2, tg, form='psi'): # pylint: disable=invalid-name,too-many-arguments
    """
    Implement a RAD CZ gate between 2 qubits depending on your control and
    target qubit.

    * state: the vector state representation or density matrix representation
    of your system (default is state vector)
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * form: either 'psi' for vector representation or 'rho' for density matrix
    that user inputs
    """
    tot_qubits = int(np.log(len(state))/np.log(2))
    qubit_error_probs = [0.] * tot_qubits
    final_rho = prob_line_rad_CZ(
        state, control, target, t1, t2, tg, qubit_error_probs, form
    )
    return final_rho


### - Gates which contain Krauss and rad (relaxation & dephasing) errors (line connectivity) - ###

def check_apply_krauss(qubit_error_probs):
    """Check the Krauss error probabilities vector for error application conditions."""
    return (qubit_error_probs is not None) and (sum(qubit_error_probs) > 0)


def check_apply_rad(t1, t2, tg): # pylint: disable=invalid-name
    """Check qubit and gate timing info for RAD error applicaiton conditions."""
    return (t1 is not None) and (t2 is not None) and (tg is not None)


### - - - CNOT GATES - - - ###

def prob_rad_adj_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs
    ): # pylint: disable=invalid-name,too-many-arguments
    """
    Apply an adjacent CNOT gate between 2 qubits in system with line
    connectivity with Krauss errors and relaxation and dephasing errors.

    * rho: the desnity matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be > control)
    * t1: The relaxation time of each physical qubit in your system (sec)
    * t2: The dephasing time of each physical qubit in your system (sec)
    * tg: The gate time of your gate operations (sec)
    * qubit_error_probs: an array of the probability for Krauss errors of each
    qubit in your system
    """
    apply_krauss_errors = check_apply_krauss(qubit_error_probs)
    apply_rad_errors = check_apply_rad(t1, t2, tg)
    tot_qubits = int(np.log(len(rho))/np.log(2))

    # Adds the dimensions needed depending on the tot_qubits
    # exponent used to tensor the left side identity matrix for our full system
    n1 = control                 # pylint: disable=invalid-name
    # exponent used to tensor the right side identity matrix for our full system
    n2 = tot_qubits - target - 1 # pylint: disable=invalid-name

    gate = np.kron(np.identity(2**(n1)), np.kron(cnot, np.identity(2**(n2))))

    # applies the perfect gate to our density matrix
    final_rho = np.dot(gate, np.dot(rho, gate.conj().T))
    # apply our error gates and find the new density matrix
    if apply_krauss_errors:
        final_rho = gate_error(final_rho, qubit_error_probs[target], target, tot_qubits)
    if apply_rad_errors:
        final_rho = rad_error(final_rho, t1, t2, tg)

    return final_rho


def prob_rad_non_adj_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs
    ): # pylint: disable=invalid-name,too-many-arguments
    """
    Apply a non-adjacent CNOT gate between 2 qubits in system with line
    connectivity with Krauss errors and relaxation and dephasing errors.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be > control)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * qubit_error_probs: an array of the probability for errors of each qubit in your system
    """
    apply_krauss_errors = check_apply_krauss(qubit_error_probs)
    apply_rad_errors = check_apply_rad(t1, t2, tg)
    tot_qubits = int(np.log(len(rho))/np.log(2))

    # used to index over all gates neeeded to compose final gate
    p = target - control        # pylint: disable=invalid-name

    # Pads the dimensions needed depending on the tot_qubits
    # exponent used to tensor the left side identity matrix for our full system
    n1 = control                   # pylint: disable=invalid-name
    # exponent used to tensor the right side identity matrix for our full system
    n2 = tot_qubits - target - 1   # pylint: disable=invalid-name

    # accumulate errors
    error_rho = rho

    # Applies the gates twice (square in our formula)
    for _ in range(0,2):
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))
            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0)
            # sets the current gate
            gate = all_dots[j]
            # applies the perfect gate to our density matrix
            error_rho = np.dot(gate, np.dot(error_rho, gate.conj().T))
            # apply our error gate and find the new density matrix
            if apply_krauss_errors:
                error_rho = gate_error(
                    error_rho, qubit_error_probs[j+control+1], j+control+1, tot_qubits)
            if apply_rad_errors:
                error_rho = rad_error(error_rho, t1, t2, tg)

        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            error_rho = np.dot(gate, np.dot(error_rho, gate.conj().T))
            # apply our error gate and find the new density matrix
            if apply_krauss_errors:
                error_rho = gate_error(
                    error_rho, qubit_error_probs[target-j-2+1], target-j-2+1, tot_qubits)
            if apply_rad_errors:
                error_rho = rad_error(error_rho, t1, t2, tg)

    return error_rho # returns the density matrix of your system


def prob_rad_flipped_adj_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs
    ): # pylint: disable=invalid-name,too-many-arguments
    """
    Apply an adjacent flipped CNOT gate between 2 qubits in system with line
    connectivity and Krauss and relaxation and dephasing errors.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be > control)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    """
    apply_krauss_errors = check_apply_krauss(qubit_error_probs)
    apply_rad_errors = check_apply_rad(t1, t2, tg)
    tot_qubits = int(np.log(len(rho))/np.log(2))

    # Adds the dimensions needed depending on the tot_qubits
    # exponent used to tensor the left side identity matrix for our full system
    n1 = target                     # pylint: disable=invalid-name
    # exponent used to tensor the right side identity matrix for our full system
    n2 = tot_qubits - control - 1   # pylint: disable=invalid-name

    gate = np.kron(np.identity(2**(n1)), np.kron(flipped_cnot, np.identity(2**(n2))))

    # applies the perfect gate to our density matrix
    final_rho = np.dot(gate, np.dot(rho, gate.conj().T))
    # apply errors and find the new density matrix
    if apply_krauss_errors:
        final_rho = gate_error(final_rho, qubit_error_probs[target], target, tot_qubits)
    if apply_rad_errors:
        final_rho = rad_error(final_rho, t1, t2, tg)

    return final_rho


def prob_rad_flipped_non_adj_CNOT(
        rho, control, target, t1, t2, tg, qubit_error_probs
    ): # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    """
    Apply a non-adjacent flipped CNOT gate between 2 qubits in system with
    line connectivity and Krauss and relaxation and dephasing errors.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0) (must be < control)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    """
    apply_krauss_errors = check_apply_krauss(qubit_error_probs)
    apply_rad_errors = check_apply_rad(t1, t2, tg)
    tot_qubits = int(np.log(len(rho))/np.log(2))

    # used to index over all gates neeeded to compose final gate
    p = np.abs(target - control)   # pylint: disable=invalid-name

    # Adds the dimensions needed depending on the tot_qubits
    # exponent used to tensor the left side identity matrix for our full system
    n1 = target                     # pylint: disable=invalid-name
    # exponent used to tensor the right side identity matrix for our full system
    n2 = tot_qubits - control - 1   # pylint: disable=invalid-name

    # accumulate errors
    error_rho = rho

    # Applies the gates twice (square in our formula)
    for _ in range(0,2):
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(p-j-1)), np.kron(flipped_cnot, np.identity(2**(j))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))

            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0)
            gate = all_dots[j] # sets the current gate
            # applies the perfect gate to our density matrix
            error_rho = np.dot(gate, np.dot(error_rho, gate.conj().T))
            # apply our error gates and find the new density matrix
            if apply_krauss_errors:
                error_rho = gate_error(
                    error_rho, qubit_error_probs[control-j-1], control-j-1, tot_qubits)
            if apply_rad_errors:
                error_rho = rad_error(error_rho, t1, t2, tg)

        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            error_rho = np.dot(gate, np.dot(error_rho, gate.conj().T))
            # apply our error gates and find the new density matrix
            if apply_krauss_errors:
                error_rho = gate_error(
                    error_rho, qubit_error_probs[target+j+1], target+j+1, tot_qubits)
            if apply_rad_errors:
                error_rho = rad_error(error_rho, t1, t2, tg)

    return error_rho # returns the density matrix of your system


def prob_line_rad_CNOT(
        state, control, target, t1, t2, tg, qubit_error_probs,
        form='psi', remove_small_values=True
    ): # pylint: disable=invalid-name,too-many-arguments
    """
    Implement a CNOT gate between 2 qubits in system with line
    connectivity with Krauss and relaxation and dephasing errors.
    The Krauss error is applied to the target qubit only. Returns
    the density matrix after the gate with appropriate errors.

    * state: the vector state representation or density matrix representation
    of your system (default assumes a state vector)
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    * form: either 'psi' for vector representation or 'rho' for density matrix
    that user inputs
    * remove_small_values: trim entries below 1e-15 from the output density
    matrix
    """
    assert control != target, "Control cannot equal target in `prob_line_rad_CNOT()`"
    # if the form is 'psi' find the density matrix
    if form == 'psi':
        rho = np.kron(state, state[np.newaxis].conj().T)
    else:
        rho = state

    # First check if it is a normal CNOT or a flipped CNOT gate
    if control < target:
        # Check if adjacent
        if target - control == 1:
            final_rho = prob_rad_adj_CNOT(
                rho, control, target, t1, t2, tg, qubit_error_probs
            )
        else:
            final_rho = prob_rad_non_adj_CNOT(
                rho, control, target, t1, t2, tg, qubit_error_probs
            )
    elif control > target:
        # Check if adjacent
        if control - target == 1:
            final_rho = prob_rad_flipped_adj_CNOT(
                rho, control, target, t1, t2, tg, qubit_error_probs
            )
        else:
            final_rho = prob_rad_flipped_non_adj_CNOT(
                rho, control, target, t1, t2, tg, qubit_error_probs
            )

    if remove_small_values:
        final_rho[np.abs(final_rho) < 1e-15] = 0

    return final_rho


### - - - CZ GATES - - - ###

def prob_rad_adj_CZ(
        rho, control, target, t1, t2, tg, qubit_error_probs
    ): # pylint: disable=invalid-name,too-many-arguments
    """
    Implement a rad CZ gate between 2 adjacent qubits in a system with Krauss
    and relaxation and dephasing errors.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * qubit_error_probs: an array of the probability for errors of each qubit in your system
    """
    apply_krauss_errors = check_apply_krauss(qubit_error_probs)
    apply_rad_errors = check_apply_rad(t1, t2, tg)
    tot_qubits = int(np.log(len(rho))/np.log(2))

    # Adds the dimensions needed depending on the tot_qubits
    if control < target:
        # exponent used to tensor the left side identity matrix for our full system
        n1 = control                   # pylint: disable=invalid-name
        # exponent used to tensor the right side identity matrix for our full system
        n2 = tot_qubits - target - 1   # pylint: disable=invalid-name
    else:
        # exponent used to tensor the left side identity matrix for our full system
        n1 = target                    # pylint: disable=invalid-name
        # exponent used to tensor the right side identity matrix for our full system
        n2 = tot_qubits - control - 1  # pylint: disable=invalid-name

    gate = np.kron(np.identity(2**(n1)), np.kron(cz, np.identity(2**(n2))))
    # remove small values
    gate[np.abs(gate) < 1e-15] = 0
    final_rho = np.dot(gate, np.dot(rho, gate.conj().T))

    # apply our error gate and find the new density matrix
    if apply_krauss_errors:
        final_rho = gate_error(final_rho, qubit_error_probs[target], target, tot_qubits)
    if apply_rad_errors:
        final_rho = rad_error(final_rho, t1, t2, tg)

    return final_rho


def prob_rad_non_adj_CZ(
        rho, control, target, t1, t2, tg, qubit_error_probs
    ): # pylint: disable=invalid-name
    """
    Implement a non-adjacent rad CZ gate between 2 qubits in a system with
    Krauss and relaxation and dephasing (RAD) errors.

    * rho: the density matrix representation of your system
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * qubit_error_probs: an array of the probability for errors of each qubit
    in your system
    """
    apply_krauss_errors = check_apply_krauss(qubit_error_probs)
    apply_rad_errors = check_apply_rad(t1, t2, tg)
    tot_qubits = int(np.log(len(rho))/np.log(2))

    # used to index over all gates neeeded to compose final gate
    p = np.abs(target - control)   # pylint: disable=invalid-name

     # Adds the dimensions needed depending on the tot_qubits
    if control < target:
        # exponent used to tensor the left side identity matrix for our full system
        n1 = control                  # pylint: disable=invalid-name
        # exponent used to tensor the right side identity matrix for our full system
        n2 = tot_qubits - target - 1  # pylint: disable=invalid-name
    else:
        # exponent used to tensor the left side identity matrix for our full system
        n1 = target                    # pylint: disable=invalid-name
        # exponent used to tensor the right side identity matrix for our full system
        n2 = tot_qubits - control - 1  # pylint: disable=invalid-name

    # Find the correct Hadamard gate to apply so that you convert the CNOT to a CZ
    h_gate = np.kron(
        np.identity(2**(n1)),
        np.kron(
            np.kron(np.identity(2**(np.abs(target - control))), hadamard),
            np.identity(2**(n2))
        )
    )
    # apply the hadamard first to take it to the (+, -) basis
    # no errors here - just changing basis (asserting our hardware can do native CZ)
    rho = np.dot(h_gate, np.dot(rho, h_gate.conj().T))
    # accumulate errors
    error_rho = rho

    # Applies the gates twice (square in our formula)
    for _ in range(0,2):
        # Indexing over the values of p to get the first half of the formula
        for j in range(p):
            # Sets the next component of the matrix multiplication up
            next_dot = np.kron(np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1))))
            next_dot = np.kron(np.identity(2**(n1)), np.kron(next_dot, np.identity(2**(n2))))
            index = control + j + 1 if (control < target) else control - j - 1
            # Adds the components to the array and multiplies them together
            if j == 0:
                all_dots = np.array([next_dot]) # adds the perfect gate to an array
            else:
                all_dots = np.append(all_dots, [next_dot], axis = 0) # adds perfect gate
            gate = all_dots[j] # sets the current gate
            # applies the perfect gate to our density matrix
            error_rho = np.dot(gate, np.dot(error_rho, gate.conj().T))
            # apply our error gate and find the new density matrix
            if apply_krauss_errors:
                error_rho = gate_error(
                    error_rho, qubit_error_probs[index], index, tot_qubits
                )
            if apply_rad_errors:
                error_rho = rad_error(error_rho, t1, t2, tg)

        # Indexing over values of p such that we get the 2nd half of the equation together
        for j in range(p - 2):
            index = target - j - 1 if (control < target) else target + j + 1
            gate = all_dots[p-j-2] # sets the current gate
            # applies the perfect gate to our density matrix
            error_rho = np.dot(gate, np.dot(error_rho, gate.conj().T))
            # apply our error gate and find the new density matrix
            if apply_krauss_errors:
                error_rho = gate_error(
                    error_rho, qubit_error_probs[index], index, tot_qubits
                )
            if apply_rad_errors:
                error_rho = rad_error(error_rho, t1, t2, tg)

    # Calculate the final rho - return to original basis
    # apply our "effective CZ" error here (returning to basis)
    error_rho = np.dot(h_gate, np.dot(error_rho, h_gate.conj().T))
    if apply_krauss_errors:
        error_rho = gate_error(
            error_rho, qubit_error_probs[index], index, tot_qubits
        )
    if apply_rad_errors:
        error_rho = rad_error(error_rho, t1, t2, tg)

    return error_rho # returns the density matrix of your system


def prob_line_rad_CZ(
        state, control, target, t1, t2, tg, qubit_error_probs,
        form='psi', remove_small_values=True
    ): # pylint: disable=invalid-name,too-many-arguments
    """
    Implement a CNOT gate between 2 qubits depending on your control and target
    qubit with Krauss and relaxation and dephasing (RAD) errors. Returns
    the density matrix after the gate with appropriate errors.

    * state: the vector state representation or density matrix representation of
    your system (default is state vector)
    * control: control qubit index (starting from 0)
    * target: target qubit index (starting from 0)
    * t1: The relaxation time of each physical qubit in your system
    * t2: The dephasing time of each physical qubit in your system
    * tg: The gate time of your gate operations
    * qubit_error_probs: an array of the probability for errors of each qubit in
    your system
    * form: either 'psi' for vector representation or 'rho' for density matrix
    that user inputs
    * remove_small_values: trim entries below 1e-15 from the output density
    matrix
    """
    # TODO: make the error random between targtet and control? Or keep it always target?
    assert target != control, "target cannot equal control in `prob_line_rad_CZ()`"
    # if the form is 'psi' find the density matrix
    if form == 'psi':
        rho = np.kron(state, state[np.newaxis].conj().T)
    else:
        rho = state

    # Check if adjacent
    if np.abs(target - control) == 1:
        final_rho = prob_rad_adj_CZ(rho, control, target, t1, t2, tg, qubit_error_probs)
    else:
        final_rho = prob_rad_non_adj_CZ(rho, control, target, t1, t2, tg, qubit_error_probs)

    if remove_small_values:
        final_rho[np.abs(final_rho) < 1e-15] = 0

    return final_rho # output is always the density matrix after the operation


### - - - - - - - - - SPAM error function - - - - - - - - - ###

def spam_error(rho, error_prob, index):
    """
    Takes the density matrix after state preparation and before measurement
    and applies an error gate with probability.

    * rho: density matrix of qubit system after perfect gate was applied
    * error_prob: probability for state preparation error or measuremnt error
    * index: index of qubit that was prepared/measured
    """
    # total number of qubits in your system
    tot_qubits = int(np.log(len(rho))/np.log(2))

    # qubit error rates:
    KM0 = np.sqrt(1-error_prob) * sigma_I   # pylint: disable=invalid-name
    KM1 = np.sqrt(error_prob) * sigma_x     # pylint: disable=invalid-name

    # qubit error gates
    KM0 = np.kron(np.identity(2**(index)), np.kron(KM0, np.identity(2**(tot_qubits-index-1)))) # pylint: disable=invalid-name
    KM1 = np.kron(np.identity(2**(index)), np.kron(KM1, np.identity(2**(tot_qubits-index-1)))) # pylint: disable=invalid-name

    # apply error gates
    spam_rho = np.dot(KM0, np.dot(rho, KM0.conj().T)) + np.dot(KM1, np.dot(rho, KM1.conj().T))

    return spam_rho
