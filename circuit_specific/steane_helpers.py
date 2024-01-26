"""
The functions in this file are useful when implementing the seven qubit steane code.
"""
import numpy as np
from general_qec.qec_helpers import one, zero
from general_qec.qec_helpers import collapse_ancilla
from general_qec.qec_helpers import remove_small_values
from general_qec.qec_helpers import vector_state_to_bit_state
from general_qec.gates import hadamard
from general_qec.gates import sigma_x, sigma_z, sigma_I, CNOT, CZ


# - - - - - - - - - -  Useful variables - - - - - - - - - - #

# Setting up the 6 Stabilizer Operators for the 7-qubit Steane Code
k_one = np.kron(sigma_I, np.kron(sigma_I, np.kron(sigma_I, np.kron(
    sigma_x, np.kron(sigma_x, np.kron(sigma_x, sigma_x))))))
k_two = np.kron(sigma_x, np.kron(sigma_I, np.kron(sigma_x, np.kron(
    sigma_I, np.kron(sigma_x, np.kron(sigma_I, sigma_x))))))
k_three = np.kron(sigma_I, np.kron(sigma_x, np.kron(sigma_x, np.kron(
    sigma_I, np.kron(sigma_I, np.kron(sigma_x, sigma_x))))))
k_four = np.kron(sigma_I, np.kron(sigma_I, np.kron(sigma_I, np.kron(
    sigma_z, np.kron(sigma_z, np.kron(sigma_z, sigma_z))))))
k_five = np.kron(sigma_z, np.kron(sigma_I, np.kron(sigma_z, np.kron(
    sigma_I, np.kron(sigma_z, np.kron(sigma_I, sigma_z))))))
k_six = np.kron(sigma_I, np.kron(sigma_z, np.kron(sigma_z, np.kron(
    sigma_I, np.kron(sigma_I, np.kron(sigma_z, sigma_z))))))

### Gate operations for steane code using 3 ancillas ###

# phase correction gates
control_k_one = \
    np.kron(
        np.identity(2**7),
        np.kron(
            np.kron(zero, zero[np.newaxis].T),
            np.identity(2**2)
        )
    ) + \
    np.kron(
        k_one,
        np.kron(np.kron(one, one[np.newaxis].T), np.identity(2**2))
    )

control_k_two = \
    np.kron(
        np.identity(2**7),
        np.kron(np.identity(2), np.kron(np.kron(zero, zero[np.newaxis].T), np.identity(2)))
    ) + \
    np.kron(
        k_two,
        np.kron(np.identity(2), np.kron(np.kron(one, one[np.newaxis].T), np.identity(2)))
    )

control_k_three = \
    np.kron(
        np.identity(2**7),
        np.kron(np.identity(2**2), np.kron(zero, zero[np.newaxis].T))
    ) + \
    np.kron(
        k_three,
        np.kron(np.identity(2**2), np.kron(one, one[np.newaxis].T))
    )

# bit correction gates
control_k_four = \
    np.kron(
        np.identity(2**7),
        np.kron(np.kron(zero, zero[np.newaxis].T), np.identity(2**2))
    ) + \
    np.kron(
        k_four,
        np.kron(np.kron(one, one[np.newaxis].T), np.identity(2**2))
    )

control_k_five = \
    np.kron(
        np.identity(2**7),
        np.kron(np.identity(2), np.kron(np.kron(zero, zero[np.newaxis].T), np.identity(2)))
    ) + \
    np.kron(
        k_five,
        np.kron(np.identity(2), np.kron(np.kron(one, one[np.newaxis].T), np.identity(2)))
    )

control_k_six = \
    np.kron(
        np.identity(2**7),
        np.kron(np.identity(2**2), np.kron(zero, zero[np.newaxis].T))
    ) + \
    np.kron(
        k_six,
        np.kron(np.identity(2**2), np.kron(one, one[np.newaxis].T))
    )


# - - - - - - - - - -  Initializations - - - - - - - - - - #

def seven_tensor_kron(q0, q1, q2, q3, q4, q5, q6):
    """
    Take the kronecker product of 7 tensors
    """
    return \
        np.kron(q0,
        np.kron(q1,
        np.kron(q2,
        np.kron(q3,
        np.kron(q4,
        np.kron(q5, q6)
    )))))


def steane_dataq_logical_zero():
    """
    Return the Steane 7-qubit logical zero state
    """
    a = seven_tensor_kron(zero, zero, zero, zero, zero, zero, zero)
    b = seven_tensor_kron(one,  zero, one,  zero, one,  zero, one)
    c = seven_tensor_kron(zero, one,  one,  zero, zero, one,  one)
    d = seven_tensor_kron(one,  one,  zero, zero, one,  one,  zero)
    e = seven_tensor_kron(zero, zero, zero, one,  one,  one,  one)
    f = seven_tensor_kron(one,  zero, one,  one,  zero, one,  zero)
    g = seven_tensor_kron(zero, one,  one,  one,  one,  zero, zero)
    h = seven_tensor_kron(one,  one,  zero, one,  zero, zero, one)
    return (1. / np.sqrt(8.0)) * (a + b + c + d + e + f + g + h)


def steane_dataq_logical_one():
    """
    Return the Steane 7-qubit logical one state
    """
    a = seven_tensor_kron(one,  one,  one,  one,  one,  one,  one)
    b = seven_tensor_kron(zero, one,  zero, one,  zero, one,  zero)
    c = seven_tensor_kron(one,  zero, zero, one,  one,  zero, zero)
    d = seven_tensor_kron(zero, zero, one,  one,  zero, zero, one)
    e = seven_tensor_kron(one,  one,  one,  zero, zero, zero, zero)
    f = seven_tensor_kron(zero, one,  zero, zero, one,  zero, one)
    g = seven_tensor_kron(one,  zero, zero, zero, zero, one,  one)
    h = seven_tensor_kron(zero, zero, one,  zero, one,  one,  zero)
    return (1. / np.sqrt(8.0)) * (a + b + c + d + e + f + g + h)

def steane_dataq_logical_superpos():
    """
    Return the Steane 7-qubit logical one state
    """
    return (1. / np.sqrt(2.)) * (steane_dataq_logical_zero() + steane_dataq_logical_one())


def initialize_steane_logical_state(initial_state): # pylint: disable=too-many-locals
    """
    Initializes the 10 qubit (7 physical, 3 ancilla) qubit system.

    * initial_state: initial state of your 7 qubits qubit that you want to use
    as your logical state combined with ancillas
    """
    ancilla_syndrome = np.kron(zero, np.kron(zero, zero))
    full_system = np.kron(initial_state, ancilla_syndrome)

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(control_k_one, np.dot(control_k_two, np.dot(control_k_three, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # How many total qubits are in our vector representation
    nqubits = int(np.log(len(full_system))/np.log(2)) # pylint: disable=invalid-name
    nancilla = 3
    assert nqubits == 10, "Invalid initial state for the Steane code."

    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(full_system, nancilla)

    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
    bits = vector_state_to_bit_state(collapsed_state, nqubits)[0][0]

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
        final_vector_state = collapsed_state
    else:
        # apply the z gate depending on index
        operation = np.kron(
            np.identity(2**(index)),
            np.kron(sigma_z,
                    np.kron(np.identity(2**(nqubits-nancilla-index-1)), np.identity(2**nancilla))
            )
        )
        final_vector_state = np.dot(operation, collapsed_state)

    # Using this for superposition states, doesnt do anything for |0> initial states
    # becuase they are already +1 eigenstates of Z
    comparison_state = np.kron(
        zero, np.kron(zero, np.kron(zero, np.kron(zero, np.kron(zero, np.kron(zero, zero)))))
    )
    # TODO: what is this comparison really meant for?
    # ...We require every single element of the vectors to be different here? -> Why?
    if (initial_state != comparison_state).all():
        final_vector_state = steane_bit_correction(final_vector_state)

    # apply global phase correction:
    z_bar = np.kron(sigma_z, np.kron(sigma_z, np.kron(sigma_z, np.kron(sigma_z, np.kron(
        sigma_z, np.kron(sigma_z, sigma_z))))))
    z_bar = np.kron(z_bar, np.identity(2**nancilla))
    final_vector_state = np.dot(z_bar, final_vector_state)

    return final_vector_state


# - - - - - - - - - - 3 ancilla error correction protocols - - - - - - - - - - #

def steane_phase_correction(logical_state):
    """
    Corrects for a single phase flip error in the 7 qubit steane code with 3 ancillas
    * logical_state: The vector state representation of your full system
    """
    full_system = logical_state

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(control_k_one, np.dot(control_k_two, np.dot(control_k_three, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)

    # remove small values after applying operations
    vector_state = remove_small_values(vector_state)

    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)

    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))

    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
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
        final_vector_state = collapsed_state
    else:
        # apply the z gate depending on index
        operation = np.kron(
            np.identity(2**(index)),
            np.kron(sigma_z,
                    np.kron(np.identity(2**(n-3-index-1)), np.identity(2**3)))
        )
        final_vector_state = np.dot(operation, collapsed_state)

    corrected_vector_state = remove_small_values(final_vector_state)

    return corrected_vector_state


def steane_bit_correction(logical_state):
    """
    Corrects for a single bit flip error in the 7 qubit steane code with 3 ancillas
    * logical_state: The vector state representation of your full system
    """
    full_system = logical_state

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = np.dot(control_k_four, np.dot(control_k_five, np.dot(control_k_six, full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)

    # remove small values after applying operations
    vector_state = remove_small_values(vector_state)

   # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)

    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))

    # Measure the three ancilla qubits
    # Applying the X gate operation on a specific qubit
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

    # Which qubit do we perform the X gate on
    index = (m_four * 2**2) + (m_six * 2**1) + (m_five * 2**0) - 1

    # if no error occurs we dont need to apply a correction
    if index == -1:
        final_vector_state = collapsed_state
    else:
        # apply the z gate depending on index
        operation = np.kron(
            np.identity(2**(index)),
            np.kron(sigma_x,
                    np.kron(np.identity(2**(n-3-index-1)), np.identity(2**3)))
        )
        final_vector_state = np.dot(operation, collapsed_state)

    # remove small values from state
    corrected_vector_state = remove_small_values(final_vector_state)

    return corrected_vector_state


# - - - - - - - - - - Larger 6 ancilla Steane code implementation - - - - - - - - - - #

### Gate operations for larger steane code using 6 ancillas ###
# (first 3 are controlled by  first 3 ancilla, other 3 are controlled by the other 3 ancilla)

# phase correction gates
larger_control_k_one = \
    np.kron(
        np.identity(2**7),
        np.kron(np.kron(zero, zero[np.newaxis].T), np.identity(2**5))
    ) + \
    np.kron(
        k_one,
        np.kron(np.kron(one, one[np.newaxis].T), np.identity(2**5))
    )

larger_control_k_two = \
    np.kron(
        np.identity(2**7),
        np.kron(np.identity(2), np.kron(np.kron(zero, zero[np.newaxis].T), np.identity(2**4)))
    ) + \
    np.kron(
        k_two,
        np.kron(np.identity(2), np.kron(np.kron(one, one[np.newaxis].T), np.identity(2**4)))
    )

larger_control_k_three = \
    np.kron(
        np.identity(2**7),
        np.kron(np.identity(2**2), np.kron(np.kron(zero, zero[np.newaxis].T), np.identity(2**3)))
    ) + \
    np.kron(
        k_three,
        np.kron(np.identity(2**2), np.kron(np.kron(one, one[np.newaxis].T), np.identity(2**3)))
    )

# bit correction gates
larger_control_k_four = \
    np.kron(
        np.identity(2**7),
        np.kron(np.identity(2**3), np.kron(np.kron(zero, zero[np.newaxis].T), np.identity(2**2)))
    ) + \
    np.kron(
        k_four,
        np.kron(np.identity(2**3), np.kron(np.kron(one, one[np.newaxis].T), np.identity(2**2)))
    )

larger_control_k_five = \
    np.kron(
        np.identity(2**7),
        np.kron(np.identity(2**4), np.kron(np.kron(zero, zero[np.newaxis].T), np.identity(2)))
    ) + \
    np.kron(
        k_five,
        np.kron(np.identity(2**4), np.kron(np.kron(one, one[np.newaxis].T), np.identity(2)))
    )

larger_control_k_six = \
    np.kron(
        np.identity(2**7),
        np.kron(np.identity(2**5), np.kron(zero, zero[np.newaxis].T))
    ) + \
    np.kron(
        k_six,
        np.kron(np.identity(2**5), np.kron(one, one[np.newaxis].T))
    )


def initialize_larger_steane_code(initial_state): # pylint: disable=too-many-locals
    """
    Initializes the 13 qubit (7 physical, 6 ancilla) qubit system.

    * initial_state: initial state of 7 data qubits
    """
    # TODO - this is not DRY - should change this to take a 1 qubit "initial state", prep over 7, then call the full code
    n_total = 13   # Total number of qubits in our system
    n_ancilla = 6  # Total number of ancilla qubits

    # prep the full 13 qubit system
    ancilla_triple = np.kron(zero, np.kron(zero, zero))
    hadamard_triple = np.kron(hadamard, np.kron(hadamard, hadamard))
    full_system = np.kron(initial_state, np.kron(ancilla_triple, ancilla_triple))
    assert n_total == int(np.log(len(full_system))/np.log(2)), "Requires a 13 qubit total system"

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(
        np.identity(2**7),
        np.kron(hadamard_triple, hadamard_triple)
    )
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = \
        np.dot(larger_control_k_one,
        np.dot(larger_control_k_two,
        np.dot(larger_control_k_three,
        np.dot(larger_control_k_four,
        np.dot(larger_control_k_five,
        np.dot(larger_control_k_six, full_system)
    )))))

    # apply the second set of hadamards to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, _, vector_state = vector_state_to_bit_state(full_system, n_total)
    vector_state = remove_small_values(vector_state)

    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, n_ancilla)

    # decode the measurements for phase and bit flip locations
    bits = vector_state_to_bit_state(collapsed_state, n_total)[0][0]
    m_one = 0
    m_two = 0
    m_three = 0
    m_four = 0
    m_five = 0
    m_six = 0
    if bits[7] == '1':
        m_one = 1
    if bits[8] == '1':
        m_two = 1
    if bits[9] == '1':
        m_three = 1
    if bits[10] == '1':
        m_four = 1
    if bits[11] == '1':
        m_five = 1
    if bits[12] == '1':
        m_six = 1

    # Which qubit do we perform the Z gate on
    phase_index = (m_one * 2**2) + (m_three * 2**1) + (m_two * 2**0) -1

    # Which qubit do we perform the X gate on
    bit_index = (m_four * 2**2) + (m_six * 2**1) + (m_five * 2**0) -1

    if phase_index != -1:
        # apply the z gate depending on index
        operation = np.kron(
            np.identity(2**(phase_index)),
            np.kron(sigma_z, np.identity(2**(n_total-phase_index-1)))
        )
        collapsed_state = np.dot(operation, collapsed_state)

    if bit_index != -1:
        # apply the x gate depending on index
        operation = np.kron(
            np.identity(2**(bit_index)),
            np.kron(sigma_x, np.identity(2**(n_total-bit_index-1)))
        )
        collapsed_state = np.dot(operation, collapsed_state)

    final_vector_state = collapsed_state

    # Using this for superposition states, doesnt do anything for |0> initial states
    # becuase they are already +1 eigenstates of Z
    # GP -> don't understand why this is needed...
    # TODO: check this...
    # if (initial_state != np.kron(zero, np.kron(zero, np.kron(zero, np.kron(
    #     zero, np.kron(zero, np.kron(zero, zero))))))).all():
    #         final_vector_state = steane_bit_correction(final_vector_state)

    # apply global phase correction:
    seven_sigma_zs = seven_tensor_kron(*[sigma_z]*7)
    z_bar = np.kron(seven_sigma_zs, np.identity(2**n_ancilla))
    final_vector_state = np.dot(z_bar, final_vector_state)

    return final_vector_state


def simultaneous_steane_code(logical_state): # pylint: disable=too-many-locals
    """
    Applies the simultaneous (13q) initialization/error correction code

    * logical_state: the full logical state of the 13 qubit system.
    """
    # TODO - this really looks like the same code as the initialization... do we need it twice? what is the diff?
    full_system = logical_state
    n_ancilla = 6
    n_total = int(np.log(len(full_system))/np.log(2)) # Total number of qubits in our system
    assert n_total == 13, "Simultaneous code is for 13 qubits."

    # apply the first hadamard to the ancillas
    hadamard_triple = np.kron(hadamard, np.kron(hadamard, hadamard))
    ancilla_hadamard = np.kron(
        np.identity(2**7),
        np.kron(hadamard_triple, hadamard_triple)
    )
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = \
        np.dot(larger_control_k_one,
        np.dot(larger_control_k_two,
        np.dot(larger_control_k_three,
        np.dot(larger_control_k_four,
        np.dot(larger_control_k_five,
        np.dot(larger_control_k_six, full_system)
    )))))

    # apply the second set of hadamards to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, _, vector_state = vector_state_to_bit_state(full_system, n_total)
    vector_state = remove_small_values(vector_state)

    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, n_ancilla)

    # decode the measurements for phase and bit flip locations
    bits = vector_state_to_bit_state(collapsed_state, n_total)[0][0]
    m_one = 0
    m_two = 0
    m_three = 0
    m_four = 0
    m_five = 0
    m_six = 0
    if bits[7] == '1':
        m_one = 1
    if bits[8] == '1':
        m_two = 1
    if bits[9] == '1':
        m_three = 1
    if bits[10] == '1':
        m_four = 1
    if bits[11] == '1':
        m_five = 1
    if bits[12] == '1':
        m_six = 1

    # Which qubit do we perform the Z gate on
    phase_index = (m_one * 2**2) + (m_three * 2**1) + (m_two * 2**0) -1

    # Which qubit do we perform the X gate on
    bit_index = (m_four * 2**2) + (m_six * 2**1) + (m_five * 2**0) -1

    if phase_index != -1:
        # apply the z gate depending on index
        operation = np.kron(
            np.identity(2**(phase_index)),
            np.kron(sigma_z, np.identity(2**(n_total-phase_index-1)))
        )
        collapsed_state = np.dot(operation, collapsed_state)

    if bit_index != -1:
        # apply the x gate depending on index
        operation = np.kron(
            np.identity(2**(bit_index)),
            np.kron(sigma_x, np.identity(2**(n_total-bit_index-1)))
        )
        collapsed_state = np.dot(operation, collapsed_state)

    final_vector_state = collapsed_state

    # apply global phase correction:
    seven_sigma_zs = seven_tensor_kron(*[sigma_z]*7)
    z_bar = np.kron(seven_sigma_zs, np.identity(2**n_ancilla))
    final_vector_state = np.dot(z_bar, final_vector_state)

    corrected_vector_state = remove_small_values(final_vector_state)

    return corrected_vector_state


# - - - - - - - - - - Steane Code Using Line Connectivity - - - - - - - - - - #

# Define the Stabilizer Operators as CNOT gates between line adjacent qubits
# (remember that the non-adj CNOT calculation is using line connectivity)
K1_line_operation = \
    np.dot(CNOT(7, 3, 10),
    np.dot(CNOT(7, 4, 10),
    np.dot(CNOT(7, 5, 10),
           CNOT(7, 6, 10))))
K2_line_operation = \
    np.dot(CNOT(8, 0, 10),
    np.dot(CNOT(8, 2, 10),
    np.dot(CNOT(8, 4, 10),
           CNOT(8, 6, 10))))
K3_line_operation = \
    np.dot(CNOT(9, 1, 10),
    np.dot(CNOT(9, 2, 10),
    np.dot(CNOT(9, 5, 10),
           CNOT(9, 6, 10))))

K4_line_operation = \
    np.dot(CZ(7, 3, 10),
    np.dot(CZ(7, 4, 10),
    np.dot(CZ(7, 5, 10),
           CZ(7, 6, 10))))
K5_line_operation = \
    np.dot(CZ(8, 0, 10),
    np.dot(CZ(8, 2, 10),
    np.dot(CZ(8, 4, 10),
           CZ(8, 6, 10))))
K6_line_operation = \
    np.dot(CZ(9, 1, 10),
    np.dot(CZ(9, 2, 10),
    np.dot(CZ(9, 5, 10),
           CZ(9, 6, 10))))


def initialize_steane_line_conn(initial_state):
    """
    Initializes the 10 qubit (7 physical, 3 ancilla) qubit system

    * initial_state: initial state of your 7 qubits that you want to use as your
    logical state prior to combination with ancillas
    """
    ancilla_syndrome = np.kron(zero, np.kron(zero, zero))
    full_system = np.kron(initial_state, ancilla_syndrome)

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = \
        np.dot(K1_line_operation,
        np.dot(K2_line_operation,
        np.dot(K3_line_operation,
               full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)

    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)

    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))

    # Measure the three ancilla qubits
    # Applying the Z gate operation on a specific qubit
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
        final_vector_state = collapsed_state
    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))
        final_vector_state = np.dot(operation, collapsed_state)

    # Using this for superposition states, doesnt do anything for |0> initial states
    # becuase they are already +1 eigenstates of Z
    # TODO - what is this for?
    if (initial_state != seven_tensor_kron(*[zero]*7)).all():
        final_vector_state = steane_line_conn_bit_correction(final_vector_state)

    # apply global phase correction:
    seven_sigma_zs = seven_tensor_kron(*[sigma_z]*7)
    z_bar = np.kron(seven_sigma_zs, np.identity(2**3))
    final_vector_state = np.dot(z_bar, final_vector_state)

    return final_vector_state


def steane_line_conn_phase_correction(logical_state):
    """
    Implements the 7 Qubit Steane phase correction code using line connectivity

    # logical_state: The vector state representation of your 10 qubit system
    # (7 data qubits initialized to your desired logical state, 3 ancilla initialized to 0)
    """
    # TODO - seems like the API across the various Steane codes may need some rationalization
    full_system = logical_state

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = \
        np.dot(K1_line_operation,
        np.dot(K2_line_operation,
        np.dot(K3_line_operation,
               full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)

    # remove small values after applying operations
    vector_state = remove_small_values(vector_state)

    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)

    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))

    # Measure the three ancilla qubits & compute error index
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state
    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)

    corrected_vector_state = remove_small_values(final_vector_state)

    return corrected_vector_state



def steane_line_conn_bit_correction(logical_state):
    """
    Implements the 7 Qubit Steane bit correction code using line connectivity
    * logical_state: The vector state representation of your 10 qubit system
    * (7 data qubits initialized to your desired logical state, 3 ancilla initialized to 0)
    """
    full_system = logical_state

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = \
        np.dot(K4_line_operation,
        np.dot(K5_line_operation,
        np.dot(K6_line_operation,
               full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)

    # remove small values after applying operations
    vector_state = remove_small_values(vector_state)

   # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)

    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))

    # Measure the three ancilla qubits
    # Applying the X gate operation on a specific qubit
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
        final_vector_state = collapsed_state
    else:
        # apply the x gate depending on index
        operation = np.kron(
            np.identity(2**(index)),
            np.kron(sigma_x, np.identity(2**(n-index-1)))
        )
        final_vector_state = np.dot(operation, collapsed_state)

    # remove small values from state
    corrected_vector_state = remove_small_values(final_vector_state)

    return corrected_vector_state

# - - - - - - - - - - Steane Code Using Grid Connectivity - - - - - - - - - - #

# Define the Stabilizer Operators as CNOT gates between grid adjacent qubits
# (remember that the non-adj CNOT calculation is using line connectivity)

# SWAPS needed for K1
swap_a0_6 = np.dot(CNOT(7, 6, 10), np.dot(CNOT(6, 7, 10), CNOT(7, 6, 10)))
swap_6_a1 = np.dot(CNOT(8, 6, 10), np.dot(CNOT(6, 8, 10), CNOT(8, 6, 10)))
# Create K1 operator
K1_grid_operation = np.dot(CNOT(7, 3, 10), np.dot(np.dot(
    np.dot(swap_a0_6, swap_6_a1), np.dot(CNOT(8, 4, 10), np.dot(swap_6_a1, swap_a0_6))), np.dot(
    CNOT(7, 5, 10), CNOT(7, 6, 10))))

# SWAPS needed for K2
swap_a1_4 = np.dot(CNOT(8, 4, 10), np.dot(CNOT(4, 8, 10), CNOT(8, 4, 10)))
# Create K2 operator
K2_grid_operation = np.dot(CNOT(8, 0, 10), np.dot(
    np.dot(swap_a1_4, np.dot(CNOT(4, 2, 10), swap_a1_4)), np.dot(
    CNOT(8, 4, 10), CNOT(8, 6, 10))))

# SWAPS needed for K3
swap_a2_4 = np.dot(CNOT(9, 4, 10), np.dot(CNOT(4, 9, 10), CNOT(9, 4, 10)))
# Create K3 operator
K3_grid_operation = np.dot(CNOT(9, 1, 10), np.dot(
    np.dot(swap_a2_4, np.dot(CNOT(4, 2, 10), swap_a2_4)), np.dot(
    CNOT(9, 5, 10), CNOT(9, 6, 10))))

# SWAPS needed for K4 are the same as K1
# Create K1 operator
K4_grid_operation = np.dot(CZ(7, 3, 10), np.dot(np.dot(
    np.dot(swap_a0_6, swap_6_a1), np.dot(CZ(8, 4, 10), np.dot(swap_6_a1, swap_a0_6))), np.dot(
    CZ(7, 5, 10), CZ(7, 6, 10))))

# SWAPS needed for K5 are the same as K2
# Create K2 operator
K5_grid_operation = np.dot(CZ(8, 0, 10), np.dot(
    np.dot(swap_a1_4, np.dot(CZ(4, 2, 10), swap_a1_4)), np.dot(
    CZ(8, 4, 10), CZ(8, 6, 10))))

# SWAPS needed for K6 are the same as K3
# Create K3 operator
K6_grid_operation = np.dot(CZ(9, 1, 10), np.dot(
    np.dot(swap_a2_4, np.dot(CZ(4, 2, 10), swap_a2_4)), np.dot(
    CZ(9, 5, 10), CZ(9, 6, 10))))


def initialize_steane_grid_conn(initial_state):
    """
    Initializes the 10 qubit (7 physical, 3 ancilla) qubit system

    # initial_state: initial state of your 7 qubits that you want to use as your
    logical state priot to combination with ancillas
    """
    ancilla_syndrome = np.kron(zero, np.kron(zero, zero))
    full_system = np.kron(initial_state, ancilla_syndrome)

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = \
        np.dot(K1_grid_operation,
        np.dot(K2_grid_operation,
        np.dot(K3_grid_operation,
               full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)

    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)

    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))

    # Measure the three ancilla qubits
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state
    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))
        final_vector_state = np.dot(operation, collapsed_state)

    # Using this for superposition states, doesnt do anything for |0> initial states
    # becuase they are already +1 eigenstates of Z
    # TODO - what is this for? - also, assume we need grid connected version...
    if (initial_state != seven_tensor_kron(*[zero]*7)).all():
        final_vector_state = steane_grid_conn_bit_correction(final_vector_state)

    # apply global phase correction:
    # TODO - is this needed? not sure we are consistent with it
    seven_sigma_zs = seven_tensor_kron(*[sigma_z]*7)
    z_bar = np.kron(seven_sigma_zs, np.identity(2**3))
    final_vector_state = np.dot(z_bar, final_vector_state)

    return final_vector_state

def steane_grid_conn_phase_correction(logical_state):
    """
    Implements the 7 Qubit Steane phase correction code using grid connectivity
    * logical_state: The vector state representation of your 10 qubit system
    * (7 data qubits initialized to your desired logical state, 3 ancilla initialized to 0)
    """
    full_system = logical_state

    # apply the first hadamard to the ancillas
    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = \
        np.dot(K1_grid_operation,
        np.dot(K2_grid_operation,
        np.dot(K3_grid_operation,
               full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)

     # remove small values after applying operations
    vector_state = remove_small_values(vector_state)

    # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)

    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))

    # Measure the three ancilla qubits
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state
    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_z, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))

        final_vector_state = np.dot(operation, collapsed_state)

    corrected_vector_state = remove_small_values(final_vector_state)

    return corrected_vector_state


def steane_grid_conn_bit_correction(logical_state):
    """
    Implements the 7 Qubit Steane bit correction code using grid connectivity
    * logical_state: The vector state representation of your 10 qubit system
    * (7 data qubits initialized to your desired logical state, 3 ancilla initialized to 0)
    """
    full_system = logical_state

    ancilla_hadamard = np.kron(np.identity(2**7), np.kron(hadamard, np.kron(hadamard, hadamard)))
    full_system = np.dot(ancilla_hadamard, full_system)

    # apply the control stabilizer gates to the full_system
    full_system = \
        np.dot(K4_grid_operation,
        np.dot(K5_grid_operation,
        np.dot(K6_grid_operation,
               full_system)))

    # apply the second hadamard to the ancillas
    full_system = np.dot(ancilla_hadamard, full_system)

    # Find the bit representation of our full system
    bits, index, vector_state = vector_state_to_bit_state(full_system, 10)

    # remove small values after applying operations
    vector_state = remove_small_values(vector_state)

   # Measure and collapse our ancilla qubits
    collapsed_state = collapse_ancilla(vector_state, 3)

    # How many total qubits are in our vector representation
    n = int(np.log(len(full_system))/np.log(2))

    # Measure the three ancilla qubits
    bits = vector_state_to_bit_state(collapsed_state, 10)[0][0]
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
        final_vector_state = collapsed_state
    else:
        # apply the z gate depending on index
        operation = np.kron(np.identity(2**(index)), np.kron(sigma_x, np.kron(
            np.identity(2**(n-3-index-1)), np.identity(2**3))))
        final_vector_state = np.dot(operation, collapsed_state)

    # remove small values from state
    corrected_vector_state = remove_small_values(final_vector_state)

    return corrected_vector_state
