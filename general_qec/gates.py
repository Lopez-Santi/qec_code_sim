"""
This file contains functions that implement different types of useful gates to
the circuit.
"""
import numpy as np

### Pauli operators
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,1j],[-1j,0]])
sigma_z = np.array([[1,0],[0,-1]])
sigma_I = np.identity(2)

### Hadamard Gate
hadamard = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]])

### - - - - - - - - - - - CNOT GATES - - - - - - - - - - - ###

### CNOT gate
cnot = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])

# flips the roles of control and target in our usual CNOT gate
flipped_cnot = np.array([[1, 0, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0],[0, 1, 0, 0]])


def rx_theta(theta):
    """
    RX(theta) for a single qubit with theta in radians
    """
    return np.array([[np.cos(theta), -1j*np.sin(theta)],
                     [-1j*np.sin(theta), np.cos(theta)]])


def adj_CNOT(control, target, tot_qubits): # pylint: disable=invalid-name
    """
    Implement a CNOT gate between 2 adjacent qubits in a system

    control: control qubit index (starting from 0)
    target: target qubit index (starting from 0) (must > control)
    tot_qubits: total number of qubits in the system

    target - control != 1
    """
    assert target - control == 1, "target - control != 1"
    # exponent used to tensor the left side identity matrix for our full system
    n1 = control                    # pylint: disable=invalid-name
    # exponent used to tensor the right side ident matrix for our full system
    n2 = tot_qubits - target - 1    # pylint: disable=invalid-name
    final_gate = np.kron(
        np.identity(2**(n1)), np.kron(cnot, np.identity(2**(n2)))
    )

    return final_gate


def flipped_adj_CNOT(control, target, tot_qubits): # pylint: disable=invalid-name
    """
    Implement a flipped CNOT gate between 2 adjacent qubits in a system

    control: control qubit index (starting from 0)
    target: target qubit index (starting from 0) (must < control)
    tot_qubits: total number of qubits in the system

    control - target != 1
    """
    assert control - target == 1, "control - target != 1"
    # exponent used to tensor the left side identity matrix for our full system
    n1 = target                      # pylint: disable=invalid-name
    # exponent used to tensor the right side ident matrix for our full system
    n2 = tot_qubits - control - 1    # pylint: disable=invalid-name
    final_gate = np.kron(
        np.identity(2**(n1)), np.kron(flipped_cnot, np.identity(2**(n2)))
    )

    return final_gate


def small_non_adj_CNOT(): # pylint: disable=invalid-name
    """
    Used to quickly perform a CNOT gate on 2 non-adjacent qubits (i.e. |psi>
    and |q_1>) --- for 3 qubits
    """
    small_non_adj_cnot = np.identity(8)
    small_non_adj_cnot[4:] = 0
    small_non_adj_cnot[4][5] = 1
    small_non_adj_cnot[5][4] = 1
    small_non_adj_cnot[7][6] = 1
    small_non_adj_cnot[6][7] = 1
    return small_non_adj_cnot


def non_adj_CNOT(control, target, tot_qubits): # pylint: disable=invalid-name
    """
    Implement a non-adjacent CNOT gate between 2 qubits in a system

    control: control qubit index (starting from 0)
    target: target qubit index (starting from 0) (must be > control)
    tot_qubits: total number of qubits in the system

    (target - control) must be greater than 1
    """
    assert target - control > 1, "(target - control) must be greater than 1"
    # used to index over all gates neeeded to compose final gate
    p = target - control   # pylint: disable=invalid-name
    # array used to keep track of the components we will combine at the end
    all_dots = np.array([[]])

    # Indexing over the values of p to get the first half of the formula
    for j in range(p):
        # Sets the next component of the matrix multiplication up
        next_dot = np.kron(
            np.identity(2**(j)), np.kron(cnot, np.identity(2**(p-j-1)))
        )
        # Adds the components to the array and multiplies them together
        if j == 0:
            all_dots = np.array([next_dot])
            gate = all_dots[j]
        else:
            all_dots = np.append(all_dots, [next_dot], axis = 0)
            gate = np.dot(gate, all_dots[j])

    # Indexing over values of p such that we get the 2nd half of the equation
    # together
    for j in range(p - 2):
        gate = np.dot(gate, all_dots[p-j-2])

    # Squares the final matrix
    final_gate = np.dot(gate, gate)

    # exponent used to tensor the left side identity matrix for our full system
    n1 = control                    # pylint: disable=invalid-name
    # exponent used to tensor the right side ident matrix for our full system
    n2 = tot_qubits - target - 1    # pylint: disable=invalid-name
    final_total_gate = np.kron(
        np.identity(2**(n1)), np.kron(final_gate, np.identity(2**(n2)))
    )
    return final_total_gate


def flipped_non_adj_CNOT(control, target, tot_qubits): # pylint: disable=invalid-name
    """
    Implement a flipped non-adjacent CNOT gate between 2 qubits in a system

    control: control qubit index (starting from 0)
    target: target qubit index (starting from 0) (must be < control)
    tot_qubits: total number of qubits in the system

    (control - target) must be greater than 1
    """
    assert control - target > 1, "(control - target) must be greater than 1"
    # used to index over all gates neeeded to compose final gate
    p = np.abs(target - control)  # pylint: disable=invalid-name
    # array used to keep track of the components we will combine at the end
    all_dots = np.array([[]])

    # Indexing over the values of p to get the first half of the formula
    for j in range(p):
        # Sets the next component of the matrix multiplication up
        next_dot = np.kron(
            np.identity(2**(p-j-1)), np.kron(flipped_cnot, np.identity(2**(j)))
        )

        # Adds the components to the array and multiplies them together
        if j == 0:
            all_dots = np.array([next_dot])
            gate = all_dots[j]
        else:
            all_dots = np.append(all_dots, [next_dot], axis = 0)
            gate = np.dot(gate, all_dots[j])

    # Indexing over values of p such that we get the 2nd half of the equation
    # together
    for j in range(p - 2):
        gate = np.dot(gate, all_dots[p-j-2])

    # Squares the final matrix
    final_gate = np.dot(gate, gate)

    # exponent used to tensor the left side identity matrix for our full system
    n1 = target                     # pylint: disable=invalid-name
    # exponent used to tensor the right side ident matrix for our full system
    n2 = tot_qubits - control - 1   # pylint: disable=invalid-name
    final_total_gate = np.kron(
        np.identity(2**(n1)), np.kron(final_gate, np.identity(2**(n2)))
    )

    return final_total_gate


def CNOT(control, target, tot_qubits): # pylint: disable=invalid-name
    """
    Implement a CNOT gate between 2 qubits depending on your control and
    target qubit.
    """
    # First check if it is a normal CNOT or a flipped CNOT gate
    if control < target:
        # Check if adjacent
        if target - control == 1:
            gate = adj_CNOT(control, target, tot_qubits)
        else:
            gate = non_adj_CNOT(control, target, tot_qubits)

    #Check if it is a normal CNOT or a flipped CNOT gate
    elif control > target:
        # Check if adjacent
        if control - target == 1:
            gate = flipped_adj_CNOT(control, target, tot_qubits)
        else:
            gate = flipped_non_adj_CNOT(control, target, tot_qubits)

    return gate


### - - - - - - - - - - - C-Z GATES - - - - - - - - - - - ###

# Note that we will not need to impliment a flipped CZ gate becuase the logic
# table is the same for both.

### Control-Z gate
# cz = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, -1]])
## OR could also use:
cz = np.dot(
    np.kron(np.identity(2), hadamard),
    np.dot(cnot, np.kron(np.identity(2), hadamard))).round().astype(int)


def CZ(control, target, tot_qubits): # pylint: disable=invalid-name
    """
    Implement a Control-Z gate between 2 qubits depending on your parameters
    """
    # first, Hadamard the target
    target_hadamard = np.kron(
        np.kron(np.identity(2**target), hadamard),
        np.identity(2**(tot_qubits - target - 1))
    )
    gate = target_hadamard
    # second, apply a CNOT
    gate = np.dot(gate, CNOT(control, target, tot_qubits))
    # third, Hadamard the target
    gate = np.dot(gate, target_hadamard)
    # remove small values
    gate[np.abs(gate) < 1e-15] = 0

    # return the gate
    return gate
