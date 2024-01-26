# Importing required libraries
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit import user_config
from qiskit.quantum_info import partial_trace
import numpy as np
import random
from qiskit.extensions import UnitaryGate

# For creating tables:
from tabulate import tabulate
from prettytable import PrettyTable

# Initialize backends simulators to visualize circuits
sv_sim = Aer.get_backend('statevector_simulator')
qasm_sim = Aer.get_backend('qasm_simulator')

# Setting mpl as default drawer
# env QISKIT_SETTINGS {}
# user_config.set_config('circuit_drawer', 'mpl')

### Draw the full 3 qubit error correcting circuit with 2 ancilla qubits ###
def draw_three_qubit_code():
    
    # Draw the quantum circuit
    psi = QuantumRegister(1, '|ψ⟩')
    ancilla = QuantumRegister(2, '|0⟩')
    syndrome_ancilla = QuantumRegister(2, 'syndrome |0⟩')
    classical_bits = ClassicalRegister(2, 'classical_measurement')
    qc = QuantumCircuit(psi, ancilla, syndrome_ancilla, classical_bits)
    qc.cnot(0, 1)
    qc.cnot(1, 2)
    qc.barrier(0,1,2) # A bit flip error occurs here
    qc.cnot(0, 3)
    qc.cnot(1, 3)
    qc.cnot(0, 4)
    qc.cnot(2, 4)
    qc.measure(syndrome_ancilla, classical_bits)
    qc.barrier()
    
    # Draw the table to show the possible error outcomes with ancilla measurements
    error_table = PrettyTable(["Error Location", "Final State, |data⟩|ancilla⟩"])
    error_table.add_row(["No Error", "alpha|000⟩|00⟩ + beta|111⟩|00⟩"])
    error_table.add_row(["Qubit 0 (|ψ⟩)", "alpha |100⟩|11⟩ + beta|011⟩|11⟩"])
    error_table.add_row(["Qubit 1 (|0⟩_0)", "alpha |010⟩|10⟩ + beta|101⟩|10⟩"])
    error_table.add_row(["Qubit 2 (|0⟩_1)", "alpha |001⟩|01⟩ + beta|110⟩|01⟩"])

    qc.draw()
    print(error_table.get_string())

    return qc
