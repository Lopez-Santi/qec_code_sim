# QEC_code
This repository introduces quantum error correction (QEC) and applies error models to benchmark various error correction algorithms against different qubit parameters. 

There are two ways in which you can use the files in this repository:

1. If you would like to test the parameters of your qubit system implemented in a certain QEC algorithm you can use our **QEC Simulator** to do this. After you input the parameters of your qubits you will be able to see different information about your system including the logical T1, the distribution of times at which your circuit will fail, and the distribution of the estimated logical T1 of your system.

	Some things to note about your system before jumping into this section.

	(a) You will need to know some of the error parameters that you want to simulate for your system. Not all need to be used, in fact, none do, but then you would just simulate a perfect design which isn't very useful.
  	- The **depolarization error probability** (each gate will implement the same probability of depolarization for all qubits). If you would like to specify each qubit to have a different depolarization probability you can, but you will need to set up your own simulation.
  	- The **State Preparation and Measurement error probability** of your system. Each qubit will be given the same value. If you would like to specify each qubit to have a different SPAM error probability you can, but you will need to set up your own simulation.
  	- The **T1, T2, and gate time (Tg)** of the qubits in your system. Each qubit and gate will have the same time parameters. If you would like to specify each qubit and/or gate to have different time parameters you can, but you will need to set up your own simulation.

	(b) How many qubits do you want to simulate? We currently have a few codes which can be implemented in the simulation. Here is a little info about each
  	- **3-qubit code** - The most basic QEC algorithm, and it uses 5 qubits (3 data qubits and 2 ancilla qubits). The simulation is the fastest of the other codes, but it can only correct for a single error. The simulation will take about 40 sec per 1000 iterations.
  	- **7-qubit Steane code** - Implements stabilizer formalism and uses 10 qubits (7 data qubits and 3 ancilla qubits). This code can correct for a single bit and/or phase flip error in the data qubits. The simulation will take about 20 min per iteration.
  	- **7-qubit Fault Tolerant Steane code** - Implements the stabilizer formalism and fault tolerance, and uses 12 qubits ((7 data qubits and 5 ancilla qubits). This code can correct for a single bit and/or phase flip error in the data qubits and can correct for certain bit and phase flip errors in the ancilla qubits. The simulation has not been fully tested yet
  	- **9-qubit Shor code** - (cousin of the 3-qubit code) Implements the degenerate 9-qubit code and uses 11 qubits (9 data qubits and 2 ancilla qubits). This code can correct for a single bit and/or phase flip error in the data qubits and in some cases multiple bit-flip errors. The simulation has not been tested yet.

    
* *Note that these simulations were done on a 2020 MacBook Pro (M1) with 8GB RAM and macOS 13.4.1.*


2. If you would like an introduction for QEC then we have created many notebooks inside the folder **Implementation Knowledge Base**. In these files, we go through the basics of QEC and introduce the math and physics behind quantum circuits as well as the errors that occur.

	The topics which we currently cover are:
  
	- **01. Introduction to Quantum Error** - Here we introduce quantum errors and some of the basic math that we will use in further notebooks.
 	- **01a. Error Models** - Here we derive and explain the various error models that we use in our realistic implementations of our circuits. 
 	- **02. 3-qubit Code Tutorial** - Here we introduce and work through the most basic QEC algorithm, the 3-qubit code.
	- **02a. Short 3-qubit QEC** - If you want to skip the details of the 3-qubit code, you can just implement it here quickly and see how it works.
	- **02b. 3 qubit Restricting Connectivity** - Realistically, qubits are not all connected to each other, meaning we may need to ‘break’ up single gates into various gate operations. Here we introduce a connectivity restriction to the 3-qubit code.
	- **02c. 3-qubit logical T1 calculation** - Here we implement our error models to the 3-qubit code and demonstrate how we calculate the logical T1 and circuit failure distribution of our system.
 	- **03. Stabilizer Codes and Steane Code** - Stabilizer operators are a different and useful way to think about QEC. Here we introduce these ideas and implement them using the 7-qubit Steane code.
  	- **03a. Steane Code Restricting Connectivity** - Realistically, qubits are not all connected to each other, meaning we may need to ‘break’ up single gates into various gate operations. Here we introduce a connectivity restriction to the 7-qubit Steane code.
  	- **03b. Steane code logical T1 calculation** - Here we implement our error models to the 7-qubit Steane code and demonstrate how we calculate the logical T1 and circuit failure distribution of our system.
  	- **04. Fault Tolerance** - Errors do not only occur on our data qubits, but syndrome ancilla qubits will also have errors. In some cases, these errors can be much more detrimental to our system since they spread throughout the circuit. Here we introduce the idea of fault tolerance.
  	- **04a. Fault Tolerant Steane Code** - Here we implement the ideas of fault tolerance to the 7-qubit Steane code.
  	- **05. 9-qubit Code Tutorial** - Here we implement the 9-qubit code and explain some key differences from the 3-qubit code, namely that we can now correct for phase errors too.

- - - - - - - - - - - - - - - - - - - - - -

**Please look below for the requirements to successfully run our programs:**

- numpy: main mathematical resource
- matplotlib: used for plotting
- scipy: used for line fitting
- qiskit: for drawing some of the circuits in **Implementation Knowledge Base**
- qiskit-aer: for drawing some of the circuits in **Implementation Knowledge Base**
- pylatexenc: for drawing some of the circuits in **Implementation Knowledge Base**
- prettytable: used to output some tables in **Implementation Knowledge Base**
- h5py: used for saving and managing data files
- tabulate: used to output some tables for data management
- coverage: used only for checking unit test coverage, not required for general use

*The data management section was taken from Ma Quantum Lab at Purdue University, which adopted it from https://github.com/SchusterLab/slab/tree/master/slab*

