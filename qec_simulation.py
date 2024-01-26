# import plotting and math tools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# For fitting exponentials
def exp_decay(x, a, b):
    return a * np.exp(-b * x)

# for exponential warnings
import warnings
#suppress warnings
warnings.filterwarnings('ignore')

# for running qec and circuits
from general_qec.qec_helpers import *
from general_qec.errors import *
from circuit_specific.realistic_three_qubit import *
from circuit_specific.three_qubit_helpers import *
from circuit_specific.realistic_steane import *
from circuit_specific.steane_helpers import *
from circuit_specific.realistic_ft_steane import *
from circuit_specific.fault_tolerant_steane import *
from circuit_specific.nine_qubit_helpers import *
from circuit_specific.realistic_nine_qubit import *

# for data management and saving
from data_management_and_analysis.datamanagement import *
from data_management_and_analysis.dataanalysis import *
from tabulate import tabulate
import os

# using datetime module
import datetime # used to see how long things take here


### - - - - - - - - - - SIMULATION INTERFACE - - - - - - - - - - ###
### Runs the top level simulation user interface and calls on simulation function
def run_sim():
    print('Hello! Welcome to the quantum error correction simulator.')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('Key Notes before beginning the simulation.')
    print('* Some error codes take some time to iterate depending on the parameters that you input.')
    print(' - a. 3-qubit code is the fastest and can usually iterate 1000 times in roughly 25 sec.')
    print(' - b. 7-qubit Steane code usually takes about 20 min per iteration.')
    print(' - c. 7-qubit Fault tolerant Steane code has not been tested yet.')
    print(' - d. 9 qubit code has not been tested yet.')
    print('(This code was run using a 2020 MacBook Pro (M1) with 8GB RAM and macOS 13.4.1)')
    print('* For more information on the physics and mathematics behind our simulation, please read the readme or you can visit the implementation knowledge folder.')
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    while True:
        print('Please choose the circuit you would like to run: \n(type in the number as displayed in the list)')
        print('\n1. 3-qubit code\n2. 7-qubit Steane code\n3. Fault tolerant 7-qubit Steane code\n4. 9-qubit code')
        while True:
            try:
                circuit = int(input('Selection: '))
                if 1 <= circuit <=4:
                    break
                else:
                    print('Please input a valid circuit value.')
            except ValueError:
                print("Oops!  That was not a valid value.  Try again...")

        print('- - - - - - - - - -')

        print('In our simulation we will initialize the logical state as |1>.')
        state_bool = bool(input('Would you like to input your own initial state? (Leave blank if not)'))
        if state_bool:
            print('...')
            print('We represent our initial state as alpha*|0> + beta*|1> where |alpha|^2 + |beta|^2 = 1')
            while True:
                try:
                    alpha = float(input('\nalpha: '))
                    beta = float(input('\nbeta: '))
                    if round(np.abs(alpha)**2 + np.abs(beta)**2, 3) == 1:
                        break
                    else:
                        print("Oops!  Those were not valid values.  Try again...")
                except ValueError:
                    print("Oops!  Those were not valid values.  Try again...")

            psi = np.array([alpha, beta])

        else:
            psi = np.array([0, 1])

        print('Initial state: ', psi[0],'|0> + ',psi[1],'|1>')
        print('- - - - - - - - - -')

        print('We will now select whick errors we would like to implement.')
        print('Enter any value if you wish to include that error. (If not, leave it blank and press \'enter\')')
        dep_bool = bool(input('\n1. Depolarization. Adds some probability that your gate operations are incorrect.  '))
        spam_bool = bool(input('2. SPAM. Adds some probability that you will incorrectly prepare and measure your state.  '))
        rad_bool = bool(input('3. Relaxation and Dephasing. Adds qubit decay and environmental decoherence to your system.  '))

        print('- - - - - - - - - -')

        print('\nNow please imput the parameters of your errors.')
        print('* For more information on how we define each type of error please visit [05. Error Models].\n')
        if dep_bool:
            print('...')
            while True:
                try:
                    dep = float(input('\nDepolarization. error probability: '))
                    if 0 <= dep <= 1:
                        break
                    else:
                        print("Oops!  Value must be less than 1.  Try again...")
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")
        else:
            dep = None

        if spam_bool:
            print('...')
            while True:
                try:
                    spam_prob = float(input('\nSPAM. probability for state preparation and measurement errors: '))
                    if 0 <= spam_prob <= 1:
                        break
                    else:
                        print("Oops!  Value must be less than 1.  Try again...")
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")
        else:
            spam_prob = None

        if rad_bool:
            print('...')
            print('\nRelaxation and Dephasing. (For times please use the following format: ae-b or decimal representation)')

            while True:
                try:
                    t1 = float(input('T1. relaxation time of your qubits (sec) [suggested O(e-4)]: '))
                    break
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")

            while True:
                try:
                    t2 = float(input('T2. dephasing time of your qubits (sec) [suggested O(e-4)]: '))
                    break
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")

            while True:
                try:
                    tg = float(input('Tg. the gate time of all gate operations in the circuit (sec) [suggested O(e-8)]: '))
                    break
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")
        else:
            t1 = t2 = tg = None

        print('- - - - - - - - - -')

        print('Now we will select how many times you would like to iterate your circuit.')
        print('Remember that the larger circuits may take 5-15 minutes per iteration due to the size.')
        while True:
            try:
                iterations = int(input('\nIterations: '))
                break
            except ValueError:
                print("Oops!  That was not a valid value.  Try again...")
        
        while True:
            print('. . . . . . . . . .')
            selection = bool(
                input('Would you like to go directly to the sampling portion so you can save data? (leave blank if not)\n [Note that for larger codes this may be the best way to do this since they take a long time but you will be able to save the data and only run a single sample if you wish.]\n'))
            if selection:
                break
            
            print('Thank you, we will now output the information of your circuit.')

            print('. . . . . . . . . .')
            
            simulate_qec(circuit, psi, depolarization=dep, spam_prob=spam_prob, t1=t1, t2=t2, tg=tg, iterations=iterations)
            print('\n')
            print('- - - - - - - - - -')
            # Check what they want to do next
            print('What would you like to do next?\n1. Run the same simulation again.\n2. Start over and input different parameters.\n3. Run the simulation many times and create a sampled distribution of data.')
            while True:
                try:
                    selection = int(input('\nSelection: '))
                    if 1 <= selection <=3:
                        break
                    else:
                        print('Please input a valid value.')
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")
            # run simulation again
            if selection !=1:
                break
        # start over and input different parameters
        if selection !=2:
            break
        
    print('- - - - - - - - - -')
    while True: # sampling
        while True:
            print('How many samples would you like? (remember we will iteraate the circuit many times per sample) ')
            try:           
                samples = int(input())
                break
            except ValueError:
                print("Oops!  That was not a valid value.  Try again...")

        # - Run the sampling that we want - #
        if circuit == 1: # three qubit code
            data_file = three_qubit_sample(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
        elif circuit == 2: # Steane code
            data_file = steane_sample(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
        elif circuit == 3: # fault tolerant Steane code
            data_file = ft_steane_sample(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
        elif circuit == 4: # nine qubit code
            data_file = nine_qubit_sample(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
        
        print('- - - - - - - - - - ')
        print('How would you like to analyze your data?')
        print('1. Check distribution of iteration at which circuit logical state failure occurs.')
        print('2. Check distribution of the logical T1 of your circuit.')
        while True:
                try:
                    selection = int(input('\nSelection: '))
                    if 1 <= selection <=2:
                        break
                    else:
                        print('Please input a valid value.')
                except ValueError:
                    print("Oops!  That was not a valid value.  Try again...")

        # selected plotting the circuit failure iteration counts
        if selection == 1:
           
            print('...')
            print('Creating distribution iteration at which circuit logical state failure occurs.')

            # - Plot the information that we want - #
            if circuit == 1: # three qubit code
                three_qubit_plot_failure(data_file = data_file)
            elif circuit == 2: # Steane code
                steane_plot_failure(data_file = data_file)
            elif circuit == 3: # fault tolerant Steane code
                ft_steane_plot_failure(data_file = data_file)
            elif circuit == 4: # nine qubit code
                nine_qubit_plot_failure(data_file = data_file)

            print('- - - - - - - - - -')
        # selected plotting t1 time distributions
        elif selection == 2:
            print('...')

            print('We will now create a distribution of the logical T1 of your circuit.')
            print('Remember that the initial state will be changed to |1>.')
            if (t1==None and t2==None and tg==None):
                print('...')
                print('For this simulation we will need you to select physical T1, T2, and gate time (Tg).')
                print('...')
                print('Relaxation and Dephasing. (For times please use the following format: ae-b or decimal representation)')

                while True:
                    try:
                        t1 = float(input('T1. relaxation time of your qubits (sec): '))
                        break
                    except ValueError:
                        print("Oops!  That was not a valid value.  Try again...")

                while True:
                    try:
                        t2 = float(input('T2. dephasing time of your qubits (sec): '))
                        break
                    except ValueError:
                        print("Oops!  That was not a valid value.  Try again...")

                while True:
                    try:
                        tg = float(input('Tg. the gate time of all gate operations in the circuit (sec): '))
                        break
                    except ValueError:
                        print("Oops!  That was not a valid value.  Try again...")
                
                print('Sampling with T1, T2, and Tg parameters.')
                # - Run the sampling that we want - #
                if circuit == 1: # three qubit code
                    data_file = three_qubit_sample(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
                elif circuit == 2: # Steane code
                    data_file = steane_sample(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
                elif circuit == 3: # fault tolerant Steane code
                    data_file = ft_steane_sample(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
                elif circuit == 4: # nine qubit code
                    data_file = nine_qubit_sample(initial_psi=psi, t1=t1, t2=t2, tg=tg, depolarization=dep, spam_prob=spam_prob, iterations=iterations, samples=samples)
                
            while True:
                print('...')
                print('Creating your distribution histogram for logical T1 of your system...')

                # - plot the infomration that we want - #
                if circuit == 1: # three qubit code
                    three_qubit_plot_t1(data_file = data_file)
                elif circuit == 2: # Steane code
                    steane_plot_t1(data_file = data_file)
                elif circuit == 3: # fault tolerant Steane code
                    ft_steane_plot_t1(data_file = data_file)
                elif circuit == 4: # nine qubit code
                    nine_qubit_plot_t1(data_file = data_file)

                print('- - - - - - - - - -')
            
        selection = bool(input('Would you like to take another sample? (leave blank if not) '))
        if not selection:
            break
            
    ### End the simulation
    print('- - - - - - - - - - - - - - - - - - -')
    print('Thank you for using our simulation! To simulate again, go ahead run the run_sim() cell again.')
    print('- - - - - END OF SIMULATION - - - - -')


##### - - - - - FUNCTIONS USED IN THE SIMULATION INTERFACE ABOVE. - - - - - #####

### Choose which circuit we want to run.   
def simulate_qec(circuit, psi, t1, t2, tg, depolarization, spam_prob, iterations):
    # circuit: which circuit do you want to simulate
    # psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    
    if circuit == 1:
        three_qubit_simulation(psi, t1, t2, tg, depolarization, spam_prob, iterations)
    elif circuit == 2:
        steane_simulation(psi, t1, t2, tg, depolarization, spam_prob, iterations)
    elif circuit == 3:
        ft_steane_simulation(psi, t1, t2, tg, depolarization, spam_prob, iterations)
    elif circuit == 4:
        nine_qubit_simulation(psi, t1, t2, tg, depolarization, spam_prob, iterations)
        
        
        
        
### - - - - - - 3-qubit simulation functions - - - - - - ###

### Run the 3 qubit simulation realistically with paramters and a certain number of iterations.       
def three_qubit_simulation(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # the time taken in one iteration of the 3 qubit code (sec)
        three_qubit_circuit_time = tg * (CNOT_gate_tot(0, 3) + CNOT_gate_tot(
            1, 3) + CNOT_gate_tot(0, 4) + CNOT_gate_tot(2, 4) + 2)

    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    qubit_error_probs = np.array([])
    
    ideal_state = np.dot(CNOT(1, 2, 5), np.dot(CNOT(0, 1, 5), np.kron(
        initial_psi, np.kron(zero, np.kron(zero, np.kron(zero, zero))))))
                          
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(5):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
        
    initialized_rho = initialize_three_qubit_realisitc(
        initial_psi, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

    rho = initialized_rho
    
    print('Working on plotting the probability of state measurements overtime...')
    # all_pops = np.array([])
    all_pops0 = np.array([])
    all_pops1 = np.array([])
    other_probs = np.array([])
    count = np.array([])
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Apply the circuit  times
    for i in range(iterations):
        count = np.append(count, i)
        rho = three_qubit_realistic(rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        M0 = np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(np.identity(2), np.identity(2)))))
        prob0 = np.trace(np.dot(M0.conj().T, np.dot(M0, rho)))

        M1 = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(np.identity(2), np.identity(2)))))
        prob1 = np.trace(np.dot(M1.conj().T, np.dot(M1, rho)))

        prob_other = 1 - prob0 - prob1
        
        all_pops0 = np.append(all_pops0, prob0)
        all_pops1 = np.append(all_pops1, prob1)
        other_probs = np.append(other_probs, prob_other)
        
        if i == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st iteration: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    print('- - -')
    ## -- Plotting our data and finding a line of best fit -- ##
    print('The ideal state of our system:')
    print_state_info(ideal_state, 5)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')
    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob )
    
    # Add data to the plot
    plt.figure(figsize=(10,4))
    plt.scatter(count, all_pops0, s = 1, c = 'cornflowerblue', label = '|000>')
    plt.scatter(count, all_pops1, s = 1, c ='seagreen', label = '|111>')
    plt.scatter(count, other_probs, s = 1, c ='red', label = 'any other state')
    plt.title('Qubit Meaurement Probability as a function of running 3 qubit code')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Probability of Measurement')
    plt.axhline(y = 1/np.e, color = 'y', linestyle = 'dotted')
    # Find and plot the fitted exponential for the |111> state
    xdata = (count)
    ydata = all_pops1
    popt, pcov = curve_fit(exp_decay, xdata, ydata)
    if 0<popt[1]<1:
        plt.plot(xdata, exp_decay(xdata, *popt), 'black', label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
        print('- - - - -')
        circuit_runs = 1/popt[1]
        if tg!=None:
            print('Calculated Circuit iterations until logical failure: ', circuit_runs)
            print('Calculated Logical T1: ', circuit_runs*three_qubit_circuit_time + 2*tg, 'sec')
        else:
            print('Calculated Circuit iterations until logical failure: ', circuit_runs)
    plt.ylim([-0.1, 1.1])
    plt.legend()
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Sample the 3 qubit code and save all of the data to a h5 file
def three_qubit_sample(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    print('Working on sampling the circuit overtime...')
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
   
    ideal_state = np.dot(CNOT(1, 2, 5), np.dot(CNOT(0, 1, 5), np.kron(
        initial_psi, np.kron(zero, np.kron(zero, np.kron(zero, zero))))))
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(5):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
        
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        # Initialize our logical state depending on parameters
        rho = initialize_three_qubit_realisitc(
            initial_psi, t1 = t1, t2 = t2, tg = tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
        
        # append the density matrix to a running array of them for this sample
        rho_per_sample = [rho]

        for i in range(iterations):
            rho = three_qubit_realistic(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
                
            # append the density matrix to a running array of them for this sample
            rho_per_sample = np.append(rho_per_sample, [rho], axis = 0)
        
        # append the density matrices for this sample to our total density matrices taken for all samples
        if k == 0:
            rho_overall = [rho_per_sample]
        else:
            rho_overall = np.append(rho_overall, [rho_per_sample], axis = 0)
        
        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
        if k == 9:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 10th sample: ', ct)
        if k == 99:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 100th sample: ', ct)
            
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Saving Data...')
    # ----- Saves data to a file ----- #
    expt_name = 'three_qubit_sample'
    print('Experiment name: ' + expt_name)
    
    expt_path = os.getcwd()
    print("Current working dir : %s" % expt_path)

    if not os.path.exists('data/' + expt_name):
        os.makedirs('data/' + expt_name)
    data_path = expt_path + '/data/' + expt_name

    fname = get_next_filename(data_path, expt_name, suffix='.h5')
    print('Current data file: ' + fname)
    print('Path to data file: ' + data_path)
          
    # save the parameters to a numpy array
    params = np.array([t1, t2, tg, spam_prob, depolarization]).astype(np.float64)
    
    print('File contents:')
    with SlabFile(data_path + '/' + fname, 'a') as f:
        # 'a': read/write/create
        # - Adds parameters to the file - #
        f.append('params', params)

        # - Adds data to the file - #
        f.append('ideal_state', ideal_state)
        f.append('rho_overall', rho_overall) # the denstiy matrix after every iteration divided into their samples
        print(tabulate(list(f.items())))
    print('..................')
    print('Sampling complete.')
    return data_path+'/'+fname
    
### Create a plot that samples the state of logical failure for the 3 qubit code
def three_qubit_plot_failure(data_file):
    # data_file: the path to your data file within your directory
    
    print('File contents:')
    # import data from file
    with SlabFile(r'' + data_file, 'r') as f:  
        print(tabulate(list(f.items())))
        params = array(f['params'])[0]    
        rho_overall = array(f['rho_overall'])[0]
        ideal_state = array(f['ideal_state'])[0]
    
    t1 = params[0]
    t2 = params[1]
    tg = params[2]
    spam_prob = params[3]
    depolarization = params[4]
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
        
    samples = len(rho_overall)
    
    count = np.array([])
    overall_count = np.array([])
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        rho_per_sample = rho_overall[k] # list of rho for this sample
        iterations = len(rho_per_sample)
        overall_count = np.append(overall_count, k)
        for i in range(iterations):
            rho = rho_per_sample[i] # rho in this iteration
            # expectation value when measuring our ideal state
            expectation_val = np.dot(ideal_state[np.newaxis].conj(), np.dot(rho, ideal_state))
            value = random.random() # number between 0 and 1 to use as a measure if we keep going or not
            # compare to our expectation value
            if value > expectation_val:
                break
        # append the count that we stopped at
        count = np.append(count, i)

    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # the time taken in one iteration of the 3 qubit code (sec)
        three_qubit_circuit_time = tg * (CNOT_gate_tot(0, 3) + CNOT_gate_tot(
            1, 3) + CNOT_gate_tot(0, 4) + CNOT_gate_tot(2, 4) + 2)

    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    print('- - -')
    
    # Plotting our data.
    print('The ideal state of our system:')
    print_state_info(ideal_state, 5)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', depolarization)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
    bin_num = int(samples/20) + 5
        
    n, bins, patches = plt.hist(
        count, bins = bin_num, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples')
    # - - - Fitting a curve to our plot - - - #  
    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)
    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)
    
    if tg != None:
        char_time = circuit_runs*three_qubit_circuit_time + 2*tg
        print('Characteristic time until failure: ', char_time, 'sec')

    print('... Number of bins:', len(bins)-1, '...')
    
    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    

### Create a plot that samples the logical T1 of your system over many runs        
def three_qubit_plot_t1(data_file):        
    # data_file: the path to your data file within your directory
    
    print('File contents:')
    # import data from file
    with SlabFile(r'' + data_file, 'r') as f:  
        print(tabulate(list(f.items())))
        params = array(f['params'])[0]    
        rho_overall = array(f['rho_overall'])[0]
        ideal_state = array(f['ideal_state'])[0]
    
    t1 = params[0]
    t2 = params[1]
    tg = params[2]
    spam_prob = params[3]
    depolarization = params[4]
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # the time taken in one iteration of the 3 qubit code (sec)
        three_qubit_circuit_time = tg * (CNOT_gate_tot(0, 3) + CNOT_gate_tot(
            1, 3) + CNOT_gate_tot(0, 4) + CNOT_gate_tot(2, 4) + 2)
        
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    samples = len(rho_overall)
    t1_times = np.array([])
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        rho_per_sample = rho_overall[k] # list of rho for this sample
        iterations = len(rho_per_sample)
        all_pops = np.array([])
        count = np.array([])
        for i in range(iterations):
            rho = rho_per_sample[i] # rho in this iteration
            count = np.append(count, i)
            # measure the probability of being in the state |111> from the density matrix
            M = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(np.identity(2), np.identity(2)))))
            pop = np.trace(np.dot(M.conj().T, np.dot(M, rho)))

            all_pops = np.append(all_pops, pop)

        xdata = count
        ydata = all_pops
        popt, pcov = curve_fit(exp_decay, xdata, ydata)
        circuit_runs = 1/popt[1]
        circuit_t1 = circuit_runs*three_qubit_circuit_time + 2*tg
        t1_times = np.append(t1_times, circuit_t1)
        
        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
        
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    # remove_oultiers in the data
    real_t1_times = t1_times[t1_times >=0]

    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # plotting our information:
    print('The ideal state of our system:')
    print_state_info(ideal_state, 5)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', depolarization)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
    bins = 'auto'

    n, bins, patches = plt.hist(t1_times, bins = bins, label = 'Distribution of Logical T1', color = 'cornflowerblue')
    plt.title('Distribution of Logical T1')
    plt.xlabel('Logical T1 (sec)') 
    plt.ylabel('Number of Samples') 

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    
    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
    bins = 'auto'

    n, bins, patches = plt.hist(
        real_t1_times, bins = bins, label = 'Distribution of Logical T1', color = 'cornflowerblue')
    plt.title('Distribution of Logical T1 (Real Times only)')
    plt.xlabel('Logical T1 (sec)') 
    plt.ylabel('Number of Samples') 

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
        
        
### - - - - - - Steane Code simulation functions - - - - - - ###

### Run the Steane code simulation realistically with paramters and a certain number of iterations.       
def steane_simulation(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, initial_psi))))))
    
    ideal_state = initialize_steane_logical_state(initial_state)
    ideal_bits = vector_state_to_bit_state(ideal_state, 10)[0]
    
    # add ancilla bits
    initial_state = np.kron(initial_state, np.kron(zero, np.kron(zero, zero)))
    
    qubit_error_probs = np.array([])
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(10):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    rho = initial_rho
    
    print('Working on plotting the probability of state measurements overtime...')
    all_pops0 = np.array([])
    all_pops1 = np.array([])
    other_probs = np.array([])
    count = np.array([])
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)
    # Apply the circuit  times
    for i in range(iterations):
        count = np.append(count, i)
        rho = realistic_steane(
        rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        # Measurement operator to see if you are in the 0 logical state
        M0 = np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3))))))))
        
        # probability of being in the 0 logical state
        prob0 = np.trace(np.dot(M0.conj().T, np.dot(M0, rho)))
        
        # Measurement operator to see if you are in the 1 logical state
        M1 = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3))))))))
        
        # probability of being in the 1 logical state
        prob1 = np.trace(np.dot(M1.conj().T, np.dot(M1, rho)))
        
        # any other probability
        prob_other = 1 - prob0 - prob1
        
        all_pops0 = np.append(all_pops0, prob0)
        all_pops1 = np.append(all_pops1, prob1)
        other_probs = np.append(other_probs, prob_other)
    
        if i == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st iteration: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    ## -- Plotting our data and finding a line of best fit -- ##
    print('The ideal state of our system:')
    print_state_info(ideal_state, 10)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')
    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob )
    
    # Add data to the plot
    plt.figure(figsize=(10,4))
    plt.scatter(count, all_pops0, s = 1, c = 'cornflowerblue', label = 'Logical |0>')
    plt.scatter(count, all_pops1, s = 1, c ='seagreen', label = 'Logical |1>')
    plt.scatter(count, other_probs, s = 1, c ='red', label = 'any other state')
    plt.title('Qubit Meaurement Probability as a function of running Steane code')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Probability of Measurement')
    plt.axhline(y = 1/np.e, color = 'y', linestyle = 'dotted')
    # Find and plot the fitted exponential for the |111> state
    xdata = (count)
    ydata = all_pops1
    popt, pcov = curve_fit(exp_decay, xdata, ydata)
    plt.plot(xdata, exp_decay(xdata, *popt), 'black', label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - - - -')
    circuit_runs = 1/popt[1]
    if tg!=None:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
        print('Calculated Logical T1: ', (((circuit_runs * 29) + 2) * tg), 'sec')
    else:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
    plt.ylim([-0.1, 1.1])
    plt.legend()
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Sample the steane code and save all of the data to a h5 file
def steane_sample(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    print('Working on sampling the circuit overtime...')
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
   
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, initial_psi))))))
    
    ideal_state = initialize_steane_logical_state(initial_state)
    ideal_bits = vector_state_to_bit_state(ideal_state, 10)[0]
    
    # add ancilla bits
    initial_state = np.kron(initial_state, np.kron(zero, np.kron(zero, zero)))
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(10):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
        
    # Apply the circuit for (iteration) number of times (samples) times
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        rho = realistic_steane(
                initial_rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
        
        # append the density matrix to a running array of them for this sample
        rho_per_sample = [rho]

        for i in range(iterations-1):
            
            # for larger circuits this will be useful to know
            if (i == 0) and (k == 0):
                # ct stores current time
                ct = datetime.datetime.now()
                print('Time after 1st iteration: ', ct)
                
            rho = realistic_steane(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
            
            # append the density matrix to a running array of them for this sample
            rho_per_sample = np.append(rho_per_sample, [rho], axis = 0)
        
        
        # append the density matrices for this sample to our total density matrices taken for all samples
        if k == 0:
            rho_overall = [rho_per_sample]
        else:
            rho_overall = np.append(rho_overall, [rho_per_sample], axis = 0)
        
        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
        if k == 9:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 10th sample: ', ct)
        if k == 99:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 100th sample: ', ct)
            
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Saving Data...')
    # ----- Saves data to a file ----- #
    expt_name = 'steane_sample'
    print('Experiment name: ' + expt_name)
    
    expt_path = os.getcwd()
    print("Current working dir : %s" % expt_path)

    if not os.path.exists('data/' + expt_name):
        os.makedirs('data/' + expt_name)
    data_path = expt_path + '/data/' + expt_name

    fname = get_next_filename(data_path, expt_name, suffix='.h5')
    print('Current data file: ' + fname)
    print('Path to data file: ' + data_path)
          
    # save the parameters to a numpy array
    params = np.array([t1, t2, tg, spam_prob, depolarization]).astype(np.float64)

    print('File contents:')
    with SlabFile(data_path + '/' + fname, 'a') as f:
        # 'a': read/write/create
        # - Adds parameters to the file - #
        f.append('params', params)

        # - Adds data to the file - #
        f.append('ideal_state', ideal_state)
        f.append('rho_overall', rho_overall) # the denstiy matrix after every iteration divided into their samples
        print(tabulate(list(f.items())))
    print('..................')
    print('Sampling complete.')
    return data_path+'/'+fname
          
### Create a plot that samples the state of logical failure for the Steane code
def steane_plot_failure(data_file):
    # data_file: the path to your data file within your directory
    
    # calculating the number of gates in the steane code
    total_gates_z = 1 + CNOT_gate_tot(7, 3) + CNOT_gate_tot(7, 4) + CNOT_gate_tot(7, 5) + 1 + CNOT_gate_tot(
        8, 0) + CNOT_gate_tot(8, 2) + CNOT_gate_tot(8, 4) + CNOT_gate_tot(8, 6) + CNOT_gate_tot(
        9, 1) + CNOT_gate_tot(9, 2) + CNOT_gate_tot(9, 5) + CNOT_gate_tot(9, 6) + 1 + 2 #CNOT GATES

    total_gates_x = total_gates_z + (2*12) # CZ gates
    
    total_gates = total_gates_x + total_gates_z
          
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    print('File contents:')
    # import data from file
    with SlabFile(r'' + data_file, 'r') as f:  
        print(tabulate(list(f.items())))
        params = array(f['params'])[0]    
        rho_overall = array(f['rho_overall'])[0]
        ideal_state = array(f['ideal_state'])[0]
    
    t1 = params[0]
    t2 = params[1]
    tg = params[2]
    spam_prob = params[3]
    depolarization = params[4]
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # the time taken in one iteration of the steane code (sec)
        steane_circuit_time = (total_gates + 4)*tg

    samples = len(rho_overall)
    count = np.array([])
    overall_count = np.array([])
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        rho_per_sample = rho_overall[k] # list of rho for this sample
        iterations = len(rho_per_sample)
        
        overall_count = np.append(overall_count, k)
        for i in range(iterations):
            rho = rho_per_sample[i] # rho in this iteration
            
            # expectation value when measuring our ideal state
            expectation_val = np.dot(ideal_state[np.newaxis].conj(), np.dot(rho, ideal_state))
            value = random.random() # number between 0 and 1 to use as a measure if we keep going or not
            # compare to our expectation value
            if value > expectation_val:
                break
                
        # append the count that we stopped at
        count = np.append(count, i)
        
    
        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)

    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # Plotting our data.
    print('The ideal state of our system:')
    print_state_info(ideal_state, 10)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', depolarization)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    bin_num = int(samples/20) + 5

    n, bins, patches = plt.hist(
        count, bins = bin_num, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples') 
    # - - - Fitting a curve to our plot - - - #  
    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)

    char_time = (((circuit_runs * 29) + 2) * tg)
    print('Characteristic time until failure: ', char_time, 'sec')

    print('... Number of bins:', len(bins)-1, '...')

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

    
### Create a plot that samples the logical T1 of your steane code over many runs        
def steane_plot_t1(data_file):        
    # data_file: the path to your data file within your directory
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    # calculating the number of gates in the steane code
    total_gates_z = 1 + CNOT_gate_tot(7, 3) + CNOT_gate_tot(7, 4) + CNOT_gate_tot(7, 5) + 1 + CNOT_gate_tot(
        8, 0) + CNOT_gate_tot(8, 2) + CNOT_gate_tot(8, 4) + CNOT_gate_tot(8, 6) + CNOT_gate_tot(
        9, 1) + CNOT_gate_tot(9, 2) + CNOT_gate_tot(9, 5) + CNOT_gate_tot(9, 6) + 1 + 2 #CNOT GATES

    total_gates_x = total_gates_z + (2*12) # CZ gates
    
    total_gates = total_gates_x + total_gates_z
    
    
    print('File contents:')
    # import data from file
    with SlabFile(r'' + data_file, 'r') as f:  
        print(tabulate(list(f.items())))
        params = array(f['params'])[0]    
        rho_overall = array(f['rho_overall'])[0]
        ideal_state = array(f['ideal_state'])[0]
    
    t1 = params[0]
    t2 = params[1]
    tg = params[2]
    spam_prob = params[3]
    depolarization = params[4]
    
    if ((t1!=None) and (t2!=None) and (tg!=None)):
        # the time taken in one iteration of the steane code (sec)
        steane_circuit_time = (total_gates + 4)*tg

    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)
    
    samples = len(rho_overall)        
    t1_times = np.array([])
    for k in range(samples):
        rho_per_sample = rho_overall[k] # list of rho for this sample
        iterations = len(rho_per_sample)
        all_pops = np.array([])
        count = np.array([])
        # run the circuit many times
        for i in range(iterations):
            rho = rho_per_sample[i] # rho in this iteration
            count = np.append(count, i)
            
            # measure the probability of being in the Logical |1> state from the density matrix
            M = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
                one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
                zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**3)))))))) + np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**3)))))))) + np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
                one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**3))))))))
        
            # probability of being in the 1 logical state
            pop = np.trace(np.dot(M.conj().T, np.dot(M, rho)))
            
            all_pops = np.append(all_pops, pop)

        xdata = count
        ydata = all_pops
        popt, pcov = curve_fit(exp_decay, xdata, ydata)
        circuit_runs = 1/popt[1]
        circuit_t1 = (total_gates * tg)
        t1_times = np.append(t1_times, circuit_t1)

        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    # remove_oultiers in the data
    real_t1_times = t1_times[t1_times >=0]

    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # plotting our information:
    print('The ideal state of our system:')
    print_state_info(ideal_state, 10)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', depolarization)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
    bins = 'auto'

    n, bins, patches = plt.hist(t1_times, bins = bins, label = 'Distribution of Logical T1', color = 'cornflowerblue')
    plt.title('Distribution of Logical T1')
    plt.xlabel('Logical T1 (sec)') 
    plt.ylabel('Number of Samples') 

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    
    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
    bins = 'auto'

    n, bins, patches = plt.hist(
        real_t1_times, bins = bins, label = 'Distribution of Logical T1', color = 'cornflowerblue')
    plt.title('Distribution of Logical T1 (Real Times only)')
    plt.xlabel('Logical T1 (sec)') 
    plt.ylabel('Number of Samples') 

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
        
### - - - - - - Fault Tolerant Steane Code simulation functions - - - - - - ###

### Run the Steane code simulation realistically with paramters and a certain number of iterations.       
def ft_steane_simulation(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
            zero, np.kron(zero, zero)))))))))
    
    # to save time we will just calculate the normal steane and add 2 ancillas
    ideal_state = initialize_steane_logical_state(initial_state)
    ideal_state = np.kron(ideal_state, np.kron(zero, zero))
    ideal_state = ancilla_reset(ideal_state, 5)
    
    qubit_error_probs = np.array([])
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(12):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    # add 2 ancillas for the ft version of steane
    initial_state = np.kron(initial_state, np.kron(zero, zero))
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    rho = initial_rho
    
    print('Working on plotting the probability of state measurements overtime...')
    # all_pops = np.array([])
    all_pops0 = np.array([])
    all_pops1 = np.array([])
    other_probs = np.array([])
    count = np.array([])
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)

    # Apply the circuit  times
    for i in range(iterations):
        count = np.append(count, i)
        rho = realistic_ft_steane(
        rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        # Measurement operator to see if you are in the 0 logical state
        M0 = np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5))))))))
        
        # probability of being in the 0 logical state
        prob0 = np.trace(np.dot(M0.conj().T, np.dot(M0, rho)))
        
        # Measurement operator to see if you are in the 1 logical state
        M1 = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
            zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
            zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
            one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5))))))))
        
        # probability of being in the 1 logical state
        prob1 = np.trace(np.dot(M1.conj().T, np.dot(M1, rho)))
        
        # any other probability
        prob_other = 1 - prob0 - prob1
        
        all_pops0 = np.append(all_pops0, prob0)
        all_pops1 = np.append(all_pops1, prob1)
        other_probs = np.append(other_probs, prob_other)
    
        if i == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st iteration: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    
    print('- - -')
    ## -- Plotting our data and finding a line of best fit -- ##
    print('The ideal state of our system:')
    print_state_info(ideal_state, 12)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')
    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob )
    
    # Add data to the plot
    plt.figure(figsize=(10,4))
    plt.scatter(count, all_pops0, s = 1, c = 'cornflowerblue', label = 'Logical |0>')
    plt.scatter(count, all_pops1, s = 1, c ='seagreen', label = 'Logical |1>')
    plt.scatter(count, other_probs, s = 1, c ='red', label = 'any other state')
    plt.title('Qubit Meaurement Probability as a function of running fault tolerant Steane code')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Probability of Measurement')
    plt.axhline(y = 1/np.e, color = 'y', linestyle = 'dotted')
    # Find and plot the fitted exponential for the |111> state
    xdata = (count)
    ydata = all_pops1
    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black', label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - - - -')
    circuit_runs = 1/popt[1]
    if tg!=None:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
        print('Calculated Logical T1: ', (((circuit_runs * 29) + 2) * tg), 'sec')
    else:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
    plt.ylim([-0.1, 1.1])
    plt.legend()
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()


### Sample the fault tolerant steane code and save all of the data to a h5 file
def ft_steane_sample(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    print('Working on sampling the circuit overtime...')
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
   
    initial_state = np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
        initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(initial_psi, np.kron(
            zero, np.kron(zero, zero)))))))))
    
    # to save time we will just calculate the normal steane and add 2 ancillas
    ideal_state = initialize_steane_logical_state(initial_state)
    ideal_state = np.kron(ideal_state, np.kron(zero, zero))
    ideal_state = ancilla_reset(ideal_state, 5)
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(12):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    initial_state = np.kron(initial_state, np.kron(zero, zero)) # add 2 ancillas to initial state
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)
    
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        rho = realistic_ft_steane(
                initial_rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
        # append the density matrix to a running array of them for this sample
        rho_per_sample = [rho]
        for i in range(iterations):
            
            # for larger circuits this will be useful to know
            if (i == 0) and (k == 0):
                # ct stores current time
                ct = datetime.datetime.now()
                print('Time after 1st iteration: ', ct)
         
            rho = realistic_ft_steane(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
            
            # append the density matrix to a running array of them for this sample
            rho_per_sample = np.append(rho_per_sample, [rho], axis = 0)
        
            
        # append the density matrices for this sample to our total density matrices taken for all samples
        if k == 0:
            rho_overall = [rho_per_sample]
        else:
            rho_overall = np.append(rho_overall, [rho_per_sample], axis = 0)
        
        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
        if k == 9:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 10th sample: ', ct)
        if k == 99:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 100th sample: ', ct)
            
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Saving Data...')
    # ----- Saves data to a file ----- #
    expt_name = 'ft_staene_sample'
    print('Experiment name: ' + expt_name)
    
    expt_path = os.getcwd()
    print("Current working dir : %s" % expt_path)

    if not os.path.exists('data/' + expt_name):
        os.makedirs('data/' + expt_name)
    data_path = expt_path + '/data/' + expt_name

    fname = get_next_filename(data_path, expt_name, suffix='.h5')
    print('Current data file: ' + fname)
    print('Path to data file: ' + data_path)
          
    # save the parameters to a numpy array
    params = np.array([t1, t2, tg, spam_prob, depolarization]).astype(np.float64)

    print('File contents:')
    with SlabFile(data_path + '/' + fname, 'a') as f:
        # 'a': read/write/create
        # - Adds parameters to the file - #
        f.append('params', params)

        # - Adds data to the file - #
        f.append('ideal_state', ideal_state)
        f.append('rho_overall', rho_overall) # the denstiy matrix after every iteration divided into their samples
        print(tabulate(list(f.items())))
    print('..................')
    print('Sampling complete.')
    return data_path+'/'+fname

### Create a plot that samples the state of logical failure for the Steane code
def ft_steane_plot_failure(data_file):
    # data_file: the path to your data file within your directory
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    print('File contents:')
   # import data from file
    with SlabFile(r'' + data_file, 'r') as f:  
        print(tabulate(list(f.items())))
        params = array(f['params'])[0]    
        rho_overall = array(f['rho_overall'])[0]
        ideal_state = array(f['ideal_state'])[0]
    
    t1 = params[0]
    t2 = params[1]
    tg = params[2]
    spam_prob = params[3]
    depolarization = params[4]
    
    
    samples = len(rho_overall)
    count = np.array([])
    overall_count = np.array([])
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        rho_per_sample = rho_overall[k] # list of rho for this sample
        iterations = len(rho_per_sample)
        
        overall_count = np.append(overall_count, k)
        for i in range(iterations):
            rho = rho_per_sample[i] # rho in this iteration
            
            # expectation value when measuring our ideal state
            expectation_val = np.dot(ideal_state[np.newaxis].conj(), np.dot(rho, ideal_state))
            value = random.random() # number between 0 and 1 to use as a measure if we keep going or not
            # compare to our expectation value
            if value > expectation_val:
                break
                
        # append the count that we stopped at
        count = np.append(count, i)
        
    
        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)

    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # Plotting our data.
    print('The ideal state of our system:')
    print_state_info(ideal_state, 10)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', depolarization)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    bin_num = int(samples/20) + 5

    n, bins, patches = plt.hist(
        count, bins = bin_num, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples') 
    # - - - Fitting a curve to our plot - - - #  
    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)
    print('(Remember that the fault tolerant steane code has repititions within the circuit.)')

    print('... Number of bins:', len(bins)-1, '...')

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()


### Create a plot that samples the logical T1 of your steane code over many runs        
def ft_steane_plot_t1(data_file):        
    # data_file: the path to your data file within your directory
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    print('File contents:')
    # import data from file
    with SlabFile(r'' + data_file, 'r') as f:  
        print(tabulate(list(f.items())))
        params = array(f['params'])[0]    
        rho_overall = array(f['rho_overall'])[0]
        ideal_state = array(f['ideal_state'])[0]
    
    t1 = params[0]
    t2 = params[1]
    tg = params[2]
    spam_prob = params[3]
    depolarization = params[4]
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)
    
    samples = len(rho_overall)        
    t1_times = np.array([])
    for k in range(samples):
        rho_per_sample = rho_overall[k] # list of rho for this sample
        iterations = len(rho_per_sample)
        all_pops = np.array([])
        count = np.array([])
        # run the circuit many times
        for i in range(iterations):
            rho = rho_per_sample[i] # rho in this iteration
            count = np.append(count, i)
            
            # measure the probability of being in the Logical |1> state from the density matrix
            M = np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                one_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
                zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                one_meas, np.kron(zero_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                one_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                zero_meas, np.kron(one_meas, np.kron(one_meas, np.identity(2**5)))))))) + np.kron(
                zero_meas, np.kron(zero_meas, np.kron(one_meas, np.kron(zero_meas, np.kron(
                one_meas, np.kron(one_meas, np.kron(zero_meas, np.identity(2**5))))))))
        
            # probability of being in the 1 logical state
            pop = np.trace(np.dot(M.conj().T, np.dot(M, rho)))
            
            all_pops = np.append(all_pops, pop)

        xdata = count
        ydata = all_pops
        popt, pcov = curve_fit(exp_decay, xdata, ydata)
        circuit_runs = 1/popt[1]
        t1_times = np.append(t1_times, circuit_runs)

        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    # remove_oultiers in the data
    real_t1_times = t1_times[t1_times >=0]

    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # plotting our information:
    print('The ideal state of our system:')
    print_state_info(ideal_state, 12)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', depolarization)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
    bins = 'auto'

    n, bins, patches = plt.hist(t1_times, bins = bins, label = 'Distribution of Logical T1', color = 'cornflowerblue')
    plt.title('Distribution of Logical T1 (In terms of circuit runs)')
    plt.xlabel('Logical T1 (In terms of circuit runs)') 
    plt.ylabel('Number of Samples') 

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    
    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
    bins = 'auto'

    n, bins, patches = plt.hist(
        real_t1_times, bins = bins, label = 'Distribution of Logical T1', color = 'cornflowerblue')
    plt.title('Distribution of Logical T1 (In terms of circuit runs) (Real Times only)')
    plt.xlabel('Logical T1 (In terms of circuit runs)') 
    plt.ylabel('Number of Samples') 

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
        
    
### - - - - - - 9-qubit Code simulation functions - - - - - - ###

### Run the nine qubit code simulation realistically with paramters and a certain number of iterations.       
def nine_qubit_simulation(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    initial_state = np.kron(initial_psi, np.kron(zero, np.kron(zero, np.kron(
        zero, np.kron(zero, np.kron(zero, np.kron(zero, np.kron(
            zero, np.kron(zero, np.kron(zero, zero))))))))))
    
    ideal_state = nine_qubit_initialize_logical_state(initial_psi)
    
    qubit_error_probs = np.array([])
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(11):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    rho = initial_rho
    
    print('Working on plotting the probability of state measurements overtime...')
    # all_pops = np.array([])
    all_pops0 = np.array([])
    all_pops1 = np.array([])
    other_probs = np.array([])
    count = np.array([])
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)
    
    # Apply the circuit  times
    for i in range(iterations):
        count = np.append(count, i)
        rho = nine_qubit_realistic(
        rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        # Measurement operator to see if you are in the 0 logical state
        M0 = np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                     ) + np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                     ) + np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                     ) + np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                     ) + np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                     ) + np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                     ) + np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                     ) + np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                     )
        
        # probability of being in the 0 logical state
        prob0 = np.trace(np.dot(M0.conj().T, np.dot(M0, rho)))
        
        # Measurement operator to see if you are in the 1 logical state
        M1 = np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                     ) - np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                     ) - np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                     ) + np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                     ) - np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                     ) + np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                     ) + np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                     ) - np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                     one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                     )
        
        # probability of being in the 1 logical state
        prob1 = np.trace(np.dot(M1.conj().T, np.dot(M1, rho)))
        
        # any other probability
        prob_other = 1 - prob0 - prob1
        
        all_pops0 = np.append(all_pops0, prob0)
        all_pops1 = np.append(all_pops1, prob1)
        other_probs = np.append(other_probs, prob_other)

        if i == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st iteration: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')
    print('- - -')        
    ## -- Plotting our data and finding a line of best fit -- ##
    print('The ideal state of our system:')
    print_state_info(ideal_state, 11)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')
    print('Depolarizing error by probability at each qubit: ', qubit_error_probs)
    print('SPAM error probability: ', spam_prob )
    
    # Add data to the plot
    plt.figure(figsize=(10,4))
    plt.scatter(count, all_pops0, s = 1, c = 'cornflowerblue', label = 'Logical |0>')
    plt.scatter(count, all_pops1, s = 1, c ='seagreen', label = 'Logical |1>')
    plt.scatter(count, other_probs, s = 1, c ='red', label = 'any other state')
    plt.title('Qubit Meaurement Probability as a function of running nine qubit code')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Probability of Measurement')
    plt.axhline(y = 1/np.e, color = 'y', linestyle = 'dotted')
    # Find and plot the fitted exponential for the |1>_L state
    xdata = (count)
    ydata = all_pops1
    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black', label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - - - -')
    circuit_runs = 1/popt[1]
    if tg!=None:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
        print('Calculated Logical T1: ', (((circuit_runs * 29) + 2) * tg), 'sec')
    else:
        print('Calculated Circuit iterations until logical failure: ', circuit_runs)
    plt.ylim([-0.1, 1.1])
    plt.legend()
    
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()

### Sample the nine qubit code and save all of the data to a h5 file
def nine_qubit_sample(initial_psi, t1, t2, tg, depolarization, spam_prob, iterations, samples):
    # initial_psi: initial state of your system
    # t1: The relaxation time of each physical qubit in your system
    # t2: The dephasing time of each physical qubit in your system
    # tg: The gate time of your gate operations 
    # depolarization: the probability for errors of each qubit in your system
    # spam_prob: The pobability that you have a state prep or measurement error
    # iterations: number of times you want to run the circuit
    # samples: number of times you want to sample your data
    
    print('Working on sampling the circuit overtime...')
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
   
    initial_state = np.kron(initial_psi, np.kron(zero, np.kron(zero, np.kron(zero, np.kron(zero, np.kron(
        zero, np.kron(zero, np.kron(zero, np.kron(zero, np.kron(zero, zero))))))))))
    
    # to save time we will just calculate the normal nine qubit and add 2 ancillas
    ideal_state = nine_qubit_initialize_logical_state(initial_psi)
    
    if depolarization != None:
        qubit_error_probs = np.array([])            
        for i in range(11):
            qubit_error_probs = np.append(qubit_error_probs, depolarization)
    else:
        qubit_error_probs = None
    
    initial_rho = np.kron(initial_state, initial_state[np.newaxis].conj().T)

    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        # Initialize our logical state depending on parameters
        rho = initialize_nine_qubit_realisitc(
            initial_rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)

        for i in range(iterations):
            # append the density matrix to a running array of them for this sample
            if i == 0:
                rho_per_sample = [rho]
            else:
                rho_per_sample = np.append(rho_per_sample, [rho], axis = 0)
        
            rho = nine_qubit_realistic(
                rho, t1=t1, t2=t2, tg=tg, qubit_error_probs=qubit_error_probs, spam_prob=spam_prob)
            
            # for larger circuits this will be useful to know
            if (i == 0) and (k == 0):
                # ct stores current time
                ct = datetime.datetime.now()
                print('Time after 1st iteration: ', ct)
                
        # append the density matrices for this sample to our total density matrices taken for all samples
        if k == 0:
            rho_overall = [rho_per_sample]
        else:
            rho_overall = np.append(rho_overall, [rho_per_sample], axis = 0)
            
        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
        if k == 9:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 10th sample: ', ct)
        if k == 99:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 100th sample: ', ct)
            
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Saving Data...')
    # ----- Saves data to a file ----- #
    expt_name = 'nine_qubit_sample'
    print('Experiment name: ' + expt_name)
    
    expt_path = os.getcwd()
    print("Current working dir : %s" % expt_path)

    if not os.path.exists('data/' + expt_name):
        os.makedirs('data/' + expt_name)
    data_path = expt_path + '/data/' + expt_name

    fname = get_next_filename(data_path, expt_name, suffix='.h5')
    print('Current data file: ' + fname)
    print('Path to data file: ' + data_path)
          
    # save the parameters to a numpy array
    params = np.array([t1, t2, tg, spam_prob, depolarization]).astype(np.float64)

    print('File contents:')
    with SlabFile(data_path + '/' + fname, 'a') as f:
        # 'a': read/write/create
        # - Adds parameters to the file - #
        f.append('params', params)

        # - Adds data to the file - #
        f.append('ideal_state', ideal_state)
        f.append('rho_overall', rho_overall) # the denstiy matrix after every iteration divided into their samples
        print(tabulate(list(f.items())))
    print('..................')
    print('Sampling complete.')
    return data_path+'/'+fname


### Create a plot that samples the state of logical failure for the nine qubit code
def nine_qubit_plot_failure(data_file):
    # data_file: the path to your data file within your directory
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    print('File contents:')
    # import data from file
    with SlabFile(r'' + data_file, 'r') as f:  
        print(tabulate(list(f.items())))
        params = array(f['params'])[0]    
        rho_overall = array(f['rho_overall'])[0]
        ideal_state = array(f['ideal_state'])[0]
    
    t1 = params[0]
    t2 = params[1]
    tg = params[2]
    spam_prob = params[3]
    depolarization = params[4]
    
    
    samples = len(rho_overall)
    count = np.array([])
    overall_count = np.array([])
    # Apply the circuit for (iteration) number of times (samples) times
    for k in range(samples):
        rho_per_sample = rho_overall[k] # list of rho for this sample
        iterations = len(rho_per_sample)
        
        overall_count = np.append(overall_count, k)
        for i in range(iterations):
            rho = rho_per_sample[i] # rho in this iteration
            
            # expectation value when measuring our ideal state
            expectation_val = np.dot(ideal_state[np.newaxis].conj(), np.dot(rho, ideal_state))
            value = random.random() # number between 0 and 1 to use as a measure if we keep going or not
            # compare to our expectation value
            if value > expectation_val:
                break
                
        # append the count that we stopped at
        count = np.append(count, i)
        
    
        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)

    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # Plotting our data.
    print('The ideal state of our system:')
    print_state_info(ideal_state, 10)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', depolarization)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    bin_num = int(samples/20) + 5

    n, bins, patches = plt.hist(
        count, bins = bin_num, label = 'Failure iteration Distribution', color = 'cornflowerblue')
    plt.title('Distribution of circuit failure after number of iterations')
    plt.xlabel('Iterations until logical state failure') 
    plt.ylabel('Number of Samples') 
    # - - - Fitting a curve to our plot - - - #  
    xdata = (bins[1:])[n!=0]
    ydata = n[n!=0]

    popt, pcov = curve_fit(exp_decay, xdata, ydata)

    plt.plot(xdata, exp_decay(xdata, *popt), 'black',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt), linestyle = 'dashed')
    print('- - -')

    circuit_runs = 1/popt[1]
    print('Characteristic number of runs until failure: ', circuit_runs)
    print('... Number of bins:', len(bins)-1, '...')

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    

### Create a plot that samples the logical T1 of your nine qubit code over many runs        
def nine_qubit_plot_t1(data_file):        
    # data_file: the path to your data file within your directory
    
    # ct stores current time
    ct = datetime.datetime.now()
    print('Start Time: ', ct)
    
    print('File contents:')
    # import data from file
    with SlabFile(r'' + data_file, 'r') as f:  
        print(tabulate(list(f.items())))
        params = array(f['params'])[0]    
        rho_overall = array(f['rho_overall'])[0]
        ideal_state = array(f['ideal_state'])[0]
    
    t1 = params[0]
    t2 = params[1]
    tg = params[2]
    spam_prob = params[3]
    depolarization = params[4]
    
    # Masurement operators for individual qubits
    zero_meas = np.kron(zero, zero[np.newaxis].conj().T)
    one_meas = np.kron(one, one[np.newaxis].conj().T)
    
    samples = len(rho_overall)        
    t1_times = np.array([])
    for k in range(samples):
        rho_per_sample = rho_overall[k] # list of rho for this sample
        iterations = len(rho_per_sample)
        all_pops = np.array([])
        count = np.array([])
        # run the circuit many times
        for i in range(iterations):
            rho = rho_per_sample[i] # rho in this iteration
            count = np.append(count, i)
            
            # measure the probability of being in the Logical |1> state from the density matrix
            M = np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                        ) - np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                        ) - np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                        ) + np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                        ) - np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                        ) + np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                        ) + np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                        zero_meas, np.kron(zero_meas, np.kron(zero_meas, np.Identity(2**2)))))))))
                        ) - np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.kron(
                        one_meas, np.kron(one_meas, np.kron(one_meas, np.Identity(2**2)))))))))
                        )
            
            # probability of being in the 1 logical state
            pop = np.trace(np.dot(M.conj().T, np.dot(M, rho)))
            
            all_pops = np.append(all_pops, pop)

        xdata = count
        ydata = all_pops
        popt, pcov = curve_fit(exp_decay, xdata, ydata)
        circuit_runs = 1/popt[1]
        t1_times = np.append(t1_times, circuit_runs)

        if k == 0:
            # ct stores current time
            ct = datetime.datetime.now()
            print('Time after 1st sample: ', ct)
    
    ct = datetime.datetime.now()
    print('End Time: ', ct)
    
    # remove_oultiers in the data
    real_t1_times = t1_times[t1_times >=0]

    print('Plotting...')
    print('Note that the fitted line may have errors')

    print('- - -')
    # plotting our information:
    print('The ideal state of our system:')
    print_state_info(ideal_state, 11)
    print('- - -')
    print('Physical T1: ', t1, ' sec')
    print('Physical T2 range:', t2, ' sec')
    print('Gate time (Tg): ', tg, 'sec')

    print('Depolarizing error by probability at each qubit: ', depolarization)
    print('SPAM error probability: ', spam_prob)


    print('- - -')
    print('Total number of samples: ', samples)
    print('Number of iterations per sample: ', iterations)

    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
    bins = 'auto'

    n, bins, patches = plt.hist(t1_times, bins = bins, label = 'Distribution of Logical T1', color = 'cornflowerblue')
    plt.title('Distribution of Logical T1')
    plt.xlabel('Logical T1 (sec)') 
    plt.ylabel('Number of Samples') 

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    
    # Plotting the error state probabilities
    plt.figure(figsize=(10,4))# passing the histogram function
    
    bins = 'auto'

    n, bins, patches = plt.hist(
        real_t1_times, bins = bins, label = 'Distribution of Logical T1', color = 'cornflowerblue')
    plt.title('Distribution of Logical T1 (Real Times only)')
    plt.xlabel('Logical T1 (sec)') 
    plt.ylabel('Number of Samples') 

    # Add a Legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()