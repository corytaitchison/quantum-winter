"""
Module containing:

- QFT algorithm function
- QFT matrix function
- QFT state initialisation function
- Statevector printing function
"""

from Gates import *
from Circuits import *

def qft(circuit: Circuit, decompose = None, *args, **kwargs):
    """Implement the QFT algorithm onto the QiskitCircuit or MatrixCircuit.
    
    Arguments
    ---------
    circuit: Circuit
        The QiskitCircuit or MatrixCircuit object on which the QFT algorithm is applied
    decompose: optional
        If None, use the exact rotation gates. If a function, use that function to compute
        approximations to the gates in terms of H and T. 
    *args, **kwargs:
        Optional arguments to pass to the ControlGate() function.
    """
    n = circuit.n
    for i in range(n):
        # Apply a Hadamard gate to each qubit
        circuit.apply(H, i)
        # Apply a controlled-R(k) gate to each qubit, controlled by each subsequent qubit
        for k in range(2, n + 1 - i):
            circuit.apply(ControlGate(R(k), decompose = decompose, *args, **kwargs), i + k - 1, i)
            
    # Swap the order of the qubits:
    for i in range(int(np.floor(n/2))):
        circuit.swap(i, n - 1 - i)        
        
def qft_matrix(n = 3):
    """Compute the Quantum Fourier Transform (QFT) matrix based on Equation 1.1.
    
    Arguments
    ---------
    n: int, optional
        The number of qubits in the circuit (default 3)
    """
    # Calculate dimension of the state space, given n qubits
    dim = 2 ** n 
    
    # Create 2^n x 2^n matrices 
    # where row_mesh has 0...N-1 as rows and col_mesh has 0...N-1 as columns
    row_mesh, col_mesh = np.meshgrid(range(dim), range(dim))
            
    # Compute the matrix elements as per Equation 1.1. 
    # and return the matrix
    omega = np.exp(2j * np.pi / dim)
    return omega ** (row_mesh * col_mesh) / np.sqrt(dim)

def qft_initialise(circuit: Circuit, state: int, qt_notation = True):
    """Initialise a quantum circuit such that when QFT is applied, the measurement basis state 'state' 
    is outputted with amplitude 1.
    
    Arguments
    ---------
    circuit: Circuit
        A (normally blank) Circuit object (QiskitCircuit or MatrixCircuit) on which to apply the gates
    state: int
        The index of the desired output state; 0 = |00>, 1 = |001>, 2 = |011>, ...
    qt_notation: bool, optional
        Whether to apply the gates such that the output is consitent with the Qiskit state vectors
        and notation (default True)
    """
    # The number of qubits in the circuit
    n = circuit.n
    
    # Reverse the ordering if using Qiskit to comply with their notation
    # e.g. |011> -> |110> 
    if qt_notation:
        # Write as a bitstring
        b = f'{state:0{n}b}'
        # Reverse the order of the bitstring and convert back to an integer
        state = int(b[::-1], 2)
    
    # The number of S/T/P gates to apply to each qubit     
    repetitions = np.mod([2 ** n - state * 2 ** (n - 1 - i) for i in range(n)], 2 ** n)
    
    for i in range(n):
        # Apply a Hadamard to each qubit
        circuit.apply(H, i)
    for i in range(n):
        # For each qubit, i, apply the S/T/P gate k times
        # where k is given by the repetitions array
        for k in range(repetitions[i]):
            if n == 2:
                circuit.apply(S, i) # same as the P(π/2)-gate except just prints it nicer in circuit diagrams
            elif n == 3:
                circuit.apply(T, i) # same as the P(π/4)-gate
            else:
                circuit.apply(P(np.pi/(2**(n-1))), i) 
                
def print_sv(sv):
    """Prints a numpy array as a readable column showing the coefficients of 
    each computational basis state."""
    # Number of qubits (length of word)
    n = int(np.log2(len(sv)))
    
    print("\n".join([f"{sv[k]:.2f}: |{k:0{n}b}>" for k in range(len(sv))]))

