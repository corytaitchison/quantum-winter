"""
Module containing:

- Circuit class and subclasses
"""
import numpy as np
import qiskit as qt
from abc import ABC, abstractmethod

from Gates import *

class Circuit(ABC):
    """Abstract Base Class for constructing Qiskit or Matrix circuits of a given number of qubits.
    
    Child class
    -----------
    QiskitCircuit:
        For creating a circuit using Qiskit
    MatrixCircuit:
        For creating a circuit using numpy matrices
    """
    def __init__(self, n):
        self.n = n
        self.dim = 2 ** n
        
    @abstractmethod
    def apply(self, gate, *qubits):
        pass
    
    @abstractmethod
    def add_gate(self, gate, *qubits):
        pass
    
    @abstractmethod
    def get_matrix(self):
        pass
    
    def swap(self, qubit1, qubit2, keep_others = True):
        """Swap the position of two qubits, which are not necessarily adjacent.
        
        Optionally keeps other qubits in their correct position (set this
        to false if the qubits will be swapped back immediately after).
       
        First moves qubit1 to the position of qubit2. If keep_others is set to True, 
        qubit2 will then be moved to the position of qubit1, returning the other qubits
        to their original positions.
        
        Arguments
        ---------
        qubit1: int
            Index of the first qubit; indexed from 0
        qubit2: int
            Index of the second qubit
        keep_others: bool, optional
            Whether to retain the positions of the other qubits (default True)
        """
        # Sign of the swap, used in ensuring the correct movement
        s = np.sign(qubit2 - qubit1)
        
        # Swap by moving qubit1 towards the other, 
        # one qubit at a time.
        for current in range(qubit1, qubit2, s):
            self.apply(SWAP, current, current + s)

        # Swap the remaining qubits back to their original position
        # so only two qubits overall are swapped.
        if keep_others:
            for current in range(qubit2 - s, qubit1, -s):
                self.apply(SWAP, current, current - s)
                
    def apply(self, gate, *qubits):
        """Apply a one or two-qubit gate to the circuit.
        
        If the qubits are not adjacent, first apply swap operations to bring them together,
        then apply the gate, and then swap back the qubits. 
        
        Arguments
        ---------
        gate: Gate
            A Gate object representing the unitary gate to apply to the circuit
        *qubits: int
            One or two positional arguments specifying the qubits to apply to the circuit.
            In the case of a controlled gate, the first qubit is the control, the second is the target.
            Qubits are indexed from 0. 
        """
        if gate.num_qubits == 2:
            qubit1, qubit2 = qubits
            
            # If not adjacent, swap the qubits so that they are adjacent
            # then apply the gate, then swap back
            if np.abs(qubit1 - qubit2) != 1:
                s = np.sign(qubit2 - qubit1)
                qubit3 = qubit2 - s

                self.swap(qubit1, qubit3, keep_others = False)
                self.add_gate(gate, qubit3, qubit2)
                self.swap(qubit3, qubit1, keep_others = False)
            else:
                self.add_gate(gate, qubit1, qubit2)
        else:
            # If only a single qubit gate, just apply it 
            self.add_gate(gate, *qubits)
                
class QiskitCircuit(Circuit):
    """QiskitCircuit class used for creating circuits using qiskit. 
    
    This is a child of the Circuit abstract class.
    
    Arguments
    ---------
    n: int
        The number of qubits in the circuit
        
    Attributes
    ----------
    circuit: QuantumCircuit
        The qiskit quantum circuit object, on which the gates are constructed.
        Can be printed to the notebook using circuit.draw().
    dim: int
        The dimension of the state space for the circuit; equal to 2^n
    
    Methods
    -------
    get_matrix():
        Returns the matrix representation of the circuit.
    add_gate(gate, *qubits):
        Private class used to add a gate to the qiskit circuit. Do not call this directly;
        instead, use apply()
    apply(gate, *qubits):
        Public class to apply a one or two-qubit gate to the qiskit circuit. 
    swap(qubit1, qubit2, keepOthers = True):
        Swap the position of two qubits, which are not necessarily adjacent.
    """
    def __init__(self, n_qubit, n_clbit = None):
        """Construct the QiskitCircuit object.
       
       Arguments
        ---------
        n_qubit: int
            The number of qubits in the circuit
        n_clbit: int, optional
            The number of classical bits in the circuit (default None)
        """
        Circuit.__init__(self, n_qubit)
        if n_clbit is None or n_clbit == 0:
            self.circuit = qt.QuantumCircuit(n_qubit)
        else:
            self.circuit = qt.QuantumCircuit(n_qubit, n_clbit)
        
    def __repr__(self):
        print(self.circuit)
        return ""
    
    def get_matrix(self):
        return qt.quantum_info.Operator(self.circuit).reverse_qargs().data
        
    def add_gate(self, gate, *qubits):
        """Add a one or two-qubit gate to the circuit.
        
        Arguments
        ---------
        gate: Gate
            A Gate object representing the unitary matrix to add to the circuit
        *qubits: int
            Positional list of qubits to add to the circuit. Indexed from 0. 
        """
        gate.qiskit(self.circuit, *qubits)
        
class MatrixCircuit(Circuit):
    """MatrixCircuit class used for creating circuits using numpy matrices. 
    
    This is a child of the Circuit abstract class.
    
    Arguments
    ---------
    n: int
        The number of qubits in the circuit
        
    Attributes
    ----------
    circuit: matrix
        The numpy 2^n x 2^n matrix representation of the circuit.
    dim: int
        The dimension of the state space for the circuit; equal to 2^n
    
    Methods
    -------
    get_matrix():
        Returns the matrix representation of the circuit.
    add_gate(gate, *qubits):
        Private class used to add a gate to the matrix. Do not call this directly;
        instead, use apply()
    apply(gate, *qubits):
        Public class to apply a one or two-qubit gate to the circuit.  
    swap(qubit1, qubit2, keepOthers = True):
        Swap the position of two qubits, which are not necessarily adjacent.
    """
    def __init__(self, n):
        """Construct the MatrixCircuit object.
       
       Arguments
        ---------
        n: int
            The number of qubits in the circuit
        """
        Circuit.__init__(self, n)
        self.circuit = np.identity(self.dim, dtype = "complex")
        
    def __repr__(self):
        return repr(self.circuit)
    
    def get_matrix(self):
        return self.circuit
        
    def add_gate(self, gate, *qubits):
        """Add a one or two-qubit gate to the circuit. If two-qubits, then assumes that the 
        two qubits are adjacent; otherwise an Exception is raised.
        
        Arguments
        ---------
        gate: Gate
            A Gate object representing the unitary matrix to add to the circuit
        *qubits: int
            Positional list of qubits to add to the circuit. Indexed from 0. 
        """
        # Find the smallest qubit, which is where we "anchor" the matrix
        qubit = np.min(qubits)
        
        # Pad the matrix with identities for the unaffected qubits
        matrix = np.identity(2 ** qubit, dtype = "complex")
        # If two qubits, and qubit2 < qubit1, then reverse the ordering of the gate matrix
        matrix = np.kron(matrix, gate.matrix if qubit == qubits[0] else gate.reverse_matrix())
        # Pad the remaining unaffected qubits
        matrix = np.kron(matrix, np.identity(2 ** (self.n - qubit - gate.num_qubits)))
        
        # Apply the matrix to the circuit
        self.circuit = matrix @ self.circuit