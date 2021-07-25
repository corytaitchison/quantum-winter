"""
Module containing:

- Distance metrics
- Gate class and subclasses
- Common gate definitions
- GADGETS dictionary
"""

import qiskit as qt
import numpy as np
from sympy import flatten

# Absolute distance metric
dist_abs = lambda matrix_a, matrix_b: np.sum(np.abs(matrix_a - matrix_b))

# Frobenius norm - sum of squares of absolute values
dist_square = lambda matrix_a, matrix_b: np.sqrt(np.sum(np.square(np.abs(matrix_a - matrix_b))))
dist_frob = lambda matrix_a, matrix_b: np.sum(np.square(np.abs(matrix_a - matrix_b)))

# Square-root absolute distance metric
dist_sqrt = lambda matrix_a, matrix_b: np.square(np.sum(np.sqrt(np.abs(matrix_a - matrix_b))))

class Gate:
    """Gate class used for representing and applying unitary gates to a matrix or qiskit circuit.
    
    Attributes
    ----------
    matrix:
        Numpy matrix representation of the unitary operation 
    qiskit: 
        Function that accepts two arguments: a circuit object, and qubit index. Applies the gate to the 
        circuit at the given index
    num_qubits:
        The number of qubits on which this gate acts. Currently only one and two-qubit gates are supported.
    
    Methods
    -------
    reverse_matrix():
        Reverses the order of the qubits for the matrix and returns it. 
    """
    def __init__(self, matrix, qiskit, num_qubits):
        """
        Arguments
        ---------
        matrix:
            Numpy matrix representation of the unitary operation 
        qiskit: 
            Function that accepts two arguments: a circuit object, and qubit index. Applies the gate to the 
            circuit at the given index
        num_qubits:
            The number of qubits on which this gate acts. Currently only one and two-qubit gates are supported.
        """
        self.matrix = matrix
        self.qiskit = qiskit
        if num_qubits != 1 and num_qubits != 2: raise Exception("Only supports 1 and 2 qubit gates.")
        self.num_qubits = num_qubits
    
    def __repr__(self):
        return repr(self.matrix)
    
    def reverse_matrix(self):
        """Reverses the order of the qubits for the matrix and returns it."""
        return SWAP.matrix @ self.matrix @ SWAP.matrix
    
class ControlGate(Gate):
    """ControlGate class used to create a controlled-version of a single-qubit gate. A subclass of the Gate class.
    
    Arguments
    ---------
    gate: Gate
        The single qubit unitary gate from which the controlled gate is constructed.
    decompose: optional
        If None, use the actual rotation gates. If set to a function, use that function to compute approximation
        to each gate, using Hadamard and T gates (default None).
    *args, **kwargs:
        Optional arguments to pass to the DecomposedGate() function.
    
    Attributes
    ----------
    matrix
        The 4x4 numpy matrix representation of the controlled unitary operation 
    qiskit
        Function which accepts two arguments: a circuit object, and qubit indices. Applies the controlled-
        gate to the circuit at the given indices. 
    num_qubits: int
        The number of qubits on which this gate acts (2)
        
    Methods
    -------
    __control_unitary(unitary):
        Calculate and return the coefficients and matrices used to construct the controlled gate operation.
        Private class, used in the initialisation of the object.
    """
    def __init__(self, gate: Gate, decompose = None, *args, **kwargs):
        """
        Arguments
        ---------
        gate: Gate
            The single qubit unitary gate from which the controlled gate is constructed.
        """
        [alpha, beta, gamma, delta], mat_A, mat_B, mat_C = self.__control_unitary(gate.matrix)
        
        self.matrix = np.identity(4, dtype = "complex")
        self.matrix[2:4, 2:4] = gate.matrix
        
        def qiskit_func(circuit, qubit_control, qubit_target):
            """Function used to apply the control-gate to a qiskit circuit."""
            def apply_step(step, circuit, qubit):
                if decompose is not None:
                    step = DecomposedGate(step, decompose, *args, **kwargs)
                step.qiskit(circuit, qubit)

            if (delta - beta)/2 != 0:
                step = ROTZ((delta - beta)/2)
                apply_step(step, circuit, qubit_target)
            circuit.cx(qubit_control, qubit_target)
            if (delta + beta)/2 != 0:
                step = ROTZ(-(delta + beta)/2)
                apply_step(step, circuit, qubit_target)
            if gamma != 0:
                step = ROTY(-gamma/2)
                apply_step(step, circuit, qubit_target)
            circuit.cx(qubit_control, qubit_target)
            if gamma != 0:
                step = ROTY(gamma/2)
                apply_step(step, circuit, qubit_target)
            if beta != 0:
                step = ROTZ(beta)
                apply_step(step, circuit, qubit_target)
            if alpha != 0:
                step = P(alpha) 
                apply_step(step, circuit, qubit_control)

        self.qiskit = qiskit_func
        self.num_qubits = 2
            
    
    def __control_unitary(self, unitary):
        """Constructs the decomposition that allows for an arbitrary controlled-unitary gate to be
        expressed using CNOTs and single-qubit gates.

        Arguments
        ---------
        unitary:
            The 2x2 unitary matrix corresponding to the single qubit gate

        Returns
        -------
        coefs:
            The four terms: alpha, beta, gamma, delta
        mat_A, mat_B, mat_C:
            The four matrices, used as per the equations above
        """
        # Unpack the matrix elementwise, corresponding to the matrix
        # [ a,  b ]
        # [ c, d ]
        [a, b], [c, d] = unitary

        # Solve for alpha, beta, gamma, delta as per the above equations
        alpha = np.log(a*d - b*c) / 2j
        gamma = np.arccos((a*d - b*c) / (a*d + b*c))

        # If there are zeroes then we have multiple solutions.
        # In that case, set beta to 0 then solve for delta
        if c*d == 0 and a*b*np.sin(gamma) == 0:
            beta = 0
            delta = np.log(d / (a * np.cos(gamma/2)**2)) / 1j
        else:
            beta = np.log(-4*c*d / (a*b * (np.sin(gamma)**2))) / 2j
            delta = np.log(-b*d/(a*c*(np.tan(gamma/2)**2))) / 2j

        # Construct the matrices as per above equations
        mat_A = ROTZ(beta).matrix @ ROTY(gamma/2).matrix
        mat_B = ROTY(-gamma/2).matrix @ ROTZ(-(delta + beta)/2).matrix
        mat_C = ROTZ((delta - beta)/2).matrix
        mat_0 = np.array(
            [
                [1, 0],
                [0, np.exp(1j * alpha)]
            ]
        )        
        return np.real([alpha, beta, gamma, delta]), mat_A, mat_B, mat_C

class DecomposedGate(Gate):
    """Class which constructs a decomposed one-qubit gate in terms of H and T gates, using a chosen algorithm.
    
    Arguments
    ---------
    gate: Gate
        The Gate which is being approximated
    algorithm: 
        The function which computes the approximation and returns approximation, distance, sequence 
    delta: float, optional
        The delta argument for the algorithm (default 1e-1)
    dist_fun: optional
        Function which accepts two matrices and computes the distance between them (default dist_square)
    *args, **kwargs:
        Additional arguments which are passed to the algorithm function
        
    Attributes
    ----------
    matrix
        The 2x2 numpy matrix representation of the controlled unitary operation 
    qiskit
        Function which accepts two arguments: a circuit object, and qubit indices. Applies the controlled-
        gate to the circuit at the given indices. 
    num_qubits: int
        The number of qubits on which this gate acts (1)
    sequence:
        A list of 'H' and 'T' indicating the approximation used for the gate
    distance:
        The distance between the approximation and the original gate, as per the chosen dist_fun metric
    """
    def __init__(self, gate: Gate, algorithm, delta = 1e-1, dist_fun = dist_square, *args, **kwargs):
        self.num_qubits = gate.num_qubits
        approximation, distance, sequence = algorithm(gate.matrix, delta = delta, dist_fun = dist_fun, *args, **kwargs)
        self.matrix = approximation
        self.sequence = flatten(sequence)
        self.distance = distance
        
        def qiskit_func(circuit, qubit):
            # Apply Hadamard and T gates as per the approximation sequence
            for gate in list("".join(flatten(sequence))):
                if gate == "H":
                    func = circuit.h
                elif gate == "T":
                    func = circuit.t
                elif gate == "S":
                    func = circuit.s
                elif gate == "X":
                    func = circuit.x
                elif gate == "Y":
                    func = circuit.y
                elif gate == "Z":
                    func = circuit.z
                elif gate == "t":
                    func = circuit.tdg
                elif gate == "s":
                    func = circuit.sdg                    
                else:
                    raise Exception(f"Unrecognised gate: {gate}")
                func(qubit)

        self.qiskit = qiskit_func
        
        
# Hadamard gate
H = Gate(
    np.array(
        [
            [1, 1],
            [1, -1 + 0j]
        ]
    ) / np.sqrt(2),
    qt.QuantumCircuit.h,
    1
)

# 2x2 Identity
I = np.identity(2, dtype = "complex")

X = Gate(
    np.array(
        [
            [0, 1],
            [1, 0j]
        ]
    ),
    qt.QuantumCircuit.x,
    1
)

Y = Gate(
    np.array(
        [
            [0, -1j],
            [1j, 0]
        ]
    ),
    qt.QuantumCircuit.y,
    1
)

Z = Gate(
    np.array(
        [
            [1, 0j],
            [0, -1]
        ]
    ),
    qt.QuantumCircuit.z,
    1
)

S = Gate(
    np.array(
        [
            [1, 0],
            [0, np.exp(1j * np.pi / 2)]
        ]
    ),
    qt.QuantumCircuit.s,
    1
)

T = Gate(
    np.array(
        [
            [1, 0],
            [0, np.exp(1j * np.pi / 4)]
        ]
    ),
    qt.QuantumCircuit.t,
    1
)

P = lambda phi: Gate(
    np.array(
        [
            [1, 0],
            [0, np.exp(1j * phi)]
        ]
    ),
    lambda circuit, qubit: circuit.p(phi, qubit),
    1
)

# Controlled not gate, with second qubit the target 
CTRL_X = Gate(
    np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0j]
        ]
    ),
    qt.QuantumCircuit.cx,
    2
)

SWAP = Gate(
    np.array(
        [
            [1, 0, 0, 0j],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]
    ),
    qt.QuantumCircuit.swap,
    2
)

ROTZ = lambda phi: Gate(
    np.array(
        [
            [np.exp(-1j * phi / 2), 0],
            [0, np.exp(1j * phi / 2)]
        ]
    ),
    lambda circuit, qubit: circuit.rz(phi, qubit),
    1
)

ROTY = lambda theta: Gate(
    np.array(
        [
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ]
    ),
    lambda circuit, qubit: circuit.ry(theta, qubit),
    1
)

ROTX = lambda theta: Gate(
    np.array(
        [
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ]
    ),
    lambda circuit, qubit: circuit.rx(theta, qubit),
    1
)

R = lambda k: Gate(
    mat := np.array(
        [
            [1, 0j],
            [0, np.exp(2j*np.pi / (2 ** (k)))]
        ]
    ),
    lambda circuit, qubit: circuit.unitary(mat, qubit, label = f"R({k})"),
    1    
)

# Gadgets - Matrices that we want to use to approximate other unitary matrices
GADGETS = {
    'H': H.matrix,
    'T': T.matrix,
    'HTH': H.matrix @ T.matrix @ H.matrix,
    'THT': T.matrix @ H.matrix @ T.matrix
}