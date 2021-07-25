"""
Helper functions and implementation of the Solovay-Kitaev algorithm, based on the descriptions in 
Dawson and Neilsen 2008. 

To use the functions, the following module/global variables must be first constructed using init():

REQUIRED GLOBAL VARIABLES
-------------------------
- L0: int
    The maximum length word which is enumerated when considering the base case approximation 
    to the recursion algorithm
- KNOWN_GATES:
    An array of 2x2 SU(2) matrices, which forms the group G of basis gates
- KNOWN_GATES_LETTERS:
    An array of characters, corresponding to the string representations of KNOWN_GATES.
    Uses the convention that adjoint is lowercase, e.g. 'T'^dagger == 't'
- N_GATES: int
    The number of gates in KNOWN_GATES, equal to `len(KNOWN_GATES)`
- N_GATES_WORD: int
    The number of gate permutations for each word length, equal to `[N_GATES ** x for x in range(1, l0 + 1)]`
"""

import numpy as np
import qiskit as qt
import pickle
import re

from numpy import linalg as la
from scipy import linalg as spla
from functools import reduce

from sklearn.neighbors import KDTree

from Circuits import *
from Gates import *
import Functions as fn
import Approximations as ap

# ----------------------------- #
# Table of Contents             #
# ----------------------------- #
# - Initialisation              #
# - Linear Algebra              #
# - Blanced Group Commutator    #
# - Word-Matrix Representations #
# - Trees                       # 
# - Solovay-Kitaev Algorithm    #
# ----------------------------- #

# -------------- #
# Initialisation #
# -------------- #

def init(l0, known_gates, known_gates_letters, do_return = False):
    """Initialise the required global variables for the functions in this module.
    
    Arguments
    ---------
    l0: int
        The maximum length word which is enumerated when considering the base case approximation 
        to the recursion algorithm
    known_gates:
        An array of 2x2 SU(2) matrices, which forms the group G of basis gates
    known_gates_letters:
        A numpy array of characters, corresponding to the string representations of KNOWN_GATES.
        Uses the convention that adjoint is lowercase, e.g. 'T'^dagger == 't'
    do_return: bool, optional
        If True, returns N_GATES and N_GATES_WORD (default False)
    """
    global L0
    L0 = l0
    
    global KNOWN_GATES
    KNOWN_GATES = known_gates
    
    global KNOWN_GATES_LETTERS
    KNOWN_GATES_LETTERS = known_gates_letters
    
    global N_GATES
    N_GATES = len(KNOWN_GATES_LETTERS)
    
    global N_GATES_WORD
    N_GATES_WORD = [N_GATES ** x for x in range(1, l0 + 1)]
    
    if do_return:
        return N_GATES, N_GATES_WORD

# -------------- #
# Linear Algebra #
# -------------- #
    
def eigen(matrix, return_values = False):
    """Compute the eigenvectors and eigenvaues for a unitary matrix, using the Schur decomposition, 
    and return them ordered by:
            Re(lambda) + Im(lambda)
    for each eigenvalue, lambda.
    
    Arguments
    ---------
    matrix:
        The matrix on which to calculate the eigenvalue decomposition
    return_values: bool, optional
        If True, return both the eigenvalues and eigenvectors. Otherwise, only return the 
        eigenvectors (default False)
    
    Returns
    -------
    eigen_val:
        Numpy array of eigenvalues (only if return_values == True)
    eigen_vec:
        Numpy array of eigenvectors as columns
    """
    # Calculate the eigenvalues and eigenvectors
    # using the Schur decomposition, since this guarantees
    # unitary matrices, unlike np.linalg.eig()
    upper_tri, eigen_vec = spla.schur(matrix)
    eigen_val = np.diagonal(upper_tri)
    
    # Sort the eigenvalues and eigenvectors by (Re + Im)
    # to ensure that similar matrices have the same 
    # ordering of eigenvectors/eigenvalues
    idx = np.argsort(eigen_val.imag + eigen_val.real)
    eigen_val = eigen_val[idx]
    eigen_vec = eigen_vec[:,idx]
    
    # Return the eigenvalues and eigenvectors
    if return_values:
        return eigen_val, eigen_vec
    return eigen_vec    

def to_su2(unitary):
    """Convert a unitary matrix into a matrix in SU(2), i.e. with determinant = 1"""
    return unitary / np.sqrt(la.det(unitary))

def group_com(mat_A, mat_B):
    """Compute the group commutator A B A^dagger B^dagger for two matrices A, B."""
    return mat_A @ mat_B @ mat_A.T.conj() @ mat_B.T.conj()

# ------------------------- #
# Balanced Group Commutator #
# ------------------------- #

def get_angles(unitary, use_sin = True):
    """Compute the rotation angles of a unitary matrix, and the equivalent rotation when
    decomposed as a balance group commutator.
    
    Arguments
    ---------
    unitary:
        An SU(2) matrix to be used for calculating the angles
    use_sin: bool, optional (debugging)
        If True, sine is used for computing the angles. If False, then cosine
    
    Returns
    -------
    theta: float
        The rotation angle of the unitary matrix
    phi: float
        The equivalent rotation angle when decomposed
    """
    # By trial and error, the sine function gives the correct angle
    func = np.sin if use_sin else np.cos
    # Calculate the rotation angle theta, and the corresponding angle phi
    # as per Dawson and Nielsen
    theta = 2 * np.arccos(min(unitary[0][0].real, 1))
    phi = 2 * np.arcsin(np.sqrt(func(theta/4)))
    return theta, phi

def get_bgc(unitary,  debug = False):
    """Compute the balanced group commutator decomposition of a unitary matrix, as per Dawson and Nielson (2008).
    
    Arguments
    ---------
    unitary:
        The SU(2) matrix which is to be decomposed
    debug: bool, optional
        If True, performs checks to see if the computed decomposition satisfies given criteria (default False)
    
    Returns
    -------
    v, w:
        Two matrices which satisfy 
                            unitary = [V, W] = V W V^dagger W^dagger
        where V, W are composed of rotations about the x and y axes respectively.
    """
    # Conver the matrix to SU(2)
    unitary = to_su2(unitary)
    # Calculate the angles as per Dawson and Nielsen
    theta, phi = get_angles(unitary)
    # Compute the corresponding rotation matrices, V and W
    v = ROTX(phi).matrix
    w = ROTY(phi).matrix
    # Use the eigenvectors from the Schur decomposition to 
    # calculate the similarity matrix S, such that Unitary = S V W V^dagger W^dagger S^dagger
    u_eigen = eigen(unitary)
    gc_eigen = eigen(group_com(v, w))
    s = u_eigen @ gc_eigen.T.conj()
    
    # Debug cases:
    if debug:
        if dist_square(s @ s.T.conj() - I, np.zeros((2,2))) > 1e-4: raise Exception("S is not unitary")
        if dist_square(s @ group_com(v, w) @ s.T.conj(), unitary) > 1e-4: raise Exception("S is not correct transformation")
    
    # Return the matrices V^tilde and W^tilde
    return s @ v @ s.T.conj(), s @ w @ s.T.conj()

# -------------------------- #
# Word-Matrix Represenations #
# -------------------------- #

def word_to_mat(word: str):
    """Convert a word (e.g. 'HtTs') into a 2x2 matrix by applying each gate via left-multiplication.
    For example, 'HTS' would become S @ T @ H."""
    mat = I
    for letter in word:
        [[index]] = np.argwhere(KNOWN_GATES_LETTERS == letter)
        mat = KNOWN_GATES[index] @ mat
    return mat

def word_adj(word: str) -> str:
    """Convert a word into the adjoint representation by swapping the case (except for Hermitian gates)
    and reversing the order."""
    return "".join(letter.swapcase() if letter.lower() in KNOWN_GATES_LETTERS else letter for letter in word)[::-1]

def simplify_word(word: str) -> str:
    """Simplify a word by using identity and commutation relations."""
    patterns = [
        #(pattern, replacement),
        ('(HH)|(Tt)|(tT)|(sS)|(Ss)|(ZZ)|(XX)|(YY)', ''),
        ('HZH', 'X'),
        ('HXH', 'Z'),
        ('tt', 's'),
        ('TT', 'S'),
        ('(SS)|(ss)', 'Z'),
        ('(Zs)|(sZ)', 'S'),
        ('(ZS)|(SZ)', 's'),
        ('(sT)|(Ts)', 't'),
        ('(St)|(tS)', 'T'),
        ('st', 'ts'),
        ('ST', 'TS'),
        ('Zs', 'sZ'),
        ('ZS', 'SZ'),
        ('Zt', 'tZ'),
        ('ZT', 'TZ')
    ]
    # Replace the patterns in the word until no more patterns 
    # remain, and the word is unchanged after a loop
    while True:
        old_word = word
        for (pattern, repl) in patterns:
            word = re.sub(pattern, repl, word)
        if old_word == word:
            break            
    return word

def word_gen(max_length: int, out = "str"):
    """Generator function for iterating through all the strings of length less than
    or equal to max_length.
    
    Arguments
    ---------
    max_length: int
        The desired maximum length strength to iterate up to
    known_gates:
        The array containing the letters for the desired gates
    out: 'str' or 'mat'
        The desired output. If 'str', outputs the string representation; if 'mat', 
        outputs the matrix representation, formed via left-multiplication. 
        For example, string 'HTS' is equal to S @ T @ H as a matrix. 
    """
    # Iterate through all word lengths from 1,2,...,max_length
    for word_length in range(1, max_length + 1):
        # Iterate through the num_gates ^ word_length states for this length
        for state in range(N_GATES ** word_length):
            # Convert the state into the base n_gates representation
            # and pad to the desired word_length
            quint_string = np.base_repr(state, base = N_GATES)
            quint_string = "0" * (word_length - len(quint_string)) + quint_string
            
            # Return the desired representation
            if out == "str":
                # Yield the string representation, where we map 0 -> H, 1 -> T, etc.
                yield "".join(KNOWN_GATES_LETTERS[int(x)] for x in quint_string)
            elif out == "mat":
                # Yield the matrix representation, left-multiplying each gate
                yield reduce(lambda mat, next_mat: next_mat @ mat, (KNOWN_GATES[int(index)] for index in quint_string), I)
            else:
                raise Exception(f"Invalid out specification, {out}.")

def index_to_word(index: int) -> str:
    """Turn an index into the corresponding word string. This is effectively the inverse function
    of word_gen().
    
    Arguments
    ---------
    index: int
        The index corresponding to the word
    """
    # Find the length of the word by considering the first element of N_GATES_WORD
    # that is larger than the index given
    word_length = next((i + 1 for i, x in enumerate(np.cumsum(N_GATES_WORD)) if x > index), 1)
    # The "state" corresponds to the index of the word within the group of words of the 
    # same length
    state = index - sum(N_GATES_WORD[:(word_length - 1)])
    # Convert the state into a base N_GATES string and pad to the correct length
    quint_string = np.base_repr(state, base = N_GATES)
    quint_string = "0" * (word_length - len(quint_string)) + quint_string
    # Return the gate representation, by mapping 0 -> H, 1 -> T, etc. 
    return "".join(KNOWN_GATES_LETTERS[int(x)] for x in quint_string)

# ------------- #
# Create a Tree #
# ------------- #

def encode(mat):
    """Encode a 2x2 unitary matrix as a 1x8 vector, consiting of the real and imaginary 
    parts of each entry."""
    flat_mat = np.array(mat.flat)
    return np.concatenate((flat_mat.real, flat_mat.imag))

def create_tree(init_tree = False):
    """Enumerate all possible words of length less than L0, and store them in an array as a flattened matrix.
    
    Arguments
    ---------
    init_tree: bool, optional
        If True, also create the k-d tree object and return it
        
    Returns
    -------
    mat_list:
        List of flattened matrices
    tree (optional):
        k-d tree object
    """
    # Initialise the encoded list of matrices
    # with width 8, corresponding to the 4 real + 4 imaginary entries
    mat_list = np.zeros((sum(N_GATES_WORD), 8))
    # Iterate through the words as per the word generator
    for index, word in enumerate(word_gen(L0)):
        # Convert the word into a matrix and append it to the list
        mat = word_to_mat(word)
        mat_list[index] = encode(mat)
    
    if init_tree:
        return mat_list, KDTree(mat_list)
    return mat_list

def save_tree(tree, file):
    with open(file, "wb") as f:
        pickle.dump(tree, f, protocol = pickle.HIGHEST_PROTOCOL)

def load_tree(file = 'tree.pickle'):
    with open(file, 'rb') as f:
        tree = pickle.load(f)
    return tree

# ------------------------ #
# Solovay-Kitaev Algorithm #
# ------------------------ #

def sk_recurse(unitary, n, tree):
    """Recursive implementation of the Solovay-Kitaev algorithm.
    
    Arguments
    ---------
    unitary: 2x2 matrix
        The matrix in SU(2) which is being approximated
    n: int
        The current depth of the search
    tree:
        The k-d tree which is used to find the base approximations
        
    Returns
    -------
    approx:
        The matrix representation of the approximation
    word: str
        The word containing the gates required to construct the approximation
    """
    # Base case
    # compute the approximation using the pre-processed set of gates
    if n == 0:
        [[index]] = tree.query(encode(unitary).reshape(1, -1), return_distance = False)
        word = index_to_word(index)
        return word_to_mat(word), word
    else:
        # Compute the n-1 approximation to U
        prev_approx, prev_word = sk_recurse(unitary, n - 1, tree)
        
        # Compute the balanced-group-commutator representation of Delta = U U_(n-1)^dagger
        v, w = get_bgc(unitary @ prev_approx.T.conj())
        
        # Then get epsilon_n-1 approximations to the resulting v, w matrices 
        prev_v, v_word = sk_recurse(v, n - 1, tree)
        prev_w, w_word = sk_recurse(w, n - 1, tree)
        
        # Construct the final approximations and word sequences
        approx = prev_v @ prev_w @ prev_v.T.conj() @ prev_w.T.conj() @ prev_approx
        word = prev_word + word_adj(w_word) + word_adj(v_word) + w_word + v_word
        
        return approx, word
    
def solovay_kitaev(unitary, n, file = 'tree.pickle', tree = None, *args, **kwargs):
    """Compute the Solovay-Kitaev approximation to a unitary matrix, at a desired level epsilon-n.
    
    If the unitary matrix is not in SU(2), then the approximation will hold up to a global phase.
    
    Arguments
    ---------
    unitary:
        A 2x2 unitary matrix (not necessarily in SU(2)) to be approximated.
    n: int
        The desired level of epsilon-n 
    file: str, optional
        The file where the tree is stored (default 'tree.pickle')
    tree: optional
        The tree object. If None, then the file is loaded instead (default None)
        
    Returns
    -------
    approx:
        The 2x2 matrix representation of the approximation
    word:
        The word representation the gate sequence to construct the approximation
    """
    if tree is None:
        tree = load_tree(file)
    unitary = to_su2(unitary)
    approx, word = sk_recurse(unitary, n, tree)
    return approx, simplify_word(word)

def sk_wrapper(n, file = 'tree.pickle', tree = None, dist_fun = dist_abs, *args, **kwargs):
    """Wrapper for function solovay_kitaev, returning a function which takes as input
    a unitary matrix, and computes the solovay-kitaev approximation, before returning
    the approx, distance, sequence - in line with the other approximation algorithms.
    
    Arguments
    ---------
    n: int
        The desired level of epsilon-n; depth of recursion
    dist_fun: 
        Function which computes the distance between two matrices
    *args, **kwargs:
        Optional arguments passed on to solovay_kitaev()
        
    Returns
    -------
    approx_fun:
        A function which takes as input a 2x2 matrix, and returns:
        - approx:
            The 2x2 matrix approximation:
        - dist:
            The distance between approx and the input matrix:
        - sequence:
            List of letters corresponding to the the simplified word representation  
    """
    def approx_fun(gate, dist_fun = dist_fun, *args, **kwargs):
        gate = to_su2(gate)
        approx, word = solovay_kitaev(gate, n, file, tree, *args, **kwargs)
        dist = dist_fun(approx, gate)
        return approx, dist, list(word)
    return approx_fun