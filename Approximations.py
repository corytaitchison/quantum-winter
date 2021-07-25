"""
Module containing:

- Algorithms (functions) for approximating a given unitary single-qubit gate
"""
import random
from functools import reduce
import numpy as np

from Gates import *

def unitary_greedy(unitary, delta = np.pi / 4, dist_fun = dist_abs, do_print = False):
    """Greedily search for a valid approximation to a given unitary matrix, by taking the optimal
    choice at each addition of a new gate (gadget).
    
    Arguments
    ---------
    unitary:
        The 2x2 unitary matrix to be approximated
    delta: float, optional
        The desired level of accruacy (default pi/4)
    dist: function, optional
        The function to be used to calculated the difference between two matrices.
        Should take two arguments (two matrices) and output a positive number.
    """
    # The current approximation matrix
    approximation = I
    # Calculated distance between the unitary matrix and its approximation.
    distance = dist_fun(unitary, approximation)
    # The list of gates which have been applied to create the approximation matrix.
    # Approximation is conducted via left-multiplying, but the sequence is 
    # constructed via right-appending, so e.g. TTH would have sequence ["H", "T", "T"]. 
    sequence = []
    
    while distance > delta:
        # While the distance is below the given threshold, iterate through each gadget
        # and choose the one which minimises the distance.
        choice = None
        for gadget in GADGETS:
            gadget_distance = dist_fun(unitary, GADGETS[gadget] @ approximation)
            if gadget_distance < distance:
                choice = gadget
                distance = gadget_distance

        # If no gadget is found which decreases the distance further, then break the loop
        # early (failure case). 
        # If a suitable decrease is made, then apply this gate to the approximation,
        # and loop again. 
        if choice is None:
            if do_print: print("Breaking early, no improvement observed.")
            break
        else:
            sequence.append(choice)
            approximation = GADGETS[choice] @ approximation
    
    return approximation, distance, sequence
        
def unitary_brute(unitary, delta = np.pi / 4, max_depth = 6, dist_fun = dist_abs, approximation = I, depth = 1, sequence = []):
    """Recursively brute-force search for the optimal approximation to a given unitary matrix.
    
    Gates are appended via left-multiplication. For example, H -> TH -> TTH. 
    
    Arguments
    ---------
    unitary
        The 2x2 unitary matrix to be approximated
    delta: float, optional
        The desired level of accruacy (default pi/4)
    max_depth: int, optional
        The maximum number of recursive steps to be taken (default 6)
    dist: function, optional
        The function to be used to calculated the difference between two matrices.
        Should take two arguments (two matrices) and output a positive number. 
    
    Recursion Arguments
    -------------------
    approximation: matrix
        The current value of the approximation matrix
    depth: int
        The current recursion depth; depth = 1 means the entry level
    sequence: 
        List of strings identifying the matrices which make up the current approximation    
        
    Returns
    -------
    [approximation, distance, sequence]: 
        The final approximation matrix, the final distance, and the list of gates applied, if a suitable
        approximation is found.         
    """
    # Base case: failed to find an accurate approximation before hitting the recursion limit
    if depth > max_depth:
        return I, dist_fun(unitary, I), []
    for gadget in GADGETS:
        # Iterate through each gate H, T, HTH and THT
        # Calculate the distance using the given distance measure
        gadget_approx = GADGETS[gadget] @ approximation
        gadget_distance = dist_fun(unitary, gadget_approx)
        gadget_sequence = sequence + [gadget]
        
        # If the distance is below the threshold, return the given approximation and pass it up the 
        # recursion.
        # Otherwise, enter another layer of recursion to append another gate to the approximation
        if gadget_distance < delta:
            return gadget_approx, gadget_distance, gadget_sequence
        else:
            result = unitary_brute(unitary, delta = delta, max_depth = max_depth, 
                                   dist_fun = dist_fun, approximation = gadget_approx, 
                                   depth = depth + 1, sequence = gadget_sequence)

            # If an empty list is received, continue the loop to test another gadget at this same 
            # recursion level.
            # Otherwise, if a valid approximation is found, pass it up the recursion. 
            if result[2] == []:
                continue
            else:
                return result
    
    # If none of the gadgets return a valid approximation, ascend to the previous recursion level
    return I, dist_fun(unitary, I), []

def unitary_random(unitary, delta = 0.1, repeats = 1000, dist_fun = dist_abs, seed = None, initial_len = None, do_print = False):
    """Approximates a given unitary matrix using a combination of H, T, HTH, THT, and their respective adjoint matrices. 
    
    Starts the search with a random combination of the given gates.
    
    Arguments
    ---------
    unitary:
        The 2x2 unitary matrix to be approximated
    delta: float, optional
        The distance between the matrices upon which the search will terminate (default 0.1)
    repeats: int, optional
        The number of times to repeat the search using different initial combinations (default 1000)
    dist: function, optional
        The function to be used to calculated the difference between two matrices.
        Should take two arguments (two matrices) and output a positive number.
    seed: int or None, optional
        The seed for the random number generator (default None)
    initial_len: int or None, optional
        The number of matrices in the initial approximation. If None, then the number is randomly
        chosen from 1,...,10 inclusive (default None)
    do_print: bool, optional
        Whether to print out the distance after each iteration (default False)
        
    Returns
    -------
    best_approximation:
        The 2x2 matrix which best approximates the unitary among all the repeats
    best_distance:
        The distance between the approximation and the unitary matrices, as given by 
        the chosen distance function. 
    best_sequence:
        The sequence of matrices which combine to make up the given approximation.
        The sequence is read right-to-left: [H, T] means the approximation is TH 
        as matrix multiplication. 
    """
    if seed is not None: 
        random.seed(seed)
        
    # Initialise variables
    best_sequence = []
    best_approximation = []
    # Distance is infinite, so that any approximation will be taken to be better
    best_distance = np.inf
        
    for i in range(repeats):
        
        # If no initial length is provieded, randomly choose how many matrices to
        # include in the initial sequence
        if initial_len is None:
            k = random.randint(1, 10)
        else:
            k = initial_len
        sequence = random.choices(list(GADGETS.keys()), k = k)
        
        # Calculate the initial approximation by left-multiplying each
        # matrix in the sample 
        approximation = reduce(lambda x, y: x @ GADGETS[y], sequence, I)
        
        # Calculate the initial distance 
        distance = dist_fun(unitary, approximation)
        
        # Loop through each gadget combination until either the distance is
        # smaller than the threshold, or no improvement is made by including
        # any further matrices
        while distance > delta:
            # The current best matrix to include
            choice = None

            # Iterate through each gadget and see whether the distance would decrease
            for gadget in GADGETS:
                # Normal matrices
                gadget_distance = dist_fun(unitary, GADGETS[gadget] @ approximation)
                if gadget_distance < distance:
                    choice = gadget
                    distance = gadget_distance
            
            # If no gadget will decrease the distance further, break the loop.
            # Otherwise, add the best gadget to the sequence, and repeat
            if choice is None:
                if do_print: print(f"Loop {i}: Breaking early, no improvement observed.")
                break
            else:
                sequence.append(choice)
                approximation = GADGETS[choice] @ approximation
                
        if do_print: print(f"Loop {i}: distance {distance}")
        
        # Once a particular initial approximation has been exhausted, see if the final approximation
        # is better than the previous loops. If so, store it. Then repeat with the next
        # random initial approximation. 
        if distance < best_distance:
            best_distance = distance
            best_sequence = sequence
            best_approximation = approximation
        
    return best_approximation, best_distance, best_sequence
    
def unitary_lookahead(unitary, delta = 1e-1, dist_fun = dist_abs, search_depth = 3, do_print = False):
    """Approximates a given single-qubit unitary gate by considering the best matrix to add
    at each level, looking ahead at (default 3) combinations."""
    def recurse(approximation, sequence = [], depth = 1, best_distance = np.inf, 
                best_approx = I, best_sequence = []):
        if depth > search_depth:
            return best_approx, best_distance, best_sequence
        for gadget in GADGETS:
            # Iterate through each gate H, T, HTH and THT
            # Calculate the distance using the given distance measure
            gadget_approx = GADGETS[gadget] @ approximation
            gadget_distance = dist_fun(unitary, gadget_approx)
            gadget_sequence = sequence + [gadget]

            # If the distance is better than the current best approximation, make it the best
            if gadget_distance < best_distance:
                best_approx, best_distance, best_sequence = gadget_approx, gadget_distance, gadget_sequence
            
            # Continue with recursion until search depth is reached
            best_approx, best_distance, best_sequence = recurse(gadget_approx, sequence = gadget_sequence,
                                                               depth = depth + 1, best_distance = best_distance,
                                                               best_approx = best_approx, best_sequence = best_sequence)
        # Once all gadgets have been recursed through up to a given depth,
        # return the best approximation
        return best_approx, best_distance, best_sequence
        
    approximation = I
    distance = dist_fun(approximation, unitary)
    sequence = []
    
    while distance > delta:
        # Calculate the best gate to add at this current level, looking ahead by search_depth matrices
        approximation, new_distance, new_sequence = recurse(approximation, [], best_distance = distance,
                                                   best_approx = approximation, best_sequence = [])      
        sequence = sequence + [new_sequence]
        # If no improvement is found, stop the algorithm
        if new_distance > distance - 1e-8:
            if do_print: print("Breaking loop, no further improvement found")
            break
        distance = new_distance
    
    return approximation, distance, sequence        

def unitary_identity(unitary, delta = 1e-1, dist_fun = dist_abs):
    """Returns the identity matrix."""
    return I, dist_fun(I, unitary), []