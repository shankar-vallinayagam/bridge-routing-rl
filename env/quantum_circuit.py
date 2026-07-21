from env.chip_architecture import ChipHardware
import numpy as np


class GateSequence:
    '''A lightweight implementation of a quantum circuit
    
    Represents cx gates as lists of length 2 ["cx", [control, object]]
    Represents 1 qubit unitaries as lists of length 2 ["name", qubit]
    
    Parameters
    ----------
    architecture: ChipHardware
        Gives topology of chip
    circuit: list
        List of Gates
    interaction_mat: list[list]
        matrix encoding how often 2 qubits interact in circuit
    '''
    def __init__(self, architecture: ChipHardware, circuit: list):
        self.architecture = architecture
        self.circuit = circuit
        self.interaction_mat = self._get_interaction_mat()

    def _get_interaction_mat(self):
        matrix = np.zeros((self.architecture.qubit_count, self.architecture.qubit_count))
        for i in self.circuit:
            if i[0] == "cx":
                matrix[i[1][0], i[1][1]] += 1
                matrix[i[1][1], i[1][0]] += 1
        return matrix
    
    def to_qiskit(self):
        '''TODO:Converts to qiskit circuit'''

    def hardware_mapping(self, mappings):
        '''Before any gates are run, we can simply permute our choice of qubits for free.
        Mappings is a list where the index gives the position of the logical qubit and
        the value gives the index in hardware it is mapped to. This is always done at
        the end, so we don't bother with changing the interaction matrix'''
        original = self.circuit.copy()
        for a, b in enumerate(mappings):
            for i in range(len(original)):
                if original[i][0] == "cx":
                    if original[i][1][0] == a:
                        self.circuit[i][1][0] = b
                    elif original[i][1][0] == b:
                        self.circuit[i][1][0] = a
                else:
                    if original[i][1] == a:
                        self.circuit[i][1] = b
                    elif original[i][1] == b:
                        self.circuit[i][1] = a

    def check_valid(self, index):
        '''Returns whether the gate at the given index can take place on the given hardware'''
        if self.circuit[index][0] == "cx":
            a, b = self.circuit[index][1]
            return b in self.architecture.adj_list[a]
        else:
            return True
        
    def attempt_compile(self, index):
        i = index
        successfully_compiled = 0
        while i < len(self.circuit) and self.check_valid(i):
            if self.circuit[i][0] == "cx":
                successfully_compiled += 1
                a, b = self.circuit[i][1]
                self.interaction_mat[a][b] -= 1
                self.interaction_mat[b][a] -= 1
            i += 1
        return i, successfully_compiled
 
    def insert_swap(self, index, a, b):
        '''Inserts a swap gate where index i is (gate previously at index i goes to i+3). 
        Returns the number 3 to be added to the additional CNOT count. If a,b not adjacent
        raises error'''
        if b in self.architecture.adj_list[a]:
            self.circuit.insert(index, ["cx", [a, b]])
            self.circuit.insert(index, ["cx", [b, a]])
            self.circuit.insert(index, ["cx", [a, b]])
            self.interaction_mat[[a, b], :] = self.interaction_mat[[b, a], :]
            self.interaction_mat[:, [a, b]] = self.interaction_mat[:, [b, a]]
            return 3
        else:
            raise ValueError("Qubits not adjacent, cannot swap")

    def convert_bridge(self, index):
        '''Converts the CNOT between 2 points into a BRIDGE gate between them, see
        https://link.springer.com/chapter/10.1007/978-3-032-13852-1_32 for details.
        returns gates added so index can be changed accordingly'''
        a, b = self.circuit[index][1]
        # delete the existing CNOT at our index
        del self.circuit[index]
        # get the shortest path between these qubits
        path = self.architecture.get_path(a, b)
        for j in range(2):
            for i in range(1, len(path)-1):
                self.circuit.insert(index, ["cx", [path[i], path[i+1]]])
            for i in range(len(path)-1, 1, -1):
                self.circuit.insert(index, ["cx", [path[i-1], path[i]]])
        return 4*self.architecture.distances[a][b] - 4

    def context_window(self, index, window_length):
        '''Provides the next window_length cx gates. Will be mapped into an embedding
        before passed into NN. [Q, Q] means "nothing to do here, padding" '''
        Q = self.architecture.qubit_count
        i = index
        window = []
        while len(window) < window_length and i < len(self.circuit):
            if self.circuit[i][0] == "cx":
                window.append(self.circuit[i][1])
            i += 1
        window += [[Q, Q]]*window_length-len(window)
        return window, i
    
        
