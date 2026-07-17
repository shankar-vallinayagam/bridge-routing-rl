class ChipHardware:
    """Physical topology of a quantum processor.
    
    Represents qubits as nodes and physical couplings as edges.
    Provides shortest-path queries between qubits. 
    Treat as immutable.
    
    Parameters
    ----------
    qubit_count : int
        Number of physical qubits on the device.
    adj_list : list of list of int
        Adjacency list where ``adj_list[i]`` contains the qubit
        indices directly connected to qubit ``i``.
    """

    def __init__(self, qubit_count: int, adj_list: list[list[int]]):
        self.qubit_count = qubit_count
        self.adj_list = adj_list

        self.distances, self.parent = self._compute_shortest_paths()
        self.edge_count = sum(len(neighbors) for neighbors in self.adj_list) // 2
        self.edges = self._compute_edge_list()


    def _compute_shortest_paths(self):
        '''TODO: Uses BFS from all points to spit out shortest paths'''
        distances = []
        parent = []
        return distances, parent
    
    def _compute_edge_list(self):
        edge_list = []
        for i in range(len(self.adj_list)):
            for j in self.adj_list[i]:
                if [j, i] not in edge_list:
                    edge_list.append([i,j])
        return edge_list

    
    def get_path(self, a, b):
        """TODO: Gets shortest path from a to b"""
    
