import numpy as np
from collections import deque

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
        self.edges= self._compute_edge_list()


    def _compute_shortest_paths(self):
        '''Uses BFS from all points to get shortest distances and parent pointers for path reconstruction
        
        distances[i][j] = hop distance from i to j (-1 if unreachable)
        parent[i][j] = parent node of j on shortest path from i (-1 if no such path), can be used to reconstruct path
        '''
        Q = self.qubit_count
        distances = np.full((Q,Q), -1, dtype=np.int64)
        parent = [[None]*Q for node in range(Q)]

        for src in range(Q):
            # Run BFS from src through whole graph
            distances[src, src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in self.adj_list[u]:
                    if distances[src, v] == -1:
                        distances[src, v] = distances[src, u] + 1
                        parent[src, v] = u
                        queue.append(v)
        
        return distances, parent
    
    def _compute_edge_list(self):
        edge_list = []
        edge_dict = []
        for i in range(len(self.adj_list)):
            edge_dict.append([])
            for j in self.adj_list[i]:
                if [j, i] not in edge_list:
                    edge_list.append([i,j])
                edge_dict[i].append(j)
        return edge_list

    
    def get_path(self, a, b):
        """Gets shortest path from a to b. If no such exists, return None"""
        if self.distances[a, b] is None:
            return None

        path = [b]
        while path[-1] != a:
            path.append(self.parent[a, path[-1]])
        path.reverse()
        return path
    
