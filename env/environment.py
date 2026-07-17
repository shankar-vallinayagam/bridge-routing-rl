from env.chip_architecture import ChipHardware
from env.quantum_circuit import GateSequence
import copy
from typing import Optional
import gymnasium as gym
import numpy as np



class CircuitEnvironment(gym.Env):
    '''Reinforcement Learning Environment.
    
    Start off with an immutable copy of the original circuit to allow easy resets.
    Working circuit of course mutable.'''
    
    def __init__(self, circuit: GateSequence):
        # the working_circuit is treated as mutable, the original circuit
        # is saved to allow resets during training
        self.original_circuit = copy.deepcopy(circuit)
        self.working_circuit = circuit

        self.done = False
        self.Q = self.original_circuit.architecture.qubit_count
        self.E = self.original_circuit.architecture.edge_count

        # Observation space is just list of CNOT gates, we will update the GateSequence object and then query
        # it for the new observation space
        self.observation_space = gym.spaces.Sequence(self.spaces.Box(0, self.Q-1, shape=(2,), dtype=np.int32))

        # See _get_action to see actions; first Q are layout phase actions, next E are swaps, and the last is
        # turning the next gate into a BRIDGE
        self.action_space = gym.spaces.Discrete(self.Q+self.E+1)
        # layout phase is the starting few steps where we can free of charge
        # allocate logical qubits to physical ones
        self.layout_phase = True
        self.layout_count = 0
    
    def reset(self, *, seed = None):
        '''resets back to original state'''
        self.working_circuit = self.original_circuit
        self.done = False
        self.layout_phase = True
        self.layout_count = 0
        return super().reset(seed=seed)

    def step(self, action):
        return super().step(action)

    def _get_action(self, n):
        '''Given a number n it finds the action corresponding to this
        
        For the first Q this is the layout phase action of mapping the
        current layout_count logical qubit to the hardware qubit at the
        given number. For the next E this puts a swap in that specific
        edge. The last 1 it just converts the gate in front into a
        bridge'''
        edges = self.original_circuit.architecture.edges
        if n < self.Q:
            return ["LAYOUT", n]
        elif n < self.Q + self.E:
            return ["SWAP", edges[n-self.Q]]
        else:
            return ["BRIDGE"]
    
    def get_action_mask(self):
        if self.layout_phase:
            return [i<self.Q for i in range(self.Q+self.E+1)]
        else:
            return [i >= self.Q for i in range(self.Q+self.E+1)]
        


    





        