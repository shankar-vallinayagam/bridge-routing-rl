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
    
    def __init__(self, circuit: GateSequence, window_length: int):
        # the working_circuit is treated as mutable, the original circuit
        # is saved to allow resets during training
        self.original_circuit = copy.deepcopy(circuit)
        self.working_circuit = circuit

        self.done = False
        self.Q = self.original_circuit.architecture.qubit_count
        self.E = self.original_circuit.architecture.edge_count

        # Observation space is just list of CNOT gates, we will update the GateSequence object and then query
        # it for the new observation space
        self.observation_space = gym.spaces.dict({"context_window": gym.spaces.Box(
            low=0,
            high=self.Q,
            shape=(self.window_len, 2),
            dtype=np.int32,
        ),
        "interaction_matrix": gym.spaces.Box(
            low=0,
            # unlikely to be more than 100 million interactions
            high=int(1e7),
            shape=(self.Q, self.Q),
            dtype=np.int32,
        )})

        self.last_observation = None


        # See _get_action to see actions; first Q are layout phase actions, next E are swaps, and the last is
        # turning the next gate into a BRIDGE
        self.action_space = gym.spaces.Discrete(self.Q+self.E+1)
        # layout phase is the starting few steps where we can free of charge
        # allocate logical qubits to physical ones
        self.layout_phase = True
        self.layout_count = 0
        self.mapping  = np.full(self.Q, self.Q, dtype=np.int32)

        self.index = 0
        self.window_length = window_length
        self.cnot_count = 0
    
    def reset(self, *, seed = None):
        '''resets back to original state'''
        super().reset(seed=seed)
        self.working_circuit = self.original_circuit
        self.done = False
        self.layout_phase = True
        self.layout_count = 0


        # info is debugging info, may add later
        observation = self._get_obs()
        info = self._get_info

        return observation, info

    def step(self, action):
        action_type, action_info = self._get_action(action)
        if action_type == "LAYOUT":
            self.mapping[self.layout_count] = action_info
            observation = self._get_observation(unchanged=True)
            info = self.get_info()
            return observation, 0, False, False, info
        elif action_type == "SWAP":
            a, b = action_info
            self.working_circuit.insert_swap(self.index, a, b)
            self.index += 3
            self.cnot_count += 3
            steps, gates_compiled = self.working_circuit.attempt_compile(self.index)
            self.index += steps
            observation = self._get_observation()
            info = self._get_info()
            if self.index == len(self.working_circuit):
                self.done = True
                self.working_circuit.hardware_mapping(self.mapping)
            return observation, gates_compiled-3, (self.index == len(self.working_circuit)), False, info
        elif self.action_space == "BRIDGE":
            steps = self.working_circuit.convert_bridge(self.index)
            self.index += self.steps
            observation = self._get_observation()
            info = self._get_info
            if self.index == len(self.working_circuit):
                self.done = True
                self.working_circuit.hardware_mapping(self.mapping)
            return observation, 1-steps, (self.index == len(self.working_circuit)), False, info

            
    def _get_observation(self, unchanged=False):
        '''TODO: Gives the observation for the agent to consider'''
        if not unchanged and (self.last_observation is not None):
            self.last_observation = {"context_window": self.working_circuit.context_window(self.index, self.window_length),
                "interaction_matrix": self.working_circuit.interaction_mat}
        return self.last_observation

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
            return ["BRIDGE", None]
        
    def _get_info(self):
        return {"added CNOTs": self.cnot_count}
    
    def get_action_mask(self):
        if self.layout_phase:
            return [i<self.Q for i in range(self.Q+self.E+1)]
        else:
            return [i >= self.Q for i in range(self.Q+self.E+1)]
        


    





        