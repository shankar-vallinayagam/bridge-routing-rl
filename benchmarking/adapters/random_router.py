import random
import time

from benchmarking.adapters.trained_agent import _topology_from_hardware
from benchmarking.metrics import circuit_depth, count_cx, validate_physical_circuit
from benchmarking.results import RoutingResult
from env.environment import CircuitEnvironment
from env.quantum_circuit import GateSequence


class RandomRouterAdapter:
    method_name = "random_valid_router"

    def __init__(self, hardware, window_length=12, max_decisions=10000):
        self.hardware = hardware
        self.window_length = window_length
        self.max_decisions = max_decisions

    def route(self, case, seed=None) -> RoutingResult:
        rng = random.Random(seed)
        env = CircuitEnvironment(self.hardware, self.window_length, 1, max(1, len(case.gates)))
        circuit = GateSequence(self.hardware, case.as_circuit_list())
        observation, _ = env.reset(seed=seed, options={"circuit": circuit})
        start = time.perf_counter()
        decisions = 0
        bridge_count = 0
        try:
            while not env.done and decisions < self.max_decisions:
                if env.layout_phase:
                    used = set(int(q) for q in env.mapping if q < env.Q)
                    choices = [q for q in range(env.Q) if q not in used]
                    action = rng.choice(choices)
                else:
                    a, b = env.working_circuit.circuit[env.index][1]
                    incident_edges = [
                        self.hardware.Q + i
                        for i, (u, v) in enumerate(self.hardware.edges)
                        if a in (u, v) or b in (u, v)
                    ]
                    bridge_action = self.hardware.Q + self.hardware.E
                    choices = incident_edges
                    if b not in self.hardware.adj_list[a]:
                        choices = choices + [bridge_action]
                    action = rng.choice(choices)
                    bridge_count += action == bridge_action
                observation, _, terminated, truncated, _ = env.step(action)
                decisions += 1
                if terminated or truncated:
                    break
            final_circuit = env.working_circuit.circuit
            validate_physical_circuit(final_circuit, _topology_from_hardware(self.hardware))
            routed_count = count_cx(final_circuit)
            return RoutingResult(
                case_id=case.case_id,
                method=self.method_name,
                success=env.done,
                original_cnot_count=case.original_cnot_count,
                routed_cnot_count=routed_count,
                added_cnot_count=routed_count - case.original_cnot_count,
                swap_count=env.cnot_count // 3,
                bridge_count=bridge_count,
                route_time_seconds=time.perf_counter() - start,
                decision_count=decisions,
                depth=circuit_depth(final_circuit, case.qubit_count),
            )
        except Exception as exc:
            return RoutingResult(
                case_id=case.case_id,
                method=self.method_name,
                success=False,
                original_cnot_count=case.original_cnot_count,
                route_time_seconds=time.perf_counter() - start,
                decision_count=decisions,
                error=repr(exc),
            )
