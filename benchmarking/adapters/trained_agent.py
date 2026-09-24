import pickle
import time
from pathlib import Path

import torch

from benchmarking.metrics import circuit_depth, count_cx, validate_physical_circuit
from benchmarking.results import RoutingResult
from env.environment import CircuitEnvironment
from env.quantum_circuit import GateSequence
from rl.ppo import Agent


class TrainedAgentAdapter:
    method_name = "trained_agent"

    def __init__(self, checkpoint: Path, device: str = "cpu", max_decisions: int = 10000):
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        self.max_decisions = max_decisions
        with self.checkpoint.with_name(self.checkpoint.stem + "_config.pkl").open("rb") as handle:
            config = pickle.load(handle)
        self.agent_config = config["agent_config"]
        self.env_config = config["env_config"]
        self.agent = Agent(self.agent_config, self.env_config).to(self.device)
        self.agent.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.agent.eval()

    def route(self, case, seed=None) -> RoutingResult:
        env = CircuitEnvironment(
            architecture=self.env_config.hardware,
            window_length=self.env_config.window_length,
            min_gate_count=self.env_config.min_gate_count,
            max_gate_count=self.env_config.max_gate_count,
        )
        circuit = GateSequence(self.env_config.hardware, case.as_circuit_list())
        observation, _ = env.reset(seed=seed, options={"circuit": circuit})
        inference_seconds = 0.0
        environment_seconds = 0.0
        bridge_count = 0
        decisions = 0
        start = time.perf_counter()
        try:
            while not env.done and decisions < self.max_decisions:
                batched = {
                    key: torch.as_tensor(value, device=self.device).unsqueeze(0)
                    for key, value in observation.items()
                }
                inference_start = time.perf_counter()
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(
                        batched, deterministic=True
                    )
                inference_seconds += time.perf_counter() - inference_start
                action_value = int(action.item())
                if action_value == self.env_config.hardware.Q + self.env_config.hardware.E:
                    bridge_count += 1
                transition_start = time.perf_counter()
                observation, _, terminated, truncated, _ = env.step(action_value)
                environment_seconds += time.perf_counter() - transition_start
                decisions += 1
                if terminated or truncated:
                    break

            success = env.done
            final_circuit = env.working_circuit.circuit
            validate_physical_circuit(final_circuit, _topology_from_hardware(self.env_config.hardware))
            routed_count = count_cx(final_circuit)
            return RoutingResult(
                case_id=case.case_id,
                method=self.method_name,
                success=success,
                original_cnot_count=case.original_cnot_count,
                routed_cnot_count=routed_count,
                added_cnot_count=routed_count - case.original_cnot_count,
                swap_count=env.cnot_count // 3,
                bridge_count=bridge_count,
                route_time_seconds=time.perf_counter() - start,
                inference_time_seconds=inference_seconds,
                environment_time_seconds=environment_seconds,
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
                inference_time_seconds=inference_seconds,
                environment_time_seconds=environment_seconds,
                decision_count=decisions,
                error=repr(exc),
            )


def _topology_from_hardware(hardware):
    class _Topology:
        adjacency = hardware.adj_list

    return _Topology()
