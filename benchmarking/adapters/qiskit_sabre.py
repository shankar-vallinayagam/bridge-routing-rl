import time

from benchmarking.metrics import circuit_depth, count_cx
from benchmarking.results import RoutingResult


class QiskitSabreAdapter:
    """Qiskit's current SABRE implementation, including LightSABRE changes."""

    method_name = "qiskit_sabre_lightsabre"

    def __init__(self, topology):
        try:
            from qiskit import QuantumCircuit, transpile
        except ImportError as exc:
            raise ImportError(
                "Qiskit is required for the SABRE benchmark. "
                "Install requirements-benchmark.txt first."
            ) from exc
        self.QuantumCircuit = QuantumCircuit
        self.transpile = transpile
        self.topology = topology

    def route(self, case, seed=None) -> RoutingResult:
        circuit = self.QuantumCircuit(case.qubit_count)
        for name, (a, b) in case.gates:
            if name == "cx":
                circuit.cx(a, b)
        start = time.perf_counter()
        try:
            routed = self.transpile(
                circuit,
                coupling_map=[list(edge) for edge in self.topology.edges],
                routing_method="sabre",
                optimization_level=0,
                seed_transpiler=seed,
                basis_gates=["u", "cx"],
            )
            elapsed = time.perf_counter() - start
            routed_count = int(routed.count_ops().get("cx", 0))
            return RoutingResult(
                case_id=case.case_id,
                method=self.method_name,
                success=True,
                original_cnot_count=case.original_cnot_count,
                routed_cnot_count=routed_count,
                added_cnot_count=routed_count - case.original_cnot_count,
                swap_count=int(routed.count_ops().get("swap", 0)),
                bridge_count=0,
                route_time_seconds=elapsed,
                depth=routed.depth(),
                metadata={"qiskit_gate_counts": dict(routed.count_ops())},
            )
        except Exception as exc:
            return RoutingResult(
                case_id=case.case_id,
                method=self.method_name,
                success=False,
                original_cnot_count=case.original_cnot_count,
                route_time_seconds=time.perf_counter() - start,
                error=repr(exc),
            )
