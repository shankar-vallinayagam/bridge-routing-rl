# Bridge-Routing-RL

[![Status](https://img.shields.io/badge/status-active_development-ff69b4.svg)](https://github.com/YOUR_USERNAME/bridge-routing-rl)

Reinforcement learning for quantum circuit routing using arbitrary-length bridges.

## Overview

Bridge-Routing-RL is a reinforcement learning framework for compiling quantum circuits onto physical hardware, motivated by the paper [Arbitrary Bridge Lengths in Quantum Circuit Routing](https://link.springer.com/chapter/10.1007/978-3-032-13852-1_32). The agent learns to route qubits by inserting SWAP gates and arbitrary-length bridges, with the object of minimising CNOT count.

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).

## Results so far

On the 3-qubit run, with a 24-gate context window, benchmarking on up to 50-gate circuits against SABRE, the following performance was achieved:

```
qiskit_sabre_lightsabre: cases=125, mean_added_cnot=24.984, min=3, max=60
random_valid_router: cases=125, mean_added_cnot=39.408, min=6, max=102
trained_agent: cases=125, mean_added_cnot=21.432, min=3, max=51
```

## Training a serious 3-qubit run

Use the project virtual environment so the training and TensorBoard commands
do not fall back to the obsolete system Python 3.8 installation:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Start the configurable 1-million-timestep experiment:

```bash
.venv/bin/python experiments/train_3q.py
```

TensorBoard logs are written under `runs/`, and checkpoints are written under
`checkpoints/`. Monitor progress in another terminal with:

```bash
.venv/bin/tensorboard --logdir "/Users/shankarvallinayagamuser/PycharmProjects/qubit-routing-bridge-rl/runs"
```

For a quick validation run, reduce the workload without changing the script:

```bash
.venv/bin/python experiments/train_3q.py --total-timesteps 8192 --num-envs 2 --num-steps 64 --num-minibatches 2 --max-gates 8
```

## Topology-specific experiments and benchmarks

The reusable benchmark code lives under `benchmarking/`. Topology-specific
inputs and experiments live under `experiments/topologies/`.

Generate the reproducible random suite for the three-qubit line:

```bash
.venv/bin/python experiments/topologies/line_3/generate_suite.py
```

Train with outputs local to the topology experiment:

```bash
.venv/bin/python experiments/topologies/line_3/experiments/ppo_baseline/train.py --cpu
```

Run the trained agent against the random-valid baseline:

```bash
.venv/bin/python experiments/topologies/line_3/experiments/ppo_baseline/benchmark.py
```

The results are written under the experiment's `benchmark_results/` folder.

To add Qiskit's current SABRE implementation, install the optional benchmark
dependency:

```bash
.venv/bin/python -m pip install -r requirements-benchmark.txt
```

Then run all three methods:

```bash
.venv/bin/python experiments/topologies/line_3/experiments/ppo_baseline/benchmark.py --methods agent,random,sabre
```

For the deeper 10M-timestep experiment (10,000,384 effective steps with the
default batch size), first generate its held-out suite:

```bash
.venv/bin/python experiments/topologies/line_3/generate_suite.py \
  --per-depth 25 \
  --depths 10 20 30 40 50 \
  --output experiments/topologies/line_3/suites/deep_v1/cases.jsonl
```

Train it with topology-local outputs:

```bash
.venv/bin/python experiments/topologies/line_3/experiments/ppo_deep_10m/train.py --cpu
```

Track that experiment specifically with:

```bash
.venv/bin/tensorboard \
  --logdir /Users/shankarvallinayagamuser/PycharmProjects/qubit-routing-bridge-rl/experiments/topologies/line_3/experiments/ppo_deep_10m/runs
```

Benchmark its latest checkpoint:

```bash
.venv/bin/python experiments/topologies/line_3/experiments/ppo_deep_10m/benchmark.py \
  --suite experiments/topologies/line_3/suites/deep_v1/cases.jsonl \
  --methods agent,random,sabre
```
