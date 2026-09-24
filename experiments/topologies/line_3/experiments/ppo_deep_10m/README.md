# Deep 10M PPO experiment

This experiment uses the shared `line_3` training and benchmark harness with:

- at least 10,000,000 PPO timesteps (10,000,384 with the default 1,024-step batch);
- training circuits containing 10–50 CNOTs;
- a 24-gate context window;
- topology-local checkpoints, TensorBoard logs, and benchmark results.

The evaluation suite is `../../suites/deep_v1/cases.jsonl`.
