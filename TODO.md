Plan:

Implement BFS etc in chip_architecture.
Test basic operations

add function to quantum_circuit that gives permitted actions
add function that gives reward and changes state given action; this should also
compile i.e. skip past 1 qubit unitaries and compile completed CNOTs. Preliminary rewards
-3 for SWAP, -CNOT count for BRIDGE

Then use PPO algo to train agent

Architecure; IBM-Q Heron 133q
Training phase:
    Random Circuits
    OpenQASM: all circuits that fit with depth below 100 CNOT
Testing Phase:
    OpenQASM; all circuits that fit with depth above certain amount (100 CNOT)
