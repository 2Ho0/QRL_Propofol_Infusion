# Quantum Hardware Deployment Guide

## Overview

This guide explains how to deploy the Quantum Propofol Infusion Agent on real quantum hardware (IBM Quantum, AWS Braket, IonQ).

## Why Hybrid Architecture (Quantum Actor + Classical Critic)?

### Design Rationale

1. **Quantum Advantage for Policy Learning**:
   - VQC with 2 qubits explores exponentially large Hilbert spaces
   - Parameter efficiency: ~16-32 quantum parameters vs 100+ classical parameters
   - Natural exploration through quantum superposition

2. **Classical Stability for Value Estimation**:
   - Critic requires frequent updates with large batches
   - Classical NNs are more stable for regression tasks
   - Avoids 2x quantum overhead from Twin Critic architecture

3. **Cost Efficiency**:
   - **Quantum Actor only**: $10k-$70k training cost (AWS)
   - **Full Quantum**: $400k-$2M training cost
   - **Savings**: ~83% reduction in quantum executions

## Hardware-Optimized Agent Features

### Key Optimizations

- ✅ **Reduced Circuit Depth**: Automatically adjusts layers to fit NISQ constraints
- ✅ **Error Mitigation**: Built-in ZNE and readout error correction
- ✅ **Cost Tracking**: Real-time monitoring of quantum execution costs
- ✅ **Multi-Provider Support**: IBM Quantum, AWS Braket, IonQ

### Hardware Constraints (2024-2025)

| Provider | Max Circuit Depth | Gate Error Rate | Cost per Execution | Recommended Use |
|----------|------------------|-----------------|-------------------|-----------------|
| IBM Quantum | ~100 gates | 0.1-0.5% | $1.60 | Research (premium access) |
| AWS Braket (IonQ) | ~200 gates | 0.1-0.3% | $0.35 | Production (best value) |
| Rigetti | ~50 gates | 0.5-2% | Variable | Experimental |
| Simulator | Unlimited | 0% | $0 | Development/Testing |

## Setup Instructions

### 1. Install Dependencies

```bash
# Core dependencies (already installed)
pip install pennylane torch numpy

# For IBM Quantum
pip install qiskit-ibm-runtime

# For AWS Braket
pip install amazon-braket-pennylane-plugin boto3
```

### 2. Configure Credentials

#### IBM Quantum

1. Sign up at [IBM Quantum](https://quantum-computing.ibm.com/)
2. Get your API token
3. Save credentials:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save token (one time only)
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
```

#### AWS Braket

1. Create AWS account
2. Configure AWS CLI:

```bash
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1)
```

3. Verify access:

```bash
aws braket get-device --device-arn arn:aws:braket:us-east-1::device/qpu/ionq/Harmony
```

### 3. Usage Examples

#### Example 1: Simulator (Testing)

```python
from src.agents.quantum_agent import HardwareOptimizedQuantumAgent

agent = HardwareOptimizedQuantumAgent(
    state_dim=8,
    action_dim=1,
    hardware_provider='simulator',
    max_circuit_depth=30
)

# Train as usual
state = env.reset()
action = agent.select_action(state)
```

#### Example 2: IBM Quantum

```python
agent = HardwareOptimizedQuantumAgent(
    state_dim=8,
    action_dim=1,
    hardware_provider='ibm',
    backend_name='ibmq_manila',  # or None for least busy backend
    use_error_mitigation=True,
    max_circuit_depth=30,
    shots=1000
)

print(agent.get_hardware_info())
# {
#   'provider': 'ibm',
#   'backend': 'ibmq_manila',
#   'connected': True,
#   'max_circuit_depth': 30,
#   'total_executions': 0,
#   'estimated_cost_usd': '$0.00'
# }
```

#### Example 3: AWS Braket (IonQ)

```python
agent = HardwareOptimizedQuantumAgent(
    state_dim=8,
    action_dim=1,
    hardware_provider='aws',
    backend_name='arn:aws:braket:us-east-1::device/qpu/ionq/Harmony',
    use_error_mitigation=True,
    shots=1000
)

# Monitor costs during training
for episode in range(100):
    state = env.reset()
    for step in range(200):
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        
    # Check costs every episode
    hw_info = agent.get_hardware_info()
    print(f"Episode {episode}: {hw_info['estimated_cost_usd']}")
```

## Cost Optimization Strategies

### 1. Reduce Training Steps

```python
# Standard training: 200,000 steps → $70,000
# Optimized: 50,000 steps → $17,500

config = {
    'training': {
        'total_steps': 50000,  # Reduced from 200000
        'eval_frequency': 1000  # Evaluate more frequently
    }
}
```

### 2. Use Pre-trained Weights

```python
# Train on simulator first
agent_sim = QuantumDDPGAgent(...)
# ... train on simulator ...
agent_sim.save("pretrained_simulator.pt")

# Fine-tune on real hardware
agent_hw = HardwareOptimizedQuantumAgent(
    hardware_provider='aws',
    ...
)
agent_hw.load("pretrained_simulator.pt")
# Fine-tune for only 10,000 steps → $3,500
```

### 3. Batch Circuit Execution (Future)

```python
# Currently in development
agent = HardwareOptimizedQuantumAgent(
    batch_quantum_execution=True,  # Experimental
    batch_size=64
)
# Could reduce costs by 50-75%
```

## Training Cost Estimates

### Full Training (200,000 steps)

| Scenario | Provider | Estimated Cost | Time | Recommended |
|----------|----------|---------------|------|-------------|
| Development | Simulator | $0 | 4-6 hours | ✅ Yes |
| Pre-training | Simulator | $0 | 4-6 hours | ✅ Yes |
| Fine-tuning | AWS Braket | $10k-$17k | 2-3 days | ✅ Recommended |
| Full Training | AWS Braket | $70k | 10-15 days | ⚠️ Expensive |
| Full Training | IBM Quantum | $320k | 15-20 days | ❌ Not recommended |
| With Quantum Critic | AWS/IBM | $400k-$2M | 30+ days | ❌ Impractical |

### Recommended Workflow

```
1. Simulator Training (free)
   ↓
2. Simulator Validation ($0, 1-2 days)
   ↓
3. AWS Braket Fine-tuning ($10k-$20k, 2-3 days)
   ↓
4. Production Deployment
```

## Performance Monitoring

### Track Hardware Statistics

```python
# During training
agent.get_hardware_info()
# Returns:
# {
#     'provider': 'aws',
#     'backend': 'IonQ Harmony',
#     'connected': True,
#     'error_mitigation': True,
#     'max_circuit_depth': 30,
#     'shots': 1000,
#     'total_executions': 5000,
#     'estimated_cost_usd': '$1750.00'
# }

# Reset cost tracking between experiments
agent.reset_cost_tracking()
```

### Save Hardware Configuration

```python
# Saves both model and hardware config
agent.save("checkpoints/agent_aws_episode_100.pt")

# Files created:
# - agent_aws_episode_100.pt          (model weights)
# - agent_aws_episode_100_hardware.pt (hardware config)
```

## Troubleshooting

### Common Issues

#### 1. Connection Errors

```python
# Problem: Cannot connect to IBM Quantum
# Solution: Verify credentials
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
print(service.backends())  # List available backends
```

#### 2. Queue Wait Times

```python
# Problem: Long queue times on popular backends
# Solution: Use least busy backend
agent = HardwareOptimizedQuantumAgent(
    hardware_provider='ibm',
    backend_name=None,  # Auto-select least busy
    ...
)
```

#### 3. Circuit Depth Exceeded

```python
# Problem: RuntimeError: Circuit depth exceeds hardware limit
# Solution: Reduce n_layers in config
config = {
    'quantum': {
        'n_layers': 2,  # Reduced from 4
        ...
    }
}
```

#### 4. Cost Overruns

```python
# Problem: Costs exceeding budget
# Solution: Set execution limits
MAX_COST = 5000  # $5000 budget

if agent.estimated_cost >= MAX_COST:
    print("Budget exceeded! Stopping training.")
    agent.save("checkpoint_budget_limit.pt")
    break
```

## Hardware-Specific Recommendations

### IBM Quantum

- ✅ Best for: Research, academic projects
- ✅ Pros: High-quality qubits, good error rates
- ❌ Cons: Expensive, long queue times
- 💡 Tip: Use free tier (10 min/month) for testing

### AWS Braket (IonQ)

- ✅ Best for: Production, cost-sensitive projects
- ✅ Pros: Good value, reliable access
- ❌ Cons: Still expensive for full training
- 💡 Tip: Use spot pricing when available

### Rigetti

- ✅ Best for: Experimental, gate-level optimization
- ❌ Cons: Limited depth, higher error rates
- 💡 Tip: Only for shallow circuits (n_layers=1-2)

## Future Enhancements

### Planned Features

1. **Hybrid Training**: Alternate between simulator and hardware
2. **Adaptive Circuit Depth**: Dynamically adjust based on hardware availability
3. **Cost Prediction**: Estimate total training cost before starting
4. **Error Mitigation**: Advanced techniques (ZNE, PEC, CDR)
5. **Batch Execution**: Reduce overhead by batching quantum circuits

## References

- [IBM Quantum Documentation](https://docs.quantum.ibm.com/)
- [AWS Braket Documentation](https://docs.aws.amazon.com/braket/)
- [PennyLane Hardware Integration](https://pennylane.ai/qml/demos.html)
- [NISQ Algorithm Design](https://arxiv.org/abs/1801.00862)

## Support

For issues or questions:
1. Check [GitHub Issues](https://github.com/2Ho0/QRL_Propofol_Infusion/issues)
2. Review PennyLane documentation
3. Contact hardware provider support

---

**Last Updated**: December 21, 2025
**Version**: 1.0.0
