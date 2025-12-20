# Quantum Reinforcement Learning for Propofol Infusion Control

A hybrid Quantum-Classical Reinforcement Learning system for closed-loop BIS-guided propofol anesthesia control, based on the CBIM (Closed-loop BIS-guided Infusion Model) paper with Quantum enhancement using PennyLane.

## 🎯 Overview

This project implements **Quantum Deep Deterministic Policy Gradient (QDDPG)** and **Quantum Proximal Policy Optimization (QPPO)** agents for automated propofol infusion control during anesthesia. The system uses a 2-qubit Variational Quantum Circuit (VQC) as the policy network to determine optimal propofol dosing to maintain the patient's BIS (Bispectral Index) at the target level.

### Key Features

- **Dual Algorithm Support**: Both DDPG and PPO with VQC-based policy (Formulations 41-49)
- **Quantum Policy Network**: 2-qubit VQC with angle encoding and variational layers
- **Temporal Encoders**: LSTM and Transformer for sequential state processing (Fig.4)
- **Dual Drug Support**: Propofol + Remifentanil interaction model
- **Schnider PK/PD Model**: Three-compartment pharmacokinetic model with state-space form (Formulations 1-17)
- **Minto Model**: Remifentanil pharmacokinetics (Formulations 18-29)
- **Drug Interaction BIS Model**: Combined propofol-remifentanil effect (Formulation 32)
- **Clinical Metrics**: MDPE, MDAPE, Wobble evaluation (Formulations 50-52)
- **Gymnasium Environment**: Standard RL interface with extended 8-dimensional state

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Quantum RL Agent (QDDPG / QPPO)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Temporal Encoder (Optional) - Fig.4              │   │
│  │  ┌─────────────┐  or  ┌─────────────────────────────┐    │   │
│  │  │    LSTM     │      │      Transformer            │    │   │
│  │  │ Bidirectional│      │  Multi-Head Attention      │    │   │
│  │  └─────────────┘      └─────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────┐     ┌─────────────────────────────────┐       │
│  │   State      │     │     Quantum Policy (Actor)      │       │
│  │   Encoder    │──▶ │  ┌───────────────────────────┐   │       │
│  │  (Classical) │     │  │    2-Qubit VQC            │  │       │
│  └──────────────┘     │  │  • Angle Encoding         │  │       │
│                       │  │  • RY-RZ Rotations        │  │──▶ Action
│                       │  │  • CNOT Entanglement      │  │   (Dose)
│                       │  │  • 4 Variational Layers   │  │       │
│                       │  └───────────────────────────┘  │       │
│                       └─────────────────────────────────┘       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Twin Critic Networks (Classical)           │    │
│  │   Q1(s,a) & Q2(s,a) → Value Estimation (TD3 style)      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Propofol Environment                         │
├─────────────────────────────────────────────────────────────────┤
│  State (8-dim): [BIS_err, Ce_PPF, dBIS/dt, u_{t-1},             │
│                  PPF_acc, RFTN_acc, BIS_slope, RFTN_t]          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │     Dual Drug Patient Model (State-Space: ẋ = Ax + Bu)  │    │
│  │   ┌─────────────────────┐  ┌─────────────────────┐      │    │
│  │   │ Schnider (Propofol) │  │  Minto (Remifentanil)│      │    │
│  │   │ C1, C2, C3, Ce_PPF  │  │  C1, C2, C3, Ce_RFTN │      │    │
│  │   └─────────┬───────────┘  └──────────┬──────────┘      │    │
│  │             │                         │                 │    │
│  │             ▼                         ▼                 │    │
│  │   ┌─────────────────────────────────────────────────┐   │    │
│  │   │        Drug Interaction BIS Model (32)          │   │    │
│  │   │ BIS = 98·(1 + e^(Ce_PPF/4.47) + e^(Ce_RFTN/19.3))^(-1.43) │
│  │   └─────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Reward: R = 1 / (|g - BIS| + α)  (Formulation 40)              │
└─────────────────────────────────────────────────────────────────┘
```

## 🧬 Quantum Circuit

The 2-qubit Variational Quantum Circuit:

```
|0⟩ ─ RX(θ_in[0]) ─ RY(θ[0]) ─ RZ(θ[1]) ─●─ RY(θ[4]) ─ RZ(θ[5]) ─●─ ... ─ M
                                          │                       │
|0⟩ ─ RX(θ_in[1]) ─ RY(θ[2]) ─ RZ(θ[3]) ─⊕─ RY(θ[6]) ─ RZ(θ[7]) ─⊕─ ... ─ M

Where:
- θ_in: Encoded state features (BIS error, Ce)
- θ: Trainable variational parameters
- M: Measurement (expectation value → action)
```

## 📁 Project Structure

```
QRL_Propofol_Infusion/
├── config/
│   └── hyperparameters.yaml      # Configuration (DDPG/PPO, encoders, rewards)
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── patient_simulator.py  # Schnider & Minto PK/PD models
│   │   └── propofol_env.py       # Gymnasium environment (8-dim state)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vqc.py                # Variational Quantum Circuit
│   │   └── networks.py           # LSTM, Transformer, Critics, BIS Predictor
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── quantum_agent.py      # Quantum DDPG agent
│   │   └── quantum_ppo_agent.py  # Quantum PPO agent (Formulations 41-49)
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # MDPE, MDAPE, Wobble (Formulations 50-52)
│       └── visualization.py      # Plotting utilities
├── experiments/
│   └── train_quantum.py          # Training script (DDPG/PPO support)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/2Ho0/QRL_Propofol_Infusion.git
cd QRL_Propofol_Infusion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train DDPG with default configuration
python experiments/train_quantum.py

# Train PPO with LSTM encoder
python experiments/train_quantum.py --algorithm ppo --encoder lstm --episodes 1000

# Train DDPG with Transformer encoder
python experiments/train_quantum.py --algorithm ddpg --encoder transformer --seed 42

# Train with original reward function (Formulation 40)
python experiments/train_quantum.py --algorithm ppo --use_original_reward

# Train with remifentanil external input
python experiments/train_quantum.py --algorithm ddpg --encoder lstm --remifentanil

# Resume from checkpoint
python experiments/train_quantum.py --resume logs/experiment/checkpoints/checkpoint_500.pt
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--algorithm` | RL algorithm: `ddpg` or `ppo` | `ddpg` |
| `--encoder` | Temporal encoder: `none`, `lstm`, `transformer`, `hybrid` | `none` |
| `--episodes` | Number of training episodes | 1000 |
| `--seed` | Random seed | 42 |
| `--use_original_reward` | Use R = 1/(\|g-BIS\|+α) reward | False |
| `--remifentanil` | Enable remifentanil external input | False |

### Configuration

Edit `config/hyperparameters.yaml` to customize:

```yaml
# Algorithm Selection
algorithm:
  type: "ppo"  # or "ddpg"
  
  ppo:  # PPO-specific (Formulations 41-49)
    gae_lambda: 0.95      # GAE λ (46)
    clip_epsilon: 0.2     # Clipping ε (42)
    value_coef: 0.5       # Value loss coefficient (43)
    entropy_coef: 0.01    # Entropy bonus (45)

# Temporal Encoder (Fig.4)
encoder:
  type: "lstm"  # or "transformer", "none", "hybrid"
  sequence_length: 10
  lstm:
    hidden_dim: 64
    num_layers: 2
    bidirectional: true

# Quantum Circuit
quantum:
  n_qubits: 2
  n_layers: 4
  
# Environment
environment:
  bis_target: 50
  use_original_reward: true  # Formulation (40)
  remifentanil:
    enabled: true
```

## 📈 Performance Metrics

Following the CBIM paper formulations (50)-(52):

| Metric | Formula | Description | Target |
|--------|---------|-------------|---------|
| **MDPE** (50) | `Median(PE)` | Median Performance Error (bias) | \|MDPE\| < 10% |
| **MDAPE** (51) | `Median(\|PE\|)` | Median Absolute Performance Error (accuracy) | MDAPE < 20% |
| **Wobble** (52) | `Median(\|PE - MDPE\|)` | Intra-individual variability | Lower is better |
| **Time in Target** | - | % time BIS in 40-60 range | > 80% |

Where Performance Error: $PE_t = \frac{BIS_t - g}{g} \times 100$

## 🔬 Mathematical Formulation

### State-Space Form (1)-(3)
$$\dot{x} = Ax + Bu$$

### PK Model - Schnider (4)-(17)
$$\frac{dC_1}{dt} = \frac{u(t)}{V_1} - (k_{10} + k_{12} + k_{13})C_1 + k_{21}\frac{V_2}{V_1}C_2 + k_{31}\frac{V_3}{V_1}C_3$$

### Effect-Site Equilibration (16)
$$\frac{dC_e}{dt} = k_{e0}(C_1 - C_e)$$

### Drug Interaction BIS Model (32)
$$BIS = 98.0 \cdot \left(1 + e^{C_{e,PPF}/4.47} + e^{C_{e,RFTN}/19.3}\right)^{-1.43}$$

### Reward Function (40)
$$R_t = \frac{1}{|g - BIS_t| + \alpha}$$

### PPO Clipped Objective (42)
$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

### GAE Advantage Estimation (46)
$$\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$

## 🔧 Dependencies

- Python >= 3.9
- PennyLane >= 0.33.0
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy, SciPy, Matplotlib

### Optional (for Real Quantum Hardware)
- qiskit-ibm-runtime (IBM Quantum)
- amazon-braket-pennylane-plugin (AWS Braket)
- boto3 (AWS services)

## 🖥️ Running on Real Quantum Hardware

### Hardware-Optimized Agent

The `HardwareOptimizedQuantumAgent` class provides optimizations for execution on actual NISQ (Noisy Intermediate-Scale Quantum) devices:

#### Key Features:
- **Reduced Circuit Depth**: Automatically adjusts VQC layers to fit hardware constraints
- **Error Mitigation**: Built-in support for noise reduction techniques
- **Cost Tracking**: Monitors quantum execution costs in real-time
- **Multiple Providers**: Supports IBM Quantum, AWS Braket, and IonQ

#### Example Usage:

```python
from src.agents.quantum_agent import HardwareOptimizedQuantumAgent

# Option 1: Simulator (for testing)
agent = HardwareOptimizedQuantumAgent(
    state_dim=8,
    action_dim=1,
    hardware_provider='simulator',
    max_circuit_depth=30
)

# Option 2: IBM Quantum
agent = HardwareOptimizedQuantumAgent(
    state_dim=8,
    action_dim=1,
    hardware_provider='ibm',
    backend_name='ibmq_manila',  # or None for least busy
    use_error_mitigation=True,
    max_circuit_depth=30,
    shots=1000
)

# Option 3: AWS Braket (IonQ)
agent = HardwareOptimizedQuantumAgent(
    state_dim=8,
    action_dim=1,
    hardware_provider='aws',
    backend_name='arn:aws:braket:us-east-1::device/qpu/ionq/Harmony',
    use_error_mitigation=True,
    shots=1000
)

# Train as usual
action = agent.select_action(state)

# Monitor costs
print(agent.get_hardware_info())
# Output: {'provider': 'ibm', 'total_executions': 1000, 
#          'estimated_cost_usd': '$1600.00', ...}
```

#### Hardware Constraints (2024-2025):

| Provider | Max Circuit Depth | Gate Error Rate | Cost per Execution |
|----------|------------------|-----------------|-------------------|
| IBM Quantum | ~100 gates | 0.1-0.5% | ~$1.60 |
| AWS Braket (IonQ) | ~200 gates | 0.1-0.3% | ~$0.35 |
| Rigetti | ~50 gates | 0.5-2% | Variable |

#### Training Cost Estimates:

- **Full Training** (200,000 steps):
  - Simulator: $0 (free)
  - AWS Braket: $10,000 - $70,000
  - IBM Quantum: $320,000
  
- **With Quantum Critic** (not recommended): $400,000 - $2,000,000

**💡 Tip**: The hybrid architecture (Quantum Actor + Classical Critic) saves ~83% of quantum execution costs while maintaining performance!

#### Setup Requirements:

1. **IBM Quantum**:
   ```bash
   pip install qiskit-ibm-runtime
   # Save your IBM Quantum token
   # https://quantum-computing.ibm.com/
   ```

2. **AWS Braket**:
   ```bash
   pip install amazon-braket-pennylane-plugin boto3
   # Configure AWS credentials
   aws configure
   ```

3. **Environment Variables** (optional):
   ```bash
   export IBMQ_TOKEN="your_token_here"
   export AWS_REGION="us-east-1"
   ```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{qrl_propofol,
  title = {Quantum Reinforcement Learning for Propofol Infusion Control},
  author = {QRL Propofol Team},
  year = {2024},
  url = {https://github.com/2Ho0/QRL_Propofol_Infusion}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
