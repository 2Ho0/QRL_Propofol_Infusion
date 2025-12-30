# Quantum Reinforcement Learning for Propofol Infusion Control

Hybrid Quantum-Classical RL system for closed-loop BIS-guided propofol anesthesia control with VitalDB real patient data integration.

## 🎯 Key Features

- **Quantum Algorithms**: QDDPG & QPPO with 2-qubit Variational Quantum Circuit (VQC)
- **Dual Drug Control**: Propofol + Remifentanil with Greco interaction model
- **VitalDB Integration**: Train on 6,388 real surgical patients' data (2,830 transitions from 5 cases)
- **2-Stage Training**: Offline pre-training (BC/CQL) → Online fine-tuning (DDPG/PPO)
- **PK/PD Models**: Schnider (Propofol) + Minto (Remifentanil) 3-compartment models
- **Temporal Encoders**: LSTM and Transformer for time-series processing
- **Clinical Metrics**: MDPE, MDAPE, Wobble for performance evaluation

## 📊 System Architecture

```
Input: BIS monitoring + Patient state (10D)
  ↓
[Optional] Temporal Encoder (LSTM/Transformer)
  ↓
Quantum Policy Network (2-qubit VQC)
  • Angle encoding
  • RY-RZ rotations
  • CNOT entanglement
  ↓
Action: Infusion rates (Propofol + Remifentanil)
  ↓
Patient Simulator / VitalDB Data
  • Schnider PK/PD (Propofol)
  • Minto PK/PD (Remifentanil)
  • Greco interaction model
  ↓
Reward: BIS target tracking + Safety
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/2Ho0/QRL_Propofol_Infusion.git
cd QRL_Propofol_Infusion
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Training Commands

#### 1. **Offline Pre-training** (VitalDB Real Patient Data)

```bash
# Download VitalDB data (first time only)
python experiments/train_offline.py --download --max_cases 100

# Behavioral Cloning (BC) only
python experiments/train_offline.py --bc_only --bc_epochs 100

# Conservative Q-Learning (CQL)
python experiments/train_offline.py --bc_epochs 50 --cql_epochs 500
```

#### 2. **2-Stage Hybrid Training** (Offline → Online)

**Single Drug (Propofol)**:
```bash
# BC-only offline training
python experiments/train_hybrid.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500

# BC + CQL (more conservative)
python experiments/train_hybrid.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500 \
    --use_cql --cql_alpha 1.0
```

**Dual Drug (Propofol + Remifentanil)**:
```bash
# BC-only (default)
python experiments/compare_quantum_vs_classical_dualdrug.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500

# CQL-only (conservative)
python experiments/compare_quantum_vs_classical_dualdrug.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500 \
    --use_cql --bc_weight 0.0 --cql_alpha 1.0

# BC + CQL hybrid (recommended)
python experiments/compare_quantum_vs_classical_dualdrug.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500 \
    --use_cql --bc_weight 0.8 --cql_alpha 1.0
```

**CQL Parameters**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--use_cql` | Enable Conservative Q-Learning | False |
| `--bc_weight` | BC loss weight (0.0-1.0) | 1.0 |
| `--cql_alpha` | CQL penalty weight | 1.0 |
| `--cql_temp` | Temperature for logsumexp | 1.0 |
| `--cql_num_random` | Random actions for penalty | 10 |

#### 3. **Pure Online Training** (Simulator Only)

```bash
# Quantum DDPG
python experiments/train_quantum.py --n_episodes 1000 --encoder lstm

# Classical DDPG
python experiments/train_classical.py --n_episodes 1000

# Quantum PPO
python experiments/train_ppo.py --agent_type quantum --n_episodes 1000

# Classical PPO
python experiments/train_ppo.py --agent_type classical --n_episodes 1000
```

#### 4. **Algorithm Comparison**

```bash
# Quantum vs Classical (single drug)
python experiments/compare_quantum_vs_classical.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500

# Quantum vs Classical (dual drug)
python experiments/compare_quantum_vs_classical_dualdrug.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500

# DDPG vs PPO
python experiments/compare_ddpg_vs_ppo.py \
    --online_episodes 500 --n_test_episodes 50
```

#### 5. **Quick Test** (Fast debugging)

```bash
# Single drug
python experiments/train_hybrid.py \
    --n_cases 5 --offline_epochs 5 --online_episodes 50

# Dual drug
python experiments/compare_quantum_vs_classical_dualdrug.py \
    --n_cases 5 --offline_epochs 5 --online_episodes 50
```

## 🗄️ VitalDB Real Patient Data

### Dataset Overview

[VitalDB](https://vitaldb.net/): Open clinical dataset with 6,388 surgical patients
- **BIS monitoring** with signal quality indices
- **TCI pump data**: Propofol/Remifentanil concentrations, infusion rates
- **Vital signs**: HR, BP, SpO2
- **Demographics**: Age, sex, height, weight
- **Expert dosing** from anesthesiologists

### Data Statistics (Dual Drug Cases)

From **5 VitalDB cases → 2,830 transitions**:

| Metric | Range | Mean ± SD |
|--------|-------|-----------|
| **Propofol** | 0.0 - 27.9 mg/kg/h | 6.1 ± 5.2 mg/kg/h |
| **Remifentanil** | 0.02 - 2.75 μg/kg/min | 0.63 ± 0.59 μg/kg/min |
| **BIS** | 35 - 96 | 49 ± 12 |
| **Reward** | -1.74 to 0.99 | -0.08 ± 0.33 |

## 📁 Project Structure

```
QRL_Propofol_Infusion/
├── config/
│   └── hyperparameters.yaml          # Training configuration
├── src/
│   ├── environment/
│   │   ├── patient_simulator.py      # Schnider + Minto PK/PD models
│   │   ├── propofol_env.py           # Single drug environment
│   │   └── dual_drug_env.py          # Dual drug environment
│   ├── models/
│   │   ├── vqc.py                    # Variational Quantum Circuit
│   │   ├── networks.py               # LSTM/Transformer encoders
│   │   ├── pharmacokinetics/         # Schnider + Minto models
│   │   └── pharmacodynamics/         # Greco interaction model
│   ├── agents/
│   │   ├── quantum_agent.py          # Quantum DDPG
│   │   ├── quantum_ppo_agent.py      # Quantum PPO
│   │   ├── classical_agent.py        # Classical DDPG
│   │   └── classical_ppo_agent.py    # Classical PPO
│   └── data/
│       └── vitaldb_loader.py         # VitalDB data loading
├── experiments/
│   ├── train_offline.py              # Offline RL (BC/CQL)
│   ├── train_hybrid.py               # 2-stage training (DDPG)
│   ├── train_hybrid_ppo.py           # 2-stage training (PPO)
│   ├── train_quantum.py              # Pure online quantum DDPG
│   ├── train_classical.py            # Pure online classical DDPG
│   ├── train_ppo.py                  # Pure online PPO
│   ├── compare_quantum_vs_classical.py
│   ├── compare_quantum_vs_classical_dualdrug.py
│   └── compare_ddpg_vs_ppo.py
├── data/
│   ├── vitaldb_cache/                # Raw VitalDB data
│   └── offline_dataset/              # Preprocessed datasets
└── logs/                             # Training logs
```

## 🔬 Key Algorithms

### DDPG (Deep Deterministic Policy Gradient)
- **Type**: Off-policy, deterministic
- **Policy**: Actor network (π: s → a)
- **Value**: Twin critics (Q₁, Q₂)
- **Sample Efficiency**: High (replay buffer)
- **Exploration**: OU noise

### PPO (Proximal Policy Optimization)
- **Type**: On-policy, stochastic
- **Policy**: Actor with Gaussian distribution (π: s → N(μ, σ²))
- **Value**: Critic (V: s → value)
- **Stability**: High (clipped objective)
- **Exploration**: Entropy bonus

### BC (Behavioral Cloning)
- **Type**: Supervised learning
- **Loss**: MSE(π(s), a_expert)
- **Use**: Warm-start from expert demonstrations
- **Pros**: Fast convergence
- **Cons**: Distribution shift

### CQL (Conservative Q-Learning)
- **Type**: Offline RL
- **Penalty**: Minimize Q-values for out-of-distribution actions
- **Formula**: Q_loss = TD_loss + α × (logsumexp(Q_random) - Q_data)
- **Use**: Safe offline learning
- **Pros**: Conservative, robust
- **Cons**: Can be overly cautious

## 📊 Performance Metrics

### Clinical Metrics
- **MDPE** (Median Performance Error): Bias
- **MDAPE** (Median Absolute Performance Error): Accuracy
- **Wobble**: Intra-individual variability
- **Time in Target**: % time BIS within 40-60

### Training Metrics
- Episode reward
- BIS tracking error
- Drug usage
- Policy entropy (PPO)
- Q-values (DDPG)

## ⚙️ Configuration

Key hyperparameters in `config/hyperparameters.yaml`:

```yaml
# Environment
environment:
  bis_target: 50
  dt: 5.0                    # Time step (seconds)
  episode_duration: 3600     # 60 minutes

# DDPG
ddpg:
  learning_rate: 0.0003
  gamma: 0.99
  tau: 0.005                 # Soft update
  buffer_size: 100000
  exploration_noise: 0.1

# PPO
ppo:
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  n_epochs: 10
  batch_size: 256

# Offline RL
offline:
  behavioral_cloning:
    enabled: true
    epochs: 50
    batch_size: 256
  cql:
    enabled: true
    alpha: 1.0
    num_random_actions: 10
    temperature: 1.0
```

## 🔧 Dependencies

- Python >= 3.9
- PennyLane >= 0.33.0 (Quantum computing)
- PyTorch >= 2.0.0 (Deep learning)
- Gymnasium >= 0.29.0 (RL environment)
- VitalDB >= 1.7.0 (Real patient data)
- NumPy, SciPy, Matplotlib, pandas

See `requirements.txt` for complete list.

## 📝 Citation

```bibtex
@software{qrl_propofol,
  title = {Quantum Reinforcement Learning for Propofol Infusion Control},
  author = {QRL Propofol Team},
  year = {2024},
  url = {https://github.com/2Ho0/QRL_Propofol_Infusion}
}

@article{lee2022vitaldb,
  title={VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients},
  author={Lee, Hyung-Chul and Jung, Chul-Woo},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={279},
  year={2022}
}
```

## 📄 License

MIT License. VitalDB data: CC BY-NC-SA 4.0 for research use.

## 🤝 Contributing

Contributions welcome! Please submit a Pull Request.
