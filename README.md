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

## 🚧 Current Progress & Extensions

### Neural PK/PD System Identification
We are extending the project to include a **Neural PK/PD** module that learns patient-specific pharmacokinetics and pharmacodynamics in real-time.
- **Goal**: Adaptive control based on personalized patient physiology, moving beyond fixed population models (Schnider/Minto).
- **Features**: 
    - Real-time parameter estimation
    - integration with VitalDB for training
    - Support for additional vital signs (HR, MAP, SEF, SQI, SR)

### Ongoing Improvements
- **VitalDB Loader**: Enhanced to support multi-modal data loading (EEG + Vitals).
- **Quantum Agent**: Refining the comparison between Quantum and Classical DDPG/PPO agents.


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

#### 1. **Run Full Comparison Experiment (Dual Drug)**

This command executes the two-stage hybrid training (Offline Behavioral Cloning → Online DDPG fine-tuning) on dual-drug control (Propofol + Remifentanil), and seamlessly compares the Quantum vs. Classical RL agents.

```bash
python experiments/compare_quantum_vs_classical.py \
    --n_cases 6000 \
    --offline_epochs 30 \
    --batch_size 128 \
    --bc_weight 0.8 \
    --use_cql False \
    --online_episodes 300 \
    --sampling_interval 60 \
    --smoothing_beta 0.8
```

#### **Key Experiment Parameters**:
- `--n_cases 6000`: Utilizes 6000 cases from VitalDB for offline learning.
- `--offline_epochs 30`: 30 epochs for Stage 1 (offline pre-training).
- `--batch_size 128`: Batch size designated for stable optimization.
- `--bc_weight 0.8`: Behavioral Cloning (BC) weight of 0.8 (mixing BC loss and RL loss).
- `--use_cql False`: Disables Conservative Q-Learning since BC is preferred for imitation.
- `--online_episodes 300`: 300 episodes for Stage 2 (online fine-tuning).
- `--sampling_interval 60`: Reduces computational overhead by subsampling data dynamically.
- `--smoothing_beta 0.8`: Enforces action-smoothing mechanism over output distributions (EMA factor).

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
│   │   ├── neural_pkpd.py            # Neural PK/PD System ID
│   │   ├── pharmacokinetics/         # Default PK models
│   │   └── pharmacodynamics/         # Default PD models
│   ├── agents/
│   │   ├── quantum_agent.py          # Quantum DDPG
│   │   └── classical_agent.py        # Classical DDPG
│   └── data/
│       ├── vitaldb_loader.py         # VitalDB data loading
│       └── vitaldb_loader_remi.py    # Combined Propofol/Remi data loader
├── experiments/
│   ├── compare_quantum_vs_classical.py # Main train & eval script
│   ├── train_hybrid.py               # 2-stage training logic
│   └── plot_training_results.py      # Plotting utilities
├── data/
│   └── vitaldb_cache/                # Raw VitalDB patient data
└── logs/                             # Checkpoints and trajectories
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

## 🔧 Dependencies

- Python >= 3.12.4
- PennyLane >= 0.33.0 (Quantum computing)
- PyTorch >= 2.0.0 (Deep learning)
- Gymnasium >= 0.29.0 (RL environment)
- VitalDB >= 1.7.0 (Real patient data)
- NumPy, SciPy, Matplotlib, pandas

See `requirements.txt` for complete list.

## 📄 License

MIT License. VitalDB data: CC BY-NC-SA 4.0 for research use.

## 🤝 Contributing

Contributions welcome! Please submit a Pull Request.
