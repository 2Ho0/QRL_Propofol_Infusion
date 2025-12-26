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

---

## 🏗️ Detailed Model Architecture

### Pharmacokinetic Models

#### Schnider Model (Propofol)

**Location**: `src/models/pharmacokinetics/schnider_model.py`

**Type**: 3-compartment model with effect-site compartment

**Compartments**:
- A1: Central compartment (plasma)
- A2: Rapid peripheral compartment
- A3: Slow peripheral compartment  
- Ae: Effect-site compartment

**Differential Equations** (Formulations 1-17):
```
dA1/dt = -(k10 + k12 + k13)×A1 + k21×A2 + k31×A3 + u(t)
dA2/dt = k12×A1 - k21×A2
dA3/dt = k13×A1 - k31×A3
dAe/dt = ke0×(A1/V1 - Ae/V1)

Concentrations:
Cp = A1 / V1  (plasma concentration, μg/mL)
Ce = Ae / V1  (effect-site concentration, μg/mL)
```

**Patient-Specific Parameters**:
- V1, V2, V3: Compartment volumes (based on age, weight, height, gender, LBM)
- Cl1, Cl2, Cl3: Inter-compartment clearances
- k10, k12, k21, k13, k31: Rate constants
- ke0: Effect-site equilibration rate (0.456 min⁻¹)

**Example**:
```python
from src.models.pharmacokinetics.schnider_model import SchniderModel

model = SchniderModel(age=45, weight=70, height=170, gender='M')
model.set_infusion_rate(100.0)  # mg/min
model.step(dt=5.0)  # 5 seconds
ce = model.Ce  # Effect-site concentration
```

#### Minto Model (Remifentanil)

**Location**: `src/models/pharmacokinetics/minto_model.py`

**Type**: 3-compartment model with effect-site compartment

**Formulations** (CBIM Paper 18-29):

**Volumes** (L):
```python
V1 = 5.1 - 0.0201×(age - 40) + 0.072×(LBM - 55)
V2 = 9.82 - 0.0811×(age - 40) + 0.108×(LBM - 55)
V3 = 5.42
```

**Clearances** (L/min):
```python
Cl1 = 2.6 - 0.0162×(age - 40) + 0.0191×(LBM - 55)
Cl2 = 2.05 - 0.0301×(age - 40)
Cl3 = 0.076 - 0.00113×(age - 40)
```

**Rate Constants**:
```python
k10 = Cl1 / V1
k12 = Cl2 / V1
k21 = Cl2 / V2
k13 = Cl3 / V1
k31 = Cl3 / V3
ke0 = 0.595 min⁻¹  # Effect-site equilibration
```

**LBM Calculation** (gender-specific):
```python
# Male
LBM = 1.1×weight - 128×(weight/height)²

# Female
LBM = 1.07×weight - 148×(weight/height)²
```

**Example**:
```python
from src.models.pharmacokinetics.minto_model import MintoModel

model = MintoModel(age=45, weight=70, height=170, gender='M')
model.set_infusion_rate(0.5)  # μg/kg/min
model.step(dt=5.0)
ce = model.Ce  # Effect-site concentration (ng/mL)
```

### Pharmacodynamic Models

#### Single Drug BIS Prediction (Hill Equation)

**Sigmoid Emax Model**:
```
Effect = Emax × Ce^γ / (EC50^γ + Ce^γ)
BIS = E0 - Effect

Parameters:
- E0 = 100 (baseline, awake)
- Emax = 100 (maximum depression)
- EC50: Concentration for 50% effect
- γ: Hill coefficient (slope)

Propofol:
- EC50 = 3.4 μg/mL
- γ = 1.47

Remifentanil:
- EC50 = 2.0 ng/mL
- γ = 1.2
```

#### Greco Response Surface Model (Dual Drug Interaction)

**Location**: `src/models/pharmacodynamics/interaction_model.py`

**Formulation** (CBIM Paper 32):
```
BIS = E0 - Emax × U / (1 + U)

where:
U = U_ppf + U_rftn + α × U_ppf × U_rftn

U_ppf = (Ce_ppf / C50_ppf)^γ₁
U_rftn = (Ce_rftn / C50_rftn)^γ₂
```

**Interaction Parameter α**:
- α = 0: Additive (no interaction)
- α > 0: Synergistic (combined effect > sum of individual effects)
- α < 0: Antagonistic (combined effect < sum)
- **Typical**: α = 1.2 (synergistic for propofol-remifentanil)

**Benefits of Synergistic Interaction**:
- Lower doses of each drug needed
- Reduced side effects
- Better hemodynamic stability
- Clinically validated for propofol-remifentanil combination

**Example**:
```python
from src.models.pharmacodynamics.interaction_model import GrecoInteractionModel

model = GrecoInteractionModel(alpha=1.2)  # Synergistic
bis = model.compute_bis(ce_propofol=3.0, ce_remifentanil=2.0)
# Result: BIS ≈ 26 (deep anesthesia)

# Compare with additive (α=0)
model_additive = GrecoInteractionModel(alpha=0.0)
bis_additive = model_additive.compute_bis(3.0, 2.0)
# Result: BIS ≈ 53 (synergy saves ~27 BIS points)
```

### Quantum Models

#### Variational Quantum Circuit (VQC)

**Location**: `src/models/vqc.py`

**Architecture**:
```
Input: Classical state vector (n_features)
       ↓
Encoder: Amplitude encoding → 2 qubits
       ↓
VQC: Parameterized gates
     - RY rotations (variational)
     - CNOT entangling
     - n_layers repetitions
       ↓
Measurement: Pauli-Z expectation
       ↓
Output: Quantum features (n_qubits)
```

**Gate Sequence** (per layer):
```python
for i in range(n_qubits):
    qml.RY(params[layer, i], wires=i)

for i in range(n_qubits - 1):
    qml.CNOT(wires=[i, i+1])
```

**Configuration**:
- 2 qubits → 2 quantum features
- 3-4 layers → 6-8 trainable parameters
- Gradient: `diff_method="backprop"` (automatic differentiation)

#### Quantum Policy Network

**Architecture**:
```
Observation (medical time series)
       ↓
Encoder: LSTM/Transformer (optional, trainable)
  - Extract temporal patterns
  - Hidden dim: 128
       ↓
Quantum Layer: VQC (trainable)
  - 2 qubits, 3 layers
  - Amplitude encoding
  - Parameterized rotations
       ↓
Output Layer: Linear + Tanh
  - Map quantum features to action
  - Action: Infusion rate change
```

**Training**:
- Optimizer: Adam (lr=3e-4)
- Separate encoder optimizer (lr=1e-3) for DDPG
- Unified optimizer for PPO
- Gradient clipping: max_norm=1.0

### Agent Architecture Comparison

#### Quantum DDPG Agent

**Algorithm**: Deep Deterministic Policy Gradient (off-policy)

**Components**:
- **Actor (Policy)**: Encoder → VQC → Output (deterministic π(s) → a)
- **Critic (Q-function)**: Encoder → VQC → Output (Q(s, a) → value)
- **Target Networks**: Soft update (τ=0.005)
- **Replay Buffer**: Size 100,000, batch 256

**Loss Functions**:
```python
# Critic loss
critic_loss = MSE(Q(s, a), r + γ × Q'(s', π'(s')))

# Actor loss (maximize Q-value)
actor_loss = -Q(s, π(s)).mean()
```

#### Quantum PPO Agent

**Algorithm**: Proximal Policy Optimization (on-policy)

**Components**:
- **Actor (Stochastic Policy)**: Encoder → VQC → μ, σ (π(a|s) ~ N(μ, σ²))
- **Critic (Value Function)**: Encoder → VQC → V(s) (state value, no action input)
- **GAE**: Generalized Advantage Estimation (λ=0.95)

**Loss Functions** (Formulations 45-48):
```python
# Policy loss (clipped objective)
ratio = π_new(a|s) / π_old(a|s)
surrogate1 = ratio × A
surrogate2 = clip(ratio, 1-ε, 1+ε) × A
policy_loss = -min(surrogate1, surrogate2).mean()

# Value loss
value_loss = MSE(V(s), R)

# Entropy bonus
entropy_loss = -entropy(π).mean()

# Total loss
loss = policy_loss + c1×value_loss + c2×entropy_loss
```

**Hyperparameters**:
```yaml
lr: 3e-4
gamma: 0.99
gae_lambda: 0.95       # GAE λ (Formulation 46)
clip_epsilon: 0.2      # Clipping ε (Formulation 42)
c1_value: 0.5          # Value loss coefficient
c2_entropy: 0.01       # Entropy bonus
n_epochs: 10           # Updates per batch
batch_size: 256
n_steps: 2048          # Steps per update
```

### Model Comparison Table

| Feature | Quantum DDPG | Classical DDPG | Quantum PPO | Classical PPO |
|---------|--------------|----------------|-------------|---------------|
| **Policy** | VQC | MLP | VQC | MLP |
| **Value** | VQC | MLP | VQC | MLP |
| **Type** | Off-policy | Off-policy | On-policy | On-policy |
| **Sample Efficiency** | High | High | Low | Low |
| **Stability** | Medium | High | High | High |
| **Parameters** | ~1,000 | ~10,000 | ~1,000 | ~10,000 |
| **Training Time** | Slow | Fast | Medium | Fast |
| **Exploration** | OU noise | OU noise | Stochastic | Stochastic |
| **Replay Buffer** | Yes | Yes | No | No |
| **Advantage** | Fewer params | Proven | Stable | Fast |

---

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
│   │   ├── vqc.py                # Variational Quantum Circuit (VQC)
│   │   ├── networks.py           # LSTM/Transformer encoders, Critics
│   │   ├── pharmacokinetics/
│   │   │   ├── schnider_model.py # Propofol 3-compartment PK
│   │   │   └── minto_model.py    # Remifentanil 3-compartment PK
│   │   └── pharmacodynamics/
│   │       └── interaction_model.py # Greco drug interaction
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── quantum_agent.py      # Quantum DDPG agent
│   │   ├── quantum_ppo_agent.py  # Quantum PPO agent (Formulations 41-49)
│   │   ├── classical_agent.py    # Classical DDPG baseline
│   │   └── classical_ppo_agent.py # Classical PPO baseline
│   ├── data/
│   │   └── vitaldb_loader.py     # VitalDB dataset loading
│   ├── visualization/
│   │   └── training_curves.py    # Training curve plotting
│   └── utils/
│       ├── __init__.py
│       ├── replay_buffer.py      # Experience replay buffer
│       ├── metrics.py            # MDPE, MDAPE, Wobble (Formulations 50-52)
│       └── visualization.py      # Plotting utilities
├── experiments/
│   ├── train_quantum.py          # Quantum DDPG training
│   ├── train_classical.py        # Classical DDPG training
│   ├── train_ppo.py              # PPO online training
│   ├── train_hybrid.py           # DDPG hybrid (offline → online)
│   ├── train_hybrid_ppo.py       # PPO hybrid (BC → online)
│   ├── train_dual_drug.py        # Dual drug training
│   ├── train_offline.py          # Offline RL (BC, CQL)
│   ├── compare_quantum_vs_classical.py       # DDPG comparison
│   ├── compare_ddpg_vs_ppo.py                # Algorithm comparison
│   └── compare_quantum_vs_classical_dualdrug.py  # Dual drug comparison
├── data/
│   ├── prepare_vitaldb.py        # VitalDB preprocessing & caching
│   ├── vitaldb_cache/            # Raw VitalDB data cache
│   └── offline_dataset/          # Preprocessed offline datasets
├── config/
│   └── hyperparameters.yaml      # Complete hyperparameter configuration
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

#### 1. Offline RL Training (VitalDB Real Patient Data)

```bash
# Download and preprocess VitalDB data (first time only)
python experiments/train_offline.py --download --max_cases 100

# Train with Behavioral Cloning (warm-start)
python experiments/train_offline.py --bc_only --bc_epochs 100

# Train with BC + Conservative Q-Learning (CQL)
python experiments/train_offline.py --bc_epochs 50 --cql_epochs 500

# Evaluate on VitalDB test set
python experiments/train_offline.py --evaluate --checkpoint logs/offline/best_model.pt
```

#### 2. Hybrid Training (Offline → Online)

```bash
# Single drug (Propofol only) - DDPG
python experiments/train_hybrid.py --n_cases 100 --offline_epochs 50 --online_episodes 500

# Single drug - PPO
python experiments/train_hybrid_ppo.py --n_cases 100 --offline_epochs 50 --online_episodes 500

# With LSTM encoder
python experiments/train_hybrid.py --n_cases 100 --offline_epochs 50 --online_episodes 500 --encoder lstm

# Quick test
python experiments/train_hybrid.py --n_cases 20 --offline_epochs 5 --online_episodes 50
```

#### 3. Dual Drug Training (Propofol + Remifentanil)

```bash
# Train quantum agent on dual drug control
python experiments/train_dual_drug.py --n_episodes 500 --encoder none

# With LSTM encoder
python experiments/train_dual_drug.py --n_episodes 500 --encoder lstm

# Quick test
python experiments/train_dual_drug.py --n_episodes 50
```

#### 4. Online Training (Pure RL)

```bash
# Quantum DDPG
python experiments/train_quantum.py --n_episodes 1000 --encoder lstm

# Classical DDPG
python experiments/train_classical.py --n_episodes 1000 --encoder none

# Quantum PPO
python experiments/train_ppo.py --agent_type quantum --n_episodes 1000

# Classical PPO
python experiments/train_ppo.py --agent_type classical --n_episodes 1000
```

#### 5. Comparison Studies

```bash
# Compare Quantum vs Classical (single drug)
python experiments/compare_quantum_vs_classical.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500

# Compare Quantum vs Classical (dual drug)
python experiments/compare_quantum_vs_classical_dualdrug.py \
    --online_episodes 500 --n_test_episodes 50

# Quick comparison
python experiments/compare_quantum_vs_classical.py \
    --n_cases 20 --offline_epochs 5 --online_episodes 50
```

### Key Training Scripts

| Script | Purpose | Algorithm |
|--------|---------|----------|
| `train_offline.py` | VitalDB offline RL (BC, CQL) | DDPG |
| `train_hybrid.py` | Offline→Online (DDPG) | DDPG |
| `train_hybrid_ppo.py` | Offline→Online (PPO) | PPO |
| `train_quantum.py` | Pure online quantum DDPG | DDPG |
| `train_classical.py` | Pure online classical DDPG | DDPG |
| `train_ppo.py` | Pure online PPO (quantum/classical) | PPO |
| `train_dual_drug.py` | Dual drug control | DDPG |
| `compare_quantum_vs_classical.py` | Quantum vs Classical | Both |
| `compare_ddpg_vs_ppo.py` | DDPG vs PPO | Both |

### Common Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--algorithm` | RL algorithm: `ddpg` or `ppo` | `ddpg` |
| `--encoder` | Temporal encoder: `none`, `lstm`, `transformer`, `hybrid` | `none` |
| `--episodes` | Number of training episodes | 1000 |
| `--seed` | Random seed | 42 |
| `--use_original_reward` | Use R = 1/(\|g-BIS\|+α) reward | False |
| `--remifentanil` | Enable remifentanil external input | False |

## 🗄️ VitalDB Real Patient Data Integration

This project now supports training on **real patient data** from the VitalDB dataset for offline reinforcement learning.

### What is VitalDB?

[VitalDB](https://vitaldb.net/) is an open clinical dataset containing high-resolution intraoperative biosignals from 6,388 surgical patients. For propofol anesthesia control, it provides:

- **BIS monitoring** data with signal quality indices
- **TCI pump data**: Propofol and Remifentanil effect-site concentrations, infusion rates, volumes
- **Vital signs**: HR, blood pressure, SpO2
- **Patient demographics**: Age, sex, height, weight
- **Complete drug dosing histories** from expert anesthesiologists

### Quick Start: Offline RL Training

#### 1. Download and Preprocess VitalDB Data

```bash
# Download VitalDB cases and create offline dataset
python experiments/train_offline.py --download --max_cases 50
```

This will:
- Download VitalDB cases with BIS monitoring and propofol TCI
- Apply quality filters (BIS SQI > 50, missing data < 20%, duration 30-240 min)
- Extract state-action-reward tuples matching 8-dim environment state
- Save preprocessed dataset to `data/offline_dataset/vitaldb_processed.pkl`
- Split into train (70%), validation (15%), test (15%)

#### 2. Train with Behavioral Cloning (Warm-start)

```bash
# Pre-train by imitating expert anesthesiologists
python experiments/train_offline.py --bc_only --bc_epochs 100
```

Behavioral cloning learns the policy through supervised learning from expert demonstrations.

#### 3. Train with Conservative Q-Learning (Safe Offline RL)

```bash
# BC warm-start + CQL fine-tuning
python experiments/train_offline.py --bc_epochs 50 --cql_epochs 500
```

CQL adds conservative penalties to prevent unsafe actions not seen in the dataset.

#### 4. Evaluate on VitalDB Test Set and Synthetic Patients

```bash
# Evaluate trained model
python experiments/train_offline.py --evaluate --checkpoint logs/offline/best_model.pt
```

Compares performance against:
- Actual anesthesiologist performance (from VitalDB)
- Synthetic patients from PK/PD model
- Schnider TCI baseline

### VitalDB Data Quality Filters

The following filters ensure high-quality training data:

| Filter | Threshold | Reason |
|--------|-----------|--------|
| **BIS Signal Quality** | SQI > 50 | Remove poor EEG signal quality |
| **Missing Data** | < 20% missing | Ensure sufficient observations |
| **Duration** | 30-240 minutes | Focus on typical anesthesia duration |
| **Demographics** | Age 18-90, Weight 40-150kg | Valid PK/PD model range |
| **Propofol TCI** | Required | Need infusion rate history |

### Offline RL Configuration

Edit `config/hyperparameters.yaml`:

```yaml
# VitalDB Data Configuration
vitaldb:
  cache_dir: "data/vitaldb_cache"
  max_cases: 100
  include_remifentanil: true
  bis_sqi_threshold: 50
  max_missing_ratio: 0.2
  sampling_interval: 5.0

# Offline RL Configuration
offline:
  # Behavioral Cloning
  behavioral_cloning:
    enabled: true
    epochs: 50
    batch_size: 256
    learning_rate: 0.0001
    early_stopping_patience: 10
  
  # Conservative Q-Learning
  cql:
    enabled: true
    alpha: 1.0              # CQL regularization coefficient
    min_q_weight: 5.0       # Conservative underestimation
    num_random_actions: 10  # Random actions for penalty
  
  # Evaluation
  evaluation:
    eval_on_vitaldb: true
    eval_on_synthetic: true
    n_synthetic_patients: 20
```

### VitalDB Command Reference

```bash
# Download specific number of cases
python experiments/train_offline.py --download --max_cases 100

# Train behavioral cloning only (no CQL)
python experiments/train_offline.py --bc_only --bc_epochs 100

# Train with BC + CQL (recommended)
python experiments/train_offline.py --bc_epochs 50 --cql_epochs 500

# Evaluate model on both VitalDB and synthetic patients
python experiments/train_offline.py --evaluate --checkpoint logs/offline/best.pt

# Custom configuration
python experiments/train_offline.py --config config/hyperparameters.yaml --seed 42
```

### Offline Dataset Structure

The preprocessed dataset (`vitaldb_processed.pkl`) contains:

```
train/
  episode_0/
    states         # [T, 8] - BIS error, Ce, slopes, accumulated doses, time
    actions        # [T, 1] - Propofol infusion rate (μg/kg/min)
    rewards        # [T, 1] - R = 1/(|g-BIS|+α) with safety penalties
    dones          # [T, 1] - Episode termination flags
    demographics   # Patient: age, weight, height, sex
  episode_1/
    ...
val/
  ...
test/
  ...
```

### Performance Comparison

Expected performance metrics on VitalDB test set:

| Method | MDPE | MDAPE | Wobble | Time in Target |
|--------|------|-------|--------|----------------|
| **Anesthesiologist (actual)** | -5.2% | 18.4% | 12.1% | 82.3% |
| **Schnider TCI** | -3.8% | 21.5% | 15.3% | 76.5% |
| **BC Only** | -7.1% | 22.8% | 16.8% | 74.2% |
| **BC + CQL (ours)** | **-4.5%** | **19.7%** | **13.4%** | **80.1%** |

*Note: Values are illustrative. Actual performance depends on training configuration.*

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

### Clinical Performance Metrics (50-52)

**Median Performance Error (MDPE)** - Bias:
$$MDPE = \text{Median}\left(\frac{BIS_t - g}{g} \times 100\right)$$

**Median Absolute Performance Error (MDAPE)** - Accuracy:
$$MDAPE = \text{Median}\left(\left|\frac{BIS_t - g}{g} \times 100\right|\right)$$

**Wobble** - Intra-individual Variability:
$$\text{Wobble} = \text{Median}\left(|PE_t - MDPE|\right)$$

Where $PE_t = \frac{BIS_t - g}{g} \times 100$ is the Performance Error at time $t$, and $g$ is the target BIS (typically 50).

---

## 🧪 Training Scripts Reference

### Online Training

**Quantum DDPG**:
```bash
python experiments/train_quantum.py --n_episodes 1000 --encoder lstm
```

**Classical DDPG**:
```bash
python experiments/train_classical.py --n_episodes 1000 --encoder none
```

**Quantum PPO**:
```bash
python experiments/train_ppo.py --agent_type quantum --n_episodes 1000 --encoder lstm
```

**Classical PPO**:
```bash
python experiments/train_ppo.py --agent_type classical --n_episodes 1000 --encoder lstm
```

### Hybrid Training (Offline → Online)

**DDPG Hybrid**:
```bash
python experiments/train_hybrid.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500
```

**PPO Hybrid** (Behavioral Cloning → PPO):
```bash
python experiments/train_hybrid_ppo.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500
```

### Algorithm Comparison

**Quantum vs Classical**:
```bash
python experiments/compare_quantum_vs_classical.py \
    --n_cases 100 --offline_epochs 50 --online_episodes 500
```

**DDPG vs PPO**:
```bash
python experiments/compare_ddpg_vs_ppo.py \
    --online_episodes 500 --n_test_episodes 50
```

### Hyperparameter Configuration

The complete configuration is in `config/hyperparameters.yaml`:

```yaml
# =============================================================================
# Environment Configuration
# =============================================================================
environment:
  bis_target: 50              # Target BIS (Formulation 40)
  bis_min: 40
  bis_max: 60
  bis_baseline: 97
  
  dose_min: 0.0               # μg/kg/min
  dose_max: 200.0
  
  dt: 5.0                     # Time step (seconds)
  episode_duration: 3600      # 60 minutes
  
  use_extended_state: true    # 8-dim state (Formulations 36-39)
  use_original_reward: true   # R = 1/(|g-BIS|+α) (Formulation 40)
  reward_alpha: 0.1
  
  remifentanil:
    enabled: true
    profile_type: "random"    # "constant", "random", "surgical"
    min_rate: 0.0
    max_rate: 0.5
    mean_rate: 0.2
    std_rate: 0.1

# =============================================================================
# PK/PD Model Configuration (Schnider & Minto)
# =============================================================================
pkpd_model:
  model_type: "schnider"      # Propofol PK model
  bis_model: "drug_interaction"  # Formulation (32)
  
  schnider:
    ke0: 0.456                # Effect-site equilibration (Formulation 16)
  
  minto:
    ke0: 0.595                # Remifentanil ke0 (Formulation 29)
  
  drug_interaction:           # Formulation (32)
    alpha_ppf: 4.47
    alpha_rftn: 19.3
    baseline: 98.0
    gamma: 1.43

# =============================================================================
# Quantum Circuit Configuration
# =============================================================================
quantum:
  n_qubits: 2
  n_layers: 4
  encoding: "angle"           # "angle" or "amplitude"
  backend: "default.qubit"
  diff_method: "backprop"     # Gradient computation

# =============================================================================
# RL Algorithm Configuration
# =============================================================================
ddpg:
  learning_rate_actor: 0.0003
  learning_rate_critic: 0.001
  learning_rate_encoder: 0.001  # Separate encoder optimizer
  gamma: 0.99                  # Discount factor
  tau: 0.005                   # Soft update rate
  batch_size: 256
  buffer_size: 100000
  exploration_noise: 0.1       # OU noise

ppo:                           # Formulations (41)-(49)
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95             # GAE λ (Formulation 46)
  clip_epsilon: 0.2            # Clipping ε (Formulation 42)
  value_loss_coef: 0.5         # c1 (Formulation 43)
  entropy_coef: 0.01           # c2 (Formulation 45)
  max_grad_norm: 0.5
  n_epochs: 10
  batch_size: 256
  n_steps: 2048                # Rollout length

# =============================================================================
# Encoder Configuration (LSTM/Transformer)
# =============================================================================
encoder:
  type: "lstm"                 # "none", "lstm", "transformer"
  enabled: true
  sequence_length: 10
  hidden_dim: 128
  
  lstm:
    num_layers: 2
    bidirectional: true
    dropout: 0.1
  
  transformer:
    num_layers: 2
    num_heads: 4
    ffn_dim: 512
    dropout: 0.1

# =============================================================================
# Training Configuration
# =============================================================================
training:
  n_episodes: 1000
  max_steps_per_episode: 720   # 60 minutes at 5s intervals
  save_interval: 50
  eval_interval: 10
  n_eval_episodes: 5
  log_interval: 10
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 50
    min_delta: 0.01

# =============================================================================
# Offline RL Configuration (VitalDB)
# =============================================================================
offline:
  # Data configuration
  data:
    cache_dir: "data/vitaldb_cache"
    processed_dir: "data/offline_dataset"
    max_cases: 100
    train_split: 0.7
    val_split: 0.15
    test_split: 0.15
    
  # Quality filters
  filters:
    bis_sqi_threshold: 50      # BIS signal quality
    max_missing_ratio: 0.2
    min_duration: 1800         # 30 minutes
    max_duration: 14400        # 4 hours
  
  # Behavioral Cloning
  behavioral_cloning:
    enabled: true
    epochs: 50
    batch_size: 256
    learning_rate: 0.0001
    early_stopping_patience: 10
  
  # Conservative Q-Learning (CQL)
  cql:
    enabled: true
    alpha: 1.0                 # CQL regularization
    min_q_weight: 5.0          # Conservative penalty
    num_random_actions: 10
    temperature: 1.0
```

---

## 🔧 Dependencies

- Python >= 3.9
- PennyLane >= 0.33.0
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy, SciPy, Matplotlib
- VitalDB >= 1.7.0 (for real patient data)
- h5py, pandas, pyarrow (for offline dataset processing)

See [requirements.txt](requirements.txt) for complete dependencies.

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

If you use the VitalDB dataset, please also cite:

```bibtex
@article{lee2022vitaldb,
  title={VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients},
  author={Lee, Hyung-Chul and Jung, Chul-Woo},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={279},
  year={2022},
  publisher={Nature Publishing Group}
}
```

## 📄 License

This project is licensed under the MIT License.

VitalDB data is licensed under CC BY-NC-SA 4.0 for research use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
