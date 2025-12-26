# Model Architecture Documentation

This document describes the model architecture and implementation details of the Quantum Reinforcement Learning system for propofol infusion control.

## Table of Contents

1. [Overview](#overview)
2. [Pharmacokinetic Models](#pharmacokinetic-models)
3. [Pharmacodynamic Models](#pharmacodynamic-models)
4. [Quantum Models](#quantum-models)
5. [Classical Models](#classical-models)
6. [Agent Architecture](#agent-architecture)
7. [Implementation Details](#implementation-details)

---

## Overview

The system implements a hybrid quantum-classical reinforcement learning approach for automated anesthesia control. The architecture combines:

- **Pharmacokinetic (PK) Models**: Schnider (propofol), Minto (remifentanil)
- **Pharmacodynamic (PD) Models**: BIS effect prediction, Greco interaction
- **Quantum Models**: Variational Quantum Circuit (VQC) for policy/value
- **Classical Models**: LSTM/Transformer encoders, MLP baselines
- **RL Algorithms**: DDPG (off-policy), PPO (on-policy)

---

## Pharmacokinetic Models

### Schnider Model (Propofol)

**Location**: `src/models/pharmacokinetics/schnider_model.py`

**Type**: 3-compartment model with effect-site compartment

**Formulations** (CBIM Paper 1-17):

```
Compartments:
- A1: Central compartment (plasma)
- A2: Rapid peripheral compartment
- A3: Slow peripheral compartment  
- Ae: Effect-site compartment

Differential Equations:
dA1/dt = -(k10 + k12 + k13)×A1 + k21×A2 + k31×A3 + u(t)
dA2/dt = k12×A1 - k21×A2
dA3/dt = k13×A1 - k31×A3
dAe/dt = ke0×(A1/V1 - Ae/V1)

where u(t) is infusion rate (mg/min)

Concentrations:
Cp = A1 / V1  (plasma)
Ce = Ae / V1  (effect-site)
```

**Patient Covariates**:
- Age (years)
- Weight (kg)
- Height (cm)
- Gender (M/F)
- Lean body mass (LBM)

**Parameters** (age, weight, height, gender dependent):
- V1, V2, V3: Compartment volumes
- Cl1, Cl2, Cl3: Clearances
- k10, k12, k21, k13, k31: Rate constants
- ke0: Effect-site equilibration rate (0.456 min⁻¹)

**Usage**:
```python
from src.models.pharmacokinetics.schnider_model import SchniderModel

model = SchniderModel(age=45, weight=70, height=170, gender='M')
model.set_infusion_rate(100.0)  # mg/min
model.step(dt=5.0)  # 5 seconds
ce = model.Ce  # Effect-site concentration (μg/mL)
```

---

### Minto Model (Remifentanil)

**Location**: `src/models/pharmacokinetics/minto_model.py`

**Type**: 3-compartment model with effect-site compartment

**Formulations** (CBIM Paper 18-29):

```
Compartments: [A1, A2, A3, Ae]

Differential Equations:
dA1/dt = -(k10 + k12 + k13)×A1 + k21×A2 + k31×A3 + u(t)
dA2/dt = k12×A1 - k21×A2
dA3/dt = k13×A1 - k31×A3
dAe/dt = ke0×(A1/V1 - Ae/V1)

where u(t) is infusion rate (μg/kg/min)

Concentrations:
Cp = A1 / V1  (ng/mL)
Ce = Ae / V1  (ng/mL)
```

**Parameters** (Formulations 21-29):
```python
# Volumes (L)
V1 = 5.1 - 0.0201×(age - 40) + 0.072×(LBM - 55)
V2 = 9.82 - 0.0811×(age - 40) + 0.108×(LBM - 55)
V3 = 5.42

# Clearances (L/min)
Cl1 = 2.6 - 0.0162×(age - 40) + 0.0191×(LBM - 55)
Cl2 = 2.05 - 0.0301×(age - 40)
Cl3 = 0.076 - 0.00113×(age - 40)

# Rate constants (min⁻¹)
k10 = Cl1 / V1
k12 = Cl2 / V1
k21 = Cl2 / V2
k13 = Cl3 / V1
k31 = Cl3 / V3

# Effect-site equilibration
ke0 = 0.595 min⁻¹
```

**LBM Calculation** (gender-specific):
```python
# Male
LBM = 1.1×weight - 128×(weight/height)²

# Female
LBM = 1.07×weight - 148×(weight/height)²
```

**Usage**:
```python
from src.models.pharmacokinetics.minto_model import MintoModel

model = MintoModel(age=45, weight=70, height=170, gender='M')
model.set_infusion_rate(0.5)  # μg/kg/min
model.step(dt=5.0)  # 5 seconds
ce = model.Ce  # Effect-site concentration (ng/mL)
```

---

## Pharmacodynamic Models

### BIS Prediction (Single Drug)

**Hill Equation** (Sigmoid Emax model):

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

---

### Greco Response Surface Model (Dual Drug)

**Location**: `src/models/pharmacodynamics/interaction_model.py`

**Type**: Response surface model with synergistic interaction

**Formulation** (CBIM Paper 32):

```
BIS = E0 - Emax × U / (1 + U)

where:
U = U_ppf + U_rftn + α × U_ppf × U_rftn

U_ppf = (Ce_ppf / C50_ppf)^γ₁
U_rftn = (Ce_rftn / C50_rftn)^γ₂

Parameters:
- α: Interaction parameter
  α = 0: Additive
  α > 0: Synergistic (combined > sum)
  α < 0: Antagonistic (combined < sum)
- Typical: α = 1.2 (synergistic)
```

**Interaction Types**:

1. **Additive** (α = 0):
   - Effect = Effect_drug1 + Effect_drug2
   - No interaction

2. **Synergistic** (α > 0):
   - Combined effect > sum of individual effects
   - Lower doses of each drug needed
   - Clinically observed for propofol-remifentanil

3. **Antagonistic** (α < 0):
   - Combined effect < sum of individual effects
   - Higher doses needed

**Isobol**: Curve of equal effect (constant BIS)
- Shows all (Ce_ppf, Ce_rftn) pairs for target BIS
- Useful for dose optimization

**Usage**:
```python
from src.models.pharmacodynamics.interaction_model import GrecoInteractionModel

model = GrecoInteractionModel(alpha=1.2)  # Synergistic
bis = model.compute_bis(ce_propofol=3.0, ce_remifentanil=2.0)
```

---

## Quantum Models

### Variational Quantum Circuit (VQC)

**Location**: `src/models/quantum_layers.py`

**Framework**: PennyLane with PyTorch interface

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

**Gradient Computation**:
- Method: `diff_method="backprop"`
- Automatic differentiation through PennyLane
- Efficient parameter updates

**Configuration**:
```python
{
    'n_qubits': 2,
    'n_layers': 3,
    'device': 'default.qubit'
}
```

**Typical Setup**:
- 2 qubits → 2 quantum features
- 3 layers → 6 trainable parameters
- Total: ~6-12 quantum parameters

---

### Quantum Policy Network

**Location**: `src/agents/quantum_agent.py`

**Architecture**:

```
Observation (medical time series)
       ↓
Encoder: LSTM/Transformer (trainable)
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

**Input Features** (per timestep):
- BIS (0-100)
- Propofol Ce (μg/mL)
- Remifentanil Ce (ng/mL) [dual drug]
- Infusion rates
- Time elapsed

**Output**:
- Action: Δ infusion rate (mL/h or μg/kg/min)
- Range: Typically [-50, +50] mL/h

**Training**:
- Optimizer: Adam (lr=3e-4)
- Separate encoder optimizer (lr=1e-3)
- Gradient clipping: max_norm=1.0

---

### Quantum Value Network

**Location**: `src/agents/quantum_agent.py`

**Architecture** (similar to policy):

```
Observation
       ↓
Encoder: LSTM/Transformer
       ↓
Quantum Layer: VQC
       ↓
Output Layer: Linear
  - Map to Q-value
  - Value: Expected return
```

**Output**:
- Q(s, a): State-action value
- Range: [-∞, +∞] (unbounded)

**Target Network**:
- Soft update: τ = 0.005
- Stabilizes training (DDPG)

---

## Classical Models

### LSTM Encoder

**Location**: `src/models/encoders.py`

**Architecture**:

```
Input: (batch, seq_len, n_features)
       ↓
LSTM: 
  - Hidden dim: 128
  - Num layers: 2
  - Bidirectional: False
  - Dropout: 0.1
       ↓
Output: Last hidden state (batch, 128)
```

**Advantages**:
- Captures long-term dependencies
- Handles variable-length sequences
- Good for medical time series

**Usage**:
```python
from src.models.encoders import LSTMEncoder

encoder = LSTMEncoder(input_dim=5, hidden_dim=128, num_layers=2)
h = encoder(obs)  # (batch, 128)
```

---

### Transformer Encoder

**Location**: `src.models/encoders.py`

**Architecture**:

```
Input: (batch, seq_len, n_features)
       ↓
Positional Encoding
       ↓
Transformer Encoder:
  - Num layers: 2
  - Num heads: 4
  - Hidden dim: 128
  - FFN dim: 512
  - Dropout: 0.1
       ↓
Output: Mean pooling (batch, 128)
```

**Advantages**:
- Parallel processing (faster)
- Better long-range dependencies
- Attention mechanism
- State-of-the-art for sequences

**Attention**:
- Multi-head self-attention
- Learns temporal relationships
- Interpretable attention weights

---

### MLP (Classical Baseline)

**Location**: `src/agents/classical_agent.py`

**Architecture**:

```
Input: Flattened observation
       ↓
FC1: Linear(input_dim, 256) + ReLU
       ↓
FC2: Linear(256, 256) + ReLU
       ↓
FC3: Linear(256, 128) + ReLU
       ↓
Output: Linear(128, output_dim)
```

**Purpose**:
- Baseline for comparison
- No quantum or encoder complexity
- Fast training and inference

---

## Agent Architecture

### Quantum DDPG Agent

**Location**: `src/agents/quantum_agent.py`

**Algorithm**: Deep Deterministic Policy Gradient (off-policy)

**Components**:

```
Actor (Policy):
  Encoder → VQC → Output
  - Deterministic: π(s) → a
  
Critic (Q-function):
  Encoder → VQC → Output
  - Q(s, a) → value
  
Target Networks:
  - Actor target
  - Critic target
  - Soft update (τ=0.005)
  
Replay Buffer:
  - Size: 100,000
  - Batch: 256
  - Random sampling
```

**Training Loop**:

1. **Collect experience**: Execute π(s) + noise
2. **Store**: (s, a, r, s', done) → buffer
3. **Sample batch**: Random from buffer
4. **Compute targets**: y = r + γ × Q'(s', π'(s'))
5. **Update critic**: Minimize (Q(s,a) - y)²
6. **Update actor**: Maximize Q(s, π(s))
7. **Update targets**: θ' ← τθ + (1-τ)θ'

**Loss Functions**:
```python
# Critic loss
critic_loss = MSE(Q(s, a), y)

# Actor loss (negative Q-value)
actor_loss = -Q(s, π(s)).mean()
```

**Hyperparameters**:
```yaml
lr_actor: 3e-4
lr_critic: 1e-3
lr_encoder: 1e-3
gamma: 0.99
tau: 0.005
batch_size: 256
buffer_size: 100000
```

---

### Quantum PPO Agent

**Location**: `src/agents/quantum_ppo_agent.py`

**Algorithm**: Proximal Policy Optimization (on-policy)

**Components**:

```
Actor (Stochastic Policy):
  Encoder → VQC → μ (mean)
  Log_std: Trainable parameter
  - Stochastic: π(a|s) ~ N(μ, σ²)
  
Critic (Value Function):
  Encoder → VQC → V(s)
  - State value (no action input)
  
GAE (Advantage Estimation):
  - λ = 0.95
  - Reduces variance
```

**Training Loop**:

1. **Collect trajectories**: Run π for T steps
2. **Compute GAE**: A^GAE(s,a) using TD errors
3. **Compute returns**: R = A + V(s)
4. **Multiple epochs**: K=10 updates per batch
5. **Update policy**: Clipped objective
6. **Update value**: MSE(V(s), R)

**Loss Functions** (Formulations 45-48):

```python
# Policy loss (clipped objective)
ratio = π_new(a|s) / π_old(a|s)
surrogate1 = ratio × A
surrogate2 = clip(ratio, 1-ε, 1+ε) × A
policy_loss = -min(surrogate1, surrogate2).mean()

# Value loss
value_loss = MSE(V(s), R)

# Entropy bonus (exploration)
entropy_loss = -entropy(π).mean()

# Total loss
loss = policy_loss + c1×value_loss + c2×entropy_loss
```

**GAE Formulation** (CBIM Paper 46):

```
δ_t = r_t + γ×V(s_{t+1}) - V(s_t)
A^GAE(s_t, a_t) = Σ_{l=0}^∞ (γλ)^l × δ_{t+l}

where:
- γ = 0.99 (discount)
- λ = 0.95 (GAE parameter)
```

**Hyperparameters**:
```yaml
lr: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
c1_value: 0.5
c2_entropy: 0.01
n_epochs: 10
batch_size: 256
n_steps: 2048
```

---

### Classical DDPG Agent

**Location**: `src/agents/classical_agent.py`

**Differences from Quantum**:
- No VQC layer
- MLP architecture throughout
- Faster training/inference
- Baseline for comparison

**Architecture**:
```
Actor: MLP(obs_dim, 256, 256, action_dim)
Critic: MLP(obs_dim + action_dim, 256, 256, 1)
```

---

### Classical PPO Agent

**Location**: `src/agents/ppo_agent.py`

**Differences from Quantum**:
- No VQC layer
- MLP for policy/value networks
- Otherwise identical to quantum PPO

---

## Implementation Details

### Training Scripts

**Online Training**:
- `experiments/train_quantum.py`: Quantum DDPG online
- `experiments/train_classical.py`: Classical DDPG online
- `experiments/train_ppo.py`: PPO online (quantum/classical)

**Hybrid Training** (Offline → Online):
- `experiments/train_hybrid.py`: DDPG hybrid
- `experiments/train_hybrid_ppo.py`: PPO hybrid
- Stage 1: Behavioral cloning on VitalDB
- Stage 2: RL fine-tuning on simulator

**Offline Training**:
- `experiments/train_offline.py`: Pure BC on VitalDB
- No simulator interaction

**Comparison**:
- `experiments/compare_quantum_vs_classical.py`: DDPG comparison
- `experiments/compare_ddpg_vs_ppo.py`: Algorithm comparison
- Statistical tests (t-test, Wilcoxon)

---

### Data Pipeline

**VitalDB Preprocessing**:
```
data/prepare_vitaldb.py
  ↓
Filtering: Quality checks
  ↓
Normalization: Zero mean, unit variance
  ↓
Splitting: 80% train, 10% val, 10% test
  ↓
Caching: Save to pickle
  ↓
data/offline_dataset/vitaldb_processed.pkl
```

**Loading** (5-10 min → 10 sec):
```python
from src.data.vitaldb_loader import VitalDBLoader

loader = VitalDBLoader(data_dir='data/offline_dataset')
train_data = loader.load_train()
```

---

### Evaluation Metrics

**Clinical Metrics** (Formulations 50-52):

```python
# MDPE: Median Performance Error
PE = 100 × (BIS - target) / target
MDPE = median(PE)

# MDAPE: Median Absolute Performance Error
MDAPE = median(|PE|)

# Wobble: Median Absolute Deviation
Wobble = median(|PE - MDPE|)
```

**Operational Metrics**:
- Time in target (TIT): % time in [40, 60]
- Induction time: Time to reach BIS ≤ 60
- Recovery time: Time from stop to BIS ≥ 80
- Overshoot: max|BIS - target|

**RL Metrics**:
- Episode return: Σ rewards
- Mean reward per step
- TD error
- Policy entropy (PPO)

---

### Visualization

**Training Curves**:
- `src/visualization/training_curves.py`
- Smoothing: Savitzky-Golay filter
- Multi-experiment comparison
- TensorBoard export

**Plots**:
1. BC loss (stage 1)
2. RL loss (stage 2)
3. Reward curves
4. MDAPE over time
5. Policy/value loss (PPO)
6. Entropy decay (PPO)

**Usage**:
```python
from src.visualization.training_curves import plot_training_curves

plot_training_curves(
    log_dir='logs/hybrid_20251221_030645',
    save_dir='logs/hybrid_20251221_030645/figures'
)
```

---

### Hyperparameter Configuration

**Location**: `config/hyperparameters.yaml`

**Categories**:
1. **Environment**: BIS target, noise, episode length
2. **Algorithm**: DDPG/PPO specific settings
3. **Network**: Hidden dims, num layers, activation
4. **Training**: Learning rates, batch size, epochs
5. **Quantum**: n_qubits, n_layers, device

**Example**:
```yaml
environment:
  target_bis: 50
  noise_std: 2.0
  max_steps: 500

ddpg:
  lr_actor: 3e-4
  lr_critic: 1e-3
  gamma: 0.99
  tau: 0.005

ppo:
  lr: 3e-4
  clip_epsilon: 0.2
  gae_lambda: 0.95
  n_epochs: 10

quantum:
  n_qubits: 2
  n_layers: 3
  encoder_type: 'lstm'
```

---

## Model Comparison Summary

| Feature | Quantum DDPG | Classical DDPG | Quantum PPO | Classical PPO |
|---------|--------------|----------------|-------------|---------------|
| **Policy** | VQC | MLP | VQC | MLP |
| **Value** | VQC | MLP | VQC | MLP |
| **Algorithm** | Off-policy | Off-policy | On-policy | On-policy |
| **Sample Efficiency** | High | High | Low | Low |
| **Stability** | Medium | High | High | High |
| **Parameters** | ~1000 | ~10000 | ~1000 | ~10000 |
| **Training Time** | Slow | Fast | Medium | Fast |
| **Exploration** | OU noise | OU noise | Stochastic | Stochastic |
| **Buffer** | Yes | Yes | No | No |

**Key Differences**:

1. **Quantum vs Classical**:
   - Quantum: Fewer parameters, potential quantum advantage
   - Classical: More parameters, proven performance

2. **DDPG vs PPO**:
   - DDPG: Sample efficient, unstable, off-policy
   - PPO: Stable, less efficient, on-policy

3. **Encoder**:
   - LSTM: Sequential, good for long-term
   - Transformer: Parallel, attention mechanism

---

## References

1. **CBIM Paper**: Formulations 1-52 for PK/PD models
2. **Schnider TW et al.**: Propofol PK model (1998)
3. **Minto CF et al.**: Remifentanil PK model (1997)
4. **Greco WR et al.**: Drug interaction model (1995)
5. **Lillicrap TP et al.**: DDPG algorithm (2015)
6. **Schulman J et al.**: PPO algorithm (2017)

---

## Code Structure

```
src/
├── agents/
│   ├── quantum_agent.py         # Quantum DDPG
│   ├── quantum_ppo_agent.py     # Quantum PPO
│   ├── classical_agent.py       # Classical DDPG
│   └── ppo_agent.py             # Classical PPO
├── models/
│   ├── quantum_layers.py        # VQC implementation
│   ├── encoders.py              # LSTM/Transformer
│   ├── pharmacokinetics/
│   │   ├── schnider_model.py    # Propofol PK
│   │   └── minto_model.py       # Remifentanil PK
│   └── pharmacodynamics/
│       └── interaction_model.py # Greco interaction
├── environment/
│   ├── propofol_env.py          # Single drug env
│   └── dual_drug_env.py         # Dual drug env
├── data/
│   └── vitaldb_loader.py        # Data loading
├── visualization/
│   └── training_curves.py       # Plotting
└── utils/
    ├── replay_buffer.py         # Experience replay
    └── metrics.py               # Evaluation metrics

experiments/
├── train_quantum.py             # Quantum DDPG training
├── train_classical.py           # Classical DDPG training
├── train_ppo.py                 # PPO training
├── train_hybrid.py              # DDPG hybrid
├── train_hybrid_ppo.py          # PPO hybrid
├── compare_quantum_vs_classical.py
└── compare_ddpg_vs_ppo.py

data/
└── prepare_vitaldb.py           # Dataset preprocessing

config/
└── hyperparameters.yaml         # Configuration
```

---

**Last Updated**: 2024-12-23

**Contributors**: Quantum RL Anesthesia Research Team
