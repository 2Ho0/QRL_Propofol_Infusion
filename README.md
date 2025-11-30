# Quantum Reinforcement Learning for Propofol Infusion Control

A hybrid Quantum-Classical Reinforcement Learning system for closed-loop BIS-guided propofol anesthesia control, based on the CBIM (Closed-loop BIS-guided Infusion Model) paper with Quantum enhancement using PennyLane.

## 🎯 Overview

This project implements a **Quantum Deep Deterministic Policy Gradient (QDDPG)** agent for automated propofol infusion control during anesthesia. The system uses a 2-qubit Variational Quantum Circuit (VQC) as the policy network to determine optimal propofol dosing to maintain the patient's BIS (Bispectral Index) at the target level.

### Key Features

- **Quantum Policy Network**: 2-qubit VQC with angle encoding and variational layers
- **Schnider PK/PD Model**: Three-compartment pharmacokinetic model with effect-site dynamics
- **Hill Sigmoid BIS Model**: Pharmacodynamic model for BIS prediction
- **Clinical Metrics**: MDPE, MDAPE, Wobble, Time-in-Target evaluation
- **Gymnasium Environment**: Standard RL interface for training and evaluation

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Quantum RL Agent (QDDPG)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌─────────────────────────────────┐       │
│  │   State      │     │     Quantum Policy (Actor)       │       │
│  │   Encoder    │────▶│  ┌───────────────────────────┐  │       │
│  │  (Classical) │     │  │    2-Qubit VQC            │  │       │
│  └──────────────┘     │  │  • Angle Encoding         │  │       │
│                       │  │  • RY-RZ Rotations        │  │──▶ Action
│                       │  │  • CNOT Entanglement      │  │   (Dose)
│                       │  │  • 4 Variational Layers   │  │       │
│                       │  └───────────────────────────┘  │       │
│                       └─────────────────────────────────┘       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Twin Critic Networks (Classical)            │    │
│  │   Q1(s,a) & Q2(s,a) → Value Estimation                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Propofol Environment                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Schnider PK/PD Patient Model               │    │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐               │    │
│  │   │Central  │  │Shallow  │  │Deep     │               │    │
│  │   │Compart. │◄─┤Periph.  │◄─┤Periph.  │               │    │
│  │   │   C1    │  │   C2    │  │   C3    │               │    │
│  │   └────┬────┘  └─────────┘  └─────────┘               │    │
│  │        │                                               │    │
│  │        ▼                                               │    │
│  │   ┌─────────┐       ┌─────────────────────────┐      │    │
│  │   │Effect   │──────▶│ Hill Sigmoid Emax Model │──▶ BIS │    │
│  │   │Site Ce  │       │ BIS = E0 - Emax*f(Ce)   │      │    │
│  │   └─────────┘       └─────────────────────────┘      │    │
│  │                                                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
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
│   └── hyperparameters.yaml      # Configuration file
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── patient_simulator.py  # Schnider PK/PD model
│   │   └── propofol_env.py       # Gymnasium environment
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vqc.py                # Variational Quantum Circuit
│   │   └── networks.py           # Classical neural networks
│   ├── agents/
│   │   ├── __init__.py
│   │   └── quantum_agent.py      # Quantum DDPG agent
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # Performance metrics (MDPE, MDAPE, etc.)
│       └── visualization.py      # Plotting utilities
├── experiments/
│   └── train_quantum.py          # Training script
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
# Train with default configuration
python experiments/train_quantum.py

# Train with custom settings
python experiments/train_quantum.py --episodes 500 --seed 42

# Resume from checkpoint
python experiments/train_quantum.py --resume logs/experiment/checkpoints/checkpoint_500.pt
```

### Configuration

Edit `config/hyperparameters.yaml` to customize:

```yaml
# Quantum Circuit
quantum:
  n_qubits: 2
  n_layers: 4
  
# Environment
environment:
  bis_target: 50
  bis_min: 40
  bis_max: 60
  dose_max: 200.0

# Training
training:
  total_episodes: 1000
  batch_size: 64
  gamma: 0.99
```

## 📈 Performance Metrics

Following the CBIM paper, we evaluate using clinical anesthesia metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| **MDPE** | Median Performance Error (bias) | |MDPE| < 10% |
| **MDAPE** | Median Absolute Performance Error (accuracy) | MDAPE < 20% |
| **Wobble** | Intra-individual variability | Lower is better |
| **Time in Target** | % time BIS in 40-60 range | > 80% |

## 🔬 Mathematical Formulation

### PK Model (Schnider)
$$\frac{dC_1}{dt} = \frac{u(t)}{V_1} - (k_{10} + k_{12} + k_{13})C_1 + k_{21}\frac{V_2}{V_1}C_2 + k_{31}\frac{V_3}{V_1}C_3$$

### Effect-Site Equilibration
$$\frac{dC_e}{dt} = k_{e0}(C_1 - C_e)$$

### BIS Prediction (Hill Model)
$$BIS = E_0 - E_{max} \cdot \frac{C_e^{\gamma}}{C_e^{\gamma} + EC_{50}^{\gamma}}$$

### Reward Function
$$r_t = -\alpha \cdot PE_t^2 - \beta \cdot u_t - \gamma \cdot |\Delta u_t| + \text{safety penalties}$$

Where $PE_t = \frac{BIS_t - BIS_{target}}{BIS_{target}} \times 100$

## 🔧 Dependencies

- Python >= 3.9
- PennyLane >= 0.33.0
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy, SciPy, Matplotlib

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
