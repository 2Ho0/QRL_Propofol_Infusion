# Hybrid Quantum-Classical Reinforcement Learning for Anesthesia Drug Infusion

Code for the paper *NISQ-Constrained Hybrid Quantum-Classical Reinforcement Learning for Multi-Drug Infusion Control in Anesthesia*.

## What it does

During general anesthesia, the depth of anesthesia is monitored with the bispectral index (BIS, 0–100; the target range is 40–60). This program learns a controller that sets the infusion rates of two drugs, propofol and remifentanil, so that BIS stays near 50.

- A PID controller computes a base dose from the measured BIS once a minute.
- A small learned policy adds a bounded correction to that dose.
- The policy reads the last 10 minutes of the patient's state with a transformer encoder.
- A three-qubit quantum circuit inside the policy is simulated with PennyLane.
- The policy is trained with TD3 reinforcement learning on simulated patients.
- The patient simulator uses standard drug models (Schnider for propofol, Minto for remifentanil, Bouillon for BIS).
- Each simulated patient has the age, sex, height and weight of one of 3,000 VitalDB surgical patients. The patients are split into 2,100 for training, 300 for validation and 600 for testing.
- The learned controller is compared with a tuned PID, a basic PID, a rule-based controller, target-controlled infusion (TCI) and classical neural-network policies.

## Quick start

Python 3.11 is required. If you have a GPU, install a CUDA build of PyTorch first (<https://pytorch.org>).

```bash
pip install -r requirements.txt
python -m pytest tests -q                   # unit tests
python tools/verify_code_fingerprint.py     # checks that the code is the code that produced the results
```

## Data

The data are not included. Download them as follows.

- **VitalDB** (<https://vitaldb.net>). The command below downloads the 3,000 cases and builds the training data (about 2 GB):

  ```bash
  python experiments/prepare_clean_retraining_data.py \
    --out data/offline_dataset/vitaldb_obs10_ctrl60_t60_subject3000_pkvalid_wtfix.pkl \
    --manifest logs/rebuild/dataset_manifest.json \
    --caseids_from logs/revision_v2/dataset_wtfix/dataset_manifest.json
  ```

  The last file lists the case ids of the training, validation and test patients.
- **NHANES** (only for the external-population test). Download `DEMO_L.xpt` and `BMX_L.xpt` of NHANES 2021–2023 from <https://wwwn.cdc.gov/nchs/nhanes/> into `data/external/nhanes_2021_2023/`.

## Training and evaluation

The command below trains every method with 10 random seeds and tests each trained controller on the 600 test patients. It took about 3 days on one GPU (NVIDIA RTX 5080).

```bash
python experiments/run_v3_campaign.py --action_prior 0.5,0.5 --gate_record README.md --jobs 2
python experiments/analyze_v3_campaign.py   # statistics of all runs
```

- `--gate_record` takes any existing notes file. The runner stores its path with the run settings.
- Finished runs are kept, so running the command again resumes the work.
- Add `--dry_run` to print the command of each training run. Run one of them to train a single controller.

Other experiments use the trained controllers:

| Script | What it does |
|---|---|
| `evaluate_noise_v2.py` | runs the quantum policies with finite shots, gate noise and readout errors |
| `evaluate_eleveld_mismatch.py` | tests on patients who follow other drug models (Eleveld) |
| `evaluate_external_cohort.py` | tests on 600 adults from NHANES |
| `run_propofol_only.py` | single-drug comparison: propofol is controlled, remifentanil follows the clinical record |
| `profile_v3_compute.py` | measures inference time, update time and memory |
| `tune_pid_patient_safe.py` | tunes the PID gains on the validation patients |
| `count_parameters.py` | counts the trainable parameters of each model |
| `gradient_variance_sweep.py` | measures the gradient variance of the circuit for several sizes |

The scripts are in `experiments/`. The `summarize_*.py` scripts and `tost_margin_sensitivity.py` compute the statistics of these experiments. Run any script with `--help` to see its options.

## Results

`results/` holds the summary tables of our runs: one row per method or per training run. Per-patient results are not included.

| Name in the code | Method |
|---|---|
| `quantum_full` | proposed hybrid quantum policy (QRL) |
| `classical_matched` | neural-network policy of the same size |
| `quantum_disabled` | QRL with the circuit replaced by fixed features |
| `classical_large` | larger neural-network policy |
| `fourier_surrogate`, `ansatz_*`, `quantum_critic` | variants of the circuit and of the critic |
| `pid_tuned`, `pid`, `guideline`, `tci` | tuned PID, basic PID, rule-based, TCI |

## Project structure

```
src/            simulator, controllers, neural networks and the quantum circuit
config/         hyperparameters
experiments/    training, evaluation and analysis scripts
tools/          code fingerprint check
protocols/      analysis plan locked before testing, and the NHANES cohort
results/        summary results
logs/           fixed inputs: tuned PID gains and the patient split
tests/          unit tests
```

The analysis plan in `protocols/campaign_locked_protocol.json` was fixed before the test patients were evaluated. It also records a fingerprint of the code that ran, and `tools/verify_code_fingerprint.py` compares a checkout with it.

## License

MIT License (see `LICENSE`). The VitalDB and NHANES data follow the terms of their providers.
