"""run_pilot options for the v3 method (D14-D17)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from run_pilot import (  # noqa: E402
    apply_quantum_action_prior, build_parser, check_residual_args, env_kwargs_from_run_config,
    learned_env_kwargs, run_config_for_checkpoint)

PID_PATH = "logs/revision_v2/pid_tuning/selected_pid.json"


def _args(*extra):
    return build_parser().parse_args(["--out", "unused"] + list(extra))


def test_defaults_keep_the_plain_environment_and_the_v2_reward():
    assert learned_env_kwargs(_args()) == {"reward_w3": 0.1, "reward_w_smooth": 2.0}


def test_residual_options_load_the_tuned_pid():
    kwargs = learned_env_kwargs(_args(
        "--residual_pid", PID_PATH, "--residual_range", "4.0,0.3",
        "--reward_w3", "0.03", "--reward_w_smooth", "0.5"))
    assert kwargs["residual_pid"] == json.loads(Path(PID_PATH).read_text())["parameters"]
    assert kwargs["residual_range"] == (4.0, 0.3)
    assert (kwargs["reward_w3"], kwargs["reward_w_smooth"]) == (0.03, 0.5)


def test_a_pid_file_without_parameters_is_refused(tmp_path):
    bad = tmp_path / "pid.json"
    bad.write_text(json.dumps({"kp_ppf": 0.3}))
    with pytest.raises(ValueError, match="parameters"):
        learned_env_kwargs(_args("--residual_pid", str(bad)))


@pytest.mark.parametrize("extra", [
    ["--online_bc_weight", "0"],                                       # stage 1 would imitate
    ["--skip_offline", "--online_bc_weight", "1.0"],                   # online imitation
    ["--skip_offline", "--online_bc_weight", "0", "--resume_stage1", "x.pt"],
])
def test_residual_mode_refuses_imitation_of_clinician_doses(extra):
    with pytest.raises(SystemExit, match="residual"):
        check_residual_args(_args("--residual_pid", PID_PATH, *extra))


def test_residual_mode_accepts_online_only_training():
    check_residual_args(_args("--residual_pid", PID_PATH, "--skip_offline",
                              "--online_bc_weight", "0"))


def test_quantum_action_prior_replaces_the_configured_value():
    config = {"quantum": {"action_prior": [0.5, 0.075]}}
    apply_quantum_action_prior(config, "0.5,0.5")
    assert config["quantum"]["action_prior"] == [0.5, 0.5]


def test_config_keeps_the_configured_quantum_prior():
    config = {"quantum": {"action_prior": [0.5, 0.075]}}
    apply_quantum_action_prior(config, "config")
    assert config["quantum"]["action_prior"] == [0.5, 0.075]


@pytest.mark.parametrize("bad", ["0.5", "0.5,1.0", "0,0.5"])
def test_quantum_action_prior_needs_two_values_inside_zero_one(bad):
    with pytest.raises(ValueError, match="quantum_action_prior"):
        apply_quantum_action_prior({"quantum": {}}, bad)


V3 = ["--residual_pid", PID_PATH, "--skip_offline", "--online_bc_weight", "0"]


def test_the_residual_range_defaults_to_the_v3_range():
    assert learned_env_kwargs(_args(*V3))["residual_range"] == (4.0, 0.05)


def test_a_residual_range_without_a_pid_is_refused():
    with pytest.raises(SystemExit, match="residual_range"):
        check_residual_args(_args("--residual_range", "4.0,0.3"))


@pytest.mark.parametrize("bad", ["4.0", "4.0,-0.05", "4.0,nan", "a,b", "4.0,0.05,1"])
def test_a_malformed_residual_range_is_refused_before_any_work(bad):
    with pytest.raises(SystemExit, match="residual_range"):
        check_residual_args(_args(*V3, "--residual_range", bad))


def test_residual_mode_refuses_action_smoothing():
    with pytest.raises(SystemExit, match="smoothing"):
        check_residual_args(_args(*V3, "--smoothing_beta", "0.5"))


def test_env_kwargs_from_run_config_rebuilds_the_learned_environment(tmp_path):
    args = _args(*V3, "--residual_range", "4.0,0.3", "--reward_w3", "0.03",
                 "--reward_w_smooth", "0.5")
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps(vars(args)))
    assert env_kwargs_from_run_config(path) == learned_env_kwargs(args)


def test_a_run_config_from_before_v3_gives_the_plain_environment(tmp_path):
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps({"out": "x", "seed": 3, "smoothing_beta": 0.0}))
    assert env_kwargs_from_run_config(path) == {"reward_w3": 0.1, "reward_w_smooth": 2.0}


def test_the_run_config_of_a_checkpoint_is_found_in_its_run_folder(tmp_path):
    checkpoint = tmp_path / "quantum_full" / "stage2" / "best_mdape.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    (tmp_path / "run_config.json").write_text("{}")
    assert run_config_for_checkpoint(checkpoint) == (tmp_path / "run_config.json").resolve()


def test_a_checkpoint_without_its_run_config_is_refused(tmp_path):
    checkpoint = tmp_path / "quantum_full" / "stage2" / "best_mdape.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="run_config"):
        run_config_for_checkpoint(checkpoint)


def test_quantum_actor_lr_sets_every_quantum_family_rate():
    from run_pilot import apply_actor_learning_rates
    config = {"training": {"actor_lr": 3e-4, "quantum_actor_lr": 1e-3, "vqc_lr": 1e-3}}
    apply_actor_learning_rates(config, _args("--quantum_actor_lr", "0.0001",
                                             "--classical_actor_lr", "0.001"))
    assert config["training"]["quantum_actor_lr"] == 1e-4
    assert config["training"]["vqc_lr"] == 1e-4
    assert config["training"]["actor_lr"] == 1e-3


def test_learning_rates_default_to_the_config():
    from run_pilot import apply_actor_learning_rates
    config = {"training": {"actor_lr": 3e-4, "quantum_actor_lr": 1e-3, "vqc_lr": 1e-3}}
    apply_actor_learning_rates(config, _args())
    assert config["training"] == {"actor_lr": 3e-4, "quantum_actor_lr": 1e-3, "vqc_lr": 1e-3}
