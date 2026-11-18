"""Extract hash-verified saved comparisons; never train a model or run ED."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SUPPORT = Path(__file__).resolve().parent
ROOT = SUPPORT.parents[3]
RESULTS = ROOT / "SPRING_Haldane_FQHE/results"
OUTPUT = SUPPORT / "constraint_comparison_20261116.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


inputs = {}


def read(path):
    path = Path(path).resolve()
    inputs[str(path)] = sha(path)
    return json.loads(path.read_text())


def verify_record(record):
    for key in ("checkpoint", "normalized_state"):
        path = Path(record[key])
        digest = sha(path)
        assert digest == record[key + "_sha256"], path
        inputs[str(path)] = digest
    assert record["energy_V1"] >= record["ED_energy0_V1"] - 1e-10
    assert 0 <= record["ground_subspace_fidelity"] <= record["physical_L0_probability"] + 1e-10


def summary(study):
    return read(RESULTS / study / "independent_evaluation/evaluation_summary.json")


def config(run):
    result = read(Path(run) / "config.json")
    assert result["ne"] == 7 and result["flux"] == 18 and not result["edge_potential"]
    assert result["v3"] == .8 and result["v5"] == .6
    return result


def main():
    matched = summary("ne7_angular_time01_matched1k_20261116")
    assert matched["status"] == "passed"
    assert matched["matched_source_and_initial_RNGs_verified"]
    assert matched["matched_candidate_and_Lplus_streams_verified"]
    arms = matched["arms"]
    configs, matched_records = {}, {}
    for name, arm in arms.items():
        configs[name] = config(arm["run"])
        path = (RESULTS / "ne7_angular_time01_matched1k_20261116"
                / "independent_evaluation/arms" / name / "exact_evaluations.json")
        records = read(path)["records"]
        assert [r["step"] for r in records] == list(range(0, 1001, 100))
        for record in records:
            verify_record(record)
        matched_records[name] = records

    c0, c1 = configs.values()
    allowed_differences = {"arm", "output", "input_sha256", "kappa"}
    common_differences = [k for k in sorted(c0.keys() & c1.keys())
                          if c0[k] != c1[k] and k not in allowed_differences]
    assert not common_differences, common_differences
    first0, first1 = [arm["first"] for arm in arms.values()]
    with np.load(first0["checkpoint"]) as a, np.load(first1["checkpoint"]) as b:
        compared_fields = ["parameters", "previous_direction", "rng_json", "probes",
                           "policy_anchors", "policy_history_bits",
                           "policy_history_residuals", "policy_history_ages"]
        for key in compared_fields:
            assert np.array_equal(a[key], b[key]), key

    baseline = summary("ne7_v5_continue10k_20261116")
    assert baseline["status"] == "passed"
    baseline_config = config(baseline["run"])
    assert "kappa" not in baseline_config or baseline_config["kappa"] == 0
    for record in baseline["checkpoint_evaluations"]:
        verify_record(record)

    segment_names = [
        "ne7_angular_time01_batch128_continue1k_20261116",
        "ne7_angular_time01_batch128_continue5k_20261116",
        "ne7_angular_time01_batch128_continue10k_20261116",
        "ne7_angular_time01_batch128_second10k_recovery1000_20261116",
        "ne7_angular_time01_batch128_third10k_20261116",
    ]
    long_records, segment_metadata = [], []
    previous_endpoint = None
    latest = None
    for name in segment_names:
        latest = data = summary(name)
        assert data["status"] == "passed"
        records = data.get("checkpoint_evaluations", data.get("combined_logical_records"))
        assert records
        if previous_endpoint is not None:
            with np.load(previous_endpoint["checkpoint"]) as a, np.load(records[0]["checkpoint"]) as b:
                assert np.array_equal(a["parameters"], b["parameters"])
            assert previous_endpoint["absolute_step"] == records[0]["absolute_step"]
            for key in ("energy_V1", "ground_subspace_fidelity", "physical_L0_probability", "physical_L2"):
                assert abs(previous_endpoint[key] - records[0][key]) < 1e-10, key
        for record in records:
            verify_record(record)
            out = dict(record, slide_segment=name,
                       updates_since_B128_start=record["absolute_step"] - 18000)
            if not long_records or out["absolute_step"] > long_records[-1]["absolute_step"]:
                long_records.append(out)
        segment_metadata.append(dict(study=name, absolute_start=records[0]["absolute_step"],
                                     absolute_end=records[-1]["absolute_step"],
                                     recovered="recovery1000" in name))
        previous_endpoint = records[-1]
    assert long_records[0]["absolute_step"] == 18000
    assert long_records[-1]["absolute_step"] == 54000
    assert latest is not None
    for record in (latest["first"], latest["last"]):
        verify_record(record)
    latest_config = config(Path(latest["last"]["checkpoint"]).parents[1])
    for field in ("architecture", "parameter_count", "head_count", "parameter_names", "parameter_shapes"):
        assert latest_config[field] == c0[field], field

    result = dict(
        status="passed", extraction_script=str(Path(__file__).resolve()),
        extraction_script_sha256=sha(__file__), no_training_or_ED_performed=True,
        system=dict(Ne=7, flux=18, orbitals=19, two_Lz=0, dimension=1656,
                    edge_potential=False, v1=1., v3=.8, v5=.6,
                    ED_energy0_V1=6.2447682602150385,
                    benchmark=latest["reused_ED_benchmark"]),
        matched_1k=dict(source_checkpoint=matched["source_checkpoint"],
                        source_checkpoint_sha256=matched["source_checkpoint_sha256"],
                        matched_first_checkpoint_fields=compared_fields,
                        same_common_config_except_kappa=True,
                        common_settings={k: c0[k] for k in ("architecture", "parameter_count", "batch", "pool_size", "eta", "mu", "damping", "cg_maxiter", "cg_tol", "parameter_radius", "state_radius", "gamma", "gamma_aux", "head_metric_weight", "h_draws", "boundary_reset")},
                        arms={name: dict(first=arm["first"], last=arm["last"],
                                          completion=arm["completion"], records=matched_records[name],
                                          config=configs[name]) for name, arm in arms.items()},
                        finding="kappa=.1 improves L0 weight and fidelity versus kappa=0 after 1k, but physical energy is worse; neither endpoint beats source physical energy",
                        one_seed=True),
        historical_no_L2=dict(first=baseline["first"], last=baseline["last"],
                              records=baseline["checkpoint_evaluations"],
                              config=baseline_config, completion=baseline["completion"]),
        long_B128_L2=dict(records=long_records, segments=segment_metadata,
                          latest=latest["last"], latest_config=latest_config,
                          total_B128_attempts=36000,
                          source_absolute_update=18000, final_absolute_update=54000,
                          CG_limit_changed_at_absolute_update=43000,
                          limitation="B128/pool512, longer training and solver recovery differ from B32/pool128 no-L2 control; not a causal L2-only or equal-time comparison"),
        recommended_main_comparison=dict(no_L2=arms["kappa_0_seed71"]["last"],
                                          latest_L2=latest["last"],
                                          label="same Hamiltonian, historical checkpoints, unequal budgets"),
        input_sha256=inputs,
    )
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="passed", output=str(OUTPUT), verified_inputs=len(inputs),
                          matched_control_energy=arms["kappa_0_seed71"]["last"]["energy_V1"],
                          matched_L2_energy=arms["kappa_0p1_seed71"]["last"]["energy_V1"],
                          latest_energy=latest["last"]["energy_V1"],
                          long_records=len(long_records)), indent=2))


if __name__ == "__main__":
    main()
