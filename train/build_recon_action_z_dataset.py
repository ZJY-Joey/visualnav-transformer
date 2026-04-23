import argparse
import io
import os
import pickle

import h5py
import numpy as np
import tqdm
import yaml
from PIL import Image


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_lengths(file_path: str, expected_len: int, arrays: dict[str, np.ndarray]) -> None:
    for name, array in arrays.items():
        if len(array) != expected_len:
            raise ValueError(
                f"{file_path}: length mismatch for {name}, expected {expected_len}, got {len(array)}"
            )


def compute_velocity_error(
    cmd_value: float,
    measured_value: float,
    threshold: float,
    epsilon: float,
) -> float:
    abs_error = np.abs(cmd_value - measured_value)
    if np.abs(cmd_value) < threshold:
        return float(abs_error)
    return float(abs_error / (np.abs(cmd_value) + epsilon))


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def compute_motion_error(
    actual_value: float,
    expected_value: float,
    threshold: float,
    epsilon: float,
) -> float:
    abs_error = np.abs(actual_value - expected_value)
    if np.abs(expected_value) < threshold:
        return float(abs_error)
    return float(abs_error / (np.abs(expected_value) + epsilon))


def clip_value(value: float, max_value: float | None) -> float:
    if max_value is None:
        return float(value)
    return float(np.clip(value, 0.0, max_value))


def compute_step_z(
    cmd_linear_t: float,
    cmd_angular_t: float,
    jackal_linear_tp1: float,
    jackal_angular_tp1: float,
    position_t: np.ndarray,
    position_tp1: np.ndarray,
    yaw_t: float,
    yaw_tp1: float,
    weights: dict,
    thresholds: dict,
    constants: dict,
    epsilons: dict,
    clips: dict,
) -> tuple[float, float, float, float, float]:
    vel_term = weights["linear_velocity"] * compute_velocity_error(
        cmd_value=cmd_linear_t,
        measured_value=jackal_linear_tp1,
        threshold=float(thresholds["linear_velocity_cmd_min"]),
        epsilon=float(epsilons["velocity"]),
    )
    vel_term = clip_value(vel_term, clips.get("vel_term_max"))

    angvel_term = weights["angular_velocity"] * compute_velocity_error(
        cmd_value=cmd_angular_t,
        measured_value=jackal_angular_tp1,
        threshold=float(thresholds["angular_velocity_cmd_min"]),
        epsilon=float(epsilons["angular_velocity"]),
    )
    angvel_term = clip_value(angvel_term, clips.get("angvel_term_max"))

    expected_delta_s = float(constants["position_command_scale"]) * np.abs(cmd_linear_t)
    actual_delta_s = float(np.linalg.norm(position_tp1 - position_t))
    pos_term = weights["position"] * compute_motion_error(
        actual_value=actual_delta_s,
        expected_value=expected_delta_s,
        threshold=float(thresholds["position_expected_min"]),
        epsilon=float(epsilons["position"]),
    )
    pos_term = clip_value(pos_term, clips.get("pos_term_max"))

    expected_delta_yaw = float(constants["yaw_command_scale"]) * np.abs(cmd_angular_t)
    actual_delta_yaw = np.abs(wrap_to_pi(yaw_tp1 - yaw_t))
    yawpos_term = weights["yaw_position"] * compute_motion_error(
        actual_value=actual_delta_yaw,
        expected_value=expected_delta_yaw,
        threshold=float(thresholds["yaw_position_expected_min"]),
        epsilon=float(epsilons["yaw_position"]),
    )
    yawpos_term = clip_value(yawpos_term, clips.get("yawpos_term_max"))

    total = clip_value(
        vel_term + angvel_term + pos_term + yawpos_term,
        clips.get("z_step_max"),
    )
    return (
        float(vel_term),
        float(angvel_term),
        float(pos_term),
        float(yawpos_term),
        total,
    )


def summarize_values(values: np.ndarray, quantiles: list[float]) -> dict:
    if len(values) == 0:
        return {"count": 0}
    summary = {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    for q in quantiles:
        summary[f"q{int(q * 100):02d}"] = float(np.quantile(values, q))
    return summary


def save_summary(summary: dict, summary_path: str) -> None:
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(np.int32)
    scores = scores.astype(np.float64)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos == 0 or neg == 0:
        return None

    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(np.int32)
    scores = scores.astype(np.float64)
    pos = int(labels.sum())
    if pos == 0 or pos == len(labels):
        return None

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def maybe_save_plot(components: dict[str, np.ndarray], plot_path: str) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    bins = 100
    titles = [
        ("vel_term", "vel_term"),
        ("angvel_term", "angvel_term"),
        ("pos_term", "pos_term"),
        ("yawpos_term", "yawpos_term"),
        ("z", "final_z"),
    ]
    for ax, (key, title) in zip(axes.flat, titles):
        values = components[key]
        ax.hist(values, bins=bins)
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    for ax in axes.flat[len(titles):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def build_samples(
    cmd_linear: np.ndarray,
    cmd_angular: np.ndarray,
    jackal_linear: np.ndarray,
    jackal_angular: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    collision_any: np.ndarray,
    collision_stuck: np.ndarray,
    collision_flipped: np.ndarray,
    collision_physical: np.ndarray,
    horizon: int,
    discount: float,
    weights: dict,
    thresholds: dict,
    constants: dict,
    epsilons: dict,
    clips: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    traj_len = len(cmd_linear)
    num_samples = traj_len - horizon
    if num_samples <= 0:
        raise ValueError(
            f"trajectory length {traj_len} is too short for horizon {horizon}"
        )

    action_seqs = np.zeros((num_samples, horizon, 2), dtype=np.float32)
    z_steps = np.zeros((num_samples, horizon), dtype=np.float32)
    z = np.zeros(num_samples, dtype=np.float32)
    vel_terms = np.zeros((num_samples, horizon), dtype=np.float32)
    angvel_terms = np.zeros((num_samples, horizon), dtype=np.float32)
    pos_terms = np.zeros((num_samples, horizon), dtype=np.float32)
    yawpos_terms = np.zeros((num_samples, horizon), dtype=np.float32)
    collision_future = np.zeros((num_samples, horizon), dtype=bool)

    for t in range(num_samples):
        running_z = 0.0
        for k in range(1, horizon + 1):
            cmd_idx = t + k - 1
            future_idx = t + k
            action_seqs[t, k - 1, 0] = cmd_linear[cmd_idx]
            action_seqs[t, k - 1, 1] = cmd_angular[cmd_idx]
            collision_future[t, k - 1] = bool(collision_any[future_idx])

            vel_term, angvel_term, pos_term, yawpos_term, step_z = compute_step_z(
                cmd_linear_t=cmd_linear[cmd_idx],
                cmd_angular_t=cmd_angular[cmd_idx],
                jackal_linear_tp1=jackal_linear[future_idx],
                jackal_angular_tp1=jackal_angular[future_idx],
                position_t=positions[cmd_idx],
                position_tp1=positions[future_idx],
                yaw_t=yaws[cmd_idx],
                yaw_tp1=yaws[future_idx],
                weights=weights,
                thresholds=thresholds,
                constants=constants,
                epsilons=epsilons,
                clips=clips,
            )
            vel_terms[t, k - 1] = vel_term
            angvel_terms[t, k - 1] = angvel_term
            pos_terms[t, k - 1] = pos_term
            yawpos_terms[t, k - 1] = yawpos_term
            z_steps[t, k - 1] = step_z
            running_z += (discount ** (k - 1)) * step_z

        z[t] = running_z

    r = collision_future.any(axis=1).astype(np.float32)

    return action_seqs, z_steps, z, vel_terms, angvel_terms, pos_terms, yawpos_terms, r


def process_file(file_path: str, output_dir: str, config: dict, export_images: bool) -> dict[str, np.ndarray]:
    traj_name = os.path.splitext(os.path.basename(file_path))[0]
    traj_folder = os.path.join(output_dir, traj_name)
    if export_images or config["dataset"]["save_dataset"]:
        os.makedirs(traj_folder, exist_ok=True)

    sampling_cfg = config["sampling"]
    thresholds_cfg = config["thresholds"]
    constants_cfg = config["constants"]
    epsilons_cfg = config["epsilons"]
    clips_cfg = config["clips"]
    weights_cfg = config["weights"]

    with h5py.File(file_path, "r") as h5_f:
        rgb_left = h5_f["images"]["rgb_left"]
        traj_len = rgb_left.shape[0]

        position_data = h5_f["jackal"]["position"][:, :2]
        yaw_data = h5_f["jackal"]["yaw"][()]
        cmd_linear = h5_f["commands"]["linear_velocity"][()]
        cmd_angular = h5_f["commands"]["angular_velocity"][()]
        jackal_linear = h5_f["jackal"]["linear_velocity"][()]
        jackal_angular = h5_f["jackal"]["angular_velocity"][()]
        collision_any = h5_f["collision"]["any"][()]
        collision_stuck = h5_f["collision"]["stuck"][()]
        collision_flipped = h5_f["collision"]["flipped"][()]
        collision_physical = h5_f["collision"]["physical"][()]

        validate_lengths(
            file_path,
            traj_len,
            {
                "jackal/position": position_data,
                "jackal/yaw": yaw_data,
                "commands/linear_velocity": cmd_linear,
                "commands/angular_velocity": cmd_angular,
                "jackal/linear_velocity": jackal_linear,
                "jackal/angular_velocity": jackal_angular,
                "collision/any": collision_any,
                "collision/stuck": collision_stuck,
                "collision/flipped": collision_flipped,
                "collision/physical": collision_physical,
            },
        )

        action_seqs, z_steps, z, vel_terms, angvel_terms, pos_terms, yawpos_terms, r = build_samples(
            cmd_linear=cmd_linear,
            cmd_angular=cmd_angular,
            jackal_linear=jackal_linear,
            jackal_angular=jackal_angular,
            positions=position_data,
            yaws=yaw_data,
            collision_any=collision_any,
            collision_stuck=collision_stuck,
            collision_flipped=collision_flipped,
            collision_physical=collision_physical,
            horizon=int(sampling_cfg["horizon"]),
            discount=float(sampling_cfg["discount"]),
            weights=weights_cfg,
            thresholds=thresholds_cfg,
            constants=constants_cfg,
            epsilons=epsilons_cfg,
            clips=clips_cfg,
        )

        start_times = np.arange(len(z), dtype=np.int32)

        traj_data = {
            "position": position_data,
            "yaw": yaw_data,
            "start_times": start_times,
            "action_seqs": action_seqs,
            "z_steps": z_steps,
            "z": z,
            "vel_terms": vel_terms,
            "angvel_terms": angvel_terms,
            "pos_terms": pos_terms,
            "yawpos_terms": yawpos_terms,
            "R": r,
            "metadata": {
                "traj_len": traj_len,
                "num_samples": len(z),
                "obs_image_name": "t.jpg where t is sample start time",
                "action_definition": (
                    "action_seqs[t, k-1] = [commands/linear_velocity[t+k-1], "
                    "commands/angular_velocity[t+k-1]]"
                ),
                "z_definition": (
                    "z_steps[t, k-1] = lambda_v * vel_error + "
                    "lambda_omega * angvel_error + "
                    "lambda_p * pos_error + "
                    "lambda_psi * yawpos_error at time t+k; "
                    "z[t] = sum_{k=1..K} discount^(k-1) * z_steps[t, k-1]"
                ),
                "r_definition": (
                    "R[t] = 1 if any collision/any[t+1:t+K] is true, else 0"
                ),
                "sampling": sampling_cfg,
                "thresholds": thresholds_cfg,
                "constants": constants_cfg,
                "epsilons": epsilons_cfg,
                "clips": clips_cfg,
                "weights": weights_cfg,
            },
        }
        if config["dataset"]["save_dataset"]:
            with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data, f)

        if export_images:
            for i in range(traj_len):
                img = Image.open(io.BytesIO(rgb_left[i]))
                img.save(os.path.join(traj_folder, f"{i}.jpg"))

    return {
        "vel_term": vel_terms.reshape(-1),
        "angvel_term": angvel_terms.reshape(-1),
        "pos_term": pos_terms.reshape(-1),
        "yawpos_term": yawpos_terms.reshape(-1),
        "z": z.reshape(-1),
        "R": r.reshape(-1),
        "trajectory_risk_ratio": np.array([float(np.mean(r))], dtype=np.float32),
    }


def print_summary(summary: dict) -> None:
    def print_stats_block(name: str, stats: dict) -> None:
        if stats.get("count", 0) == 0:
            print(f"{name}: count=0")
            return
        print(
            f"{name}: count={stats['count']} mean={stats['mean']:.6f} "
            f"std={stats['std']:.6f} min={stats['min']:.6f} max={stats['max']:.6f}"
        )
        quantile_items = [
            f"{key}={value:.6f}"
            for key, value in stats.items()
            if key.startswith("q")
        ]
        print("  " + " ".join(quantile_items))

    print("=== Risk Term Summary ===")
    for key in ("vel_term", "angvel_term", "pos_term", "yawpos_term", "z"):
        print_stats_block(key, summary[key])

    risk_ratio = summary["sample_level_risk_ratio"]
    print(
        f"sample_level_risk_ratio: risky={risk_ratio['risky_count']} "
        f"total={risk_ratio['total_count']} ratio={risk_ratio['ratio']:.6f}"
    )

    traj_ratio = summary["trajectory_level_risk_ratio"]
    print(
        f"trajectory_level_risk_ratio: count={traj_ratio['count']} "
        f"mean={traj_ratio['mean']:.6f} min={traj_ratio['min']:.6f} max={traj_ratio['max']:.6f}"
    )
    traj_quantiles = [
        f"{name}={value:.6f}"
        for name, value in traj_ratio.items()
        if name.startswith("q")
    ]
    print("  " + " ".join(traj_quantiles))

    print("=== Z By Risk Label ===")
    for key in ("z_safe", "z_risk"):
        print_stats_block(key, summary[key])

    clf = summary["z_to_R_classification"]
    print(
        "z_to_R_classification: "
        f"auroc={clf['auroc']} auprc={clf['auprc']} "
        f"positive_rate={clf['positive_rate']:.6f}"
    )

def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    report_cfg = config["report"]
    if args.stats_only:
        dataset_cfg["save_dataset"] = False
        dataset_cfg["export_images"] = False

    input_dir = args.input_dir or dataset_cfg["input_dir"]
    output_dir = args.output_dir or dataset_cfg["output_dir"]
    num_trajs = args.num_trajs if args.num_trajs is not None else int(dataset_cfg["num_trajs"])
    export_images = bool(dataset_cfg["export_images"]) and not args.stats_only

    recon_dir = os.path.join(input_dir, "recon_release")
    target_dir = recon_dir if os.path.isdir(recon_dir) else input_dir
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"input dir not found or invalid: {input_dir}"
        )

    if dataset_cfg["save_dataset"] or export_images:
        os.makedirs(output_dir, exist_ok=True)

    filenames = sorted(
        name
        for name in os.listdir(target_dir)
        if name.endswith((".h5", ".hdf5"))
    )
    if num_trajs >= 0:
        filenames = filenames[:num_trajs]

    all_components = {
        "vel_term": [],
        "angvel_term": [],
        "pos_term": [],
        "yawpos_term": [],
        "z": [],
        "R": [],
        "trajectory_risk_ratio": [],
    }
    for filename in tqdm.tqdm(filenames, desc="Trajectories processed"):
        file_path = os.path.join(target_dir, filename)
        try:
            components = process_file(
                file_path=file_path,
                output_dir=output_dir,
                config=config,
                export_images=export_images,
            )
            for key, values in components.items():
                all_components[key].append(values)
        except (OSError, KeyError, ValueError) as exc:
            print(f"Error processing {filename}. Skipping... {exc}")

    flat_components = {}
    for key, values_list in all_components.items():
        if values_list:
            flat_components[key] = np.concatenate(values_list)
        else:
            flat_components[key] = np.array([], dtype=np.float32)

    quantiles = [float(q) for q in report_cfg["quantiles"]]
    labels = flat_components["R"].astype(np.int32)
    scores = flat_components["z"].astype(np.float32)
    z_risk = scores[labels == 1]
    z_safe = scores[labels == 0]
    summary = {
        "input_dir": input_dir,
        "num_files_processed": len(filenames),
        "config": {
            "sampling": config["sampling"],
            "thresholds": config["thresholds"],
            "constants": config["constants"],
            "epsilons": config["epsilons"],
            "clips": config["clips"],
            "weights": config["weights"],
        },
        "vel_term": summarize_values(flat_components["vel_term"], quantiles),
        "angvel_term": summarize_values(flat_components["angvel_term"], quantiles),
        "pos_term": summarize_values(flat_components["pos_term"], quantiles),
        "yawpos_term": summarize_values(flat_components["yawpos_term"], quantiles),
        "z": summarize_values(flat_components["z"], quantiles),
        "sample_level_risk_ratio": {
            "risky_count": int(labels.sum()),
            "total_count": int(len(labels)),
            "ratio": float(np.mean(labels)) if len(labels) > 0 else 0.0,
        },
        "trajectory_level_risk_ratio": summarize_values(
            flat_components["trajectory_risk_ratio"], quantiles
        ),
        "z_risk": summarize_values(z_risk, quantiles),
        "z_safe": summarize_values(z_safe, quantiles),
        "z_to_R_classification": {
            "auroc": compute_auroc(labels, scores),
            "auprc": compute_auprc(labels, scores),
            "positive_rate": float(np.mean(labels)) if len(labels) > 0 else 0.0,
        },
    }
    print_summary(summary)

    summary_path = args.summary_path or report_cfg["summary_path"]
    save_summary(summary, summary_path)
    print(f"saved summary to {summary_path}")

    plot_path = args.plot_path or report_cfg["plot_path"]
    saved_plot = maybe_save_plot(flat_components, plot_path)
    if saved_plot is not None:
        print(f"saved plot to {saved_plot}")
    else:
        print("matplotlib not available, skipped plot generation")


if __name__ == "__main__":
    default_config = os.path.join(
        os.path.dirname(__file__),
        "config",
        "recon_action_z.yaml",
    )

    parser = argparse.ArgumentParser(
        description="Build RECON training samples of ((o_t, a_{t:t+K-1}), Z_t)."
    )
    parser.add_argument(
        "--config",
        "-c",
        default=default_config,
        type=str,
        help="path to YAML config (default: train/config/recon_action_z.yaml)",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=None,
        help="override dataset.input_dir from config",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="override dataset.output_dir from config",
    )
    parser.add_argument(
        "--num-trajs",
        "-n",
        type=int,
        default=None,
        help="override dataset.num_trajs from config",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="compute and save summary stats without exporting images",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="override report.summary_path from config",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=None,
        help="override report.plot_path from config",
    )

    args = parser.parse_args()
    print("STARTING BUILDING RECON ACTION-Z DATASET")
    main(args)
    print("FINISHED BUILDING RECON ACTION-Z DATASET")
