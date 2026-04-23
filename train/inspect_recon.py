import argparse
import os

import h5py


REQUIRED_PATHS = (
    "jackal/position",
    "jackal/yaw",
    "images/rgb_left",
)


def print_node(name: str, obj) -> None:
    kind = "Group" if isinstance(obj, h5py.Group) else "Dataset"
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    print(f"{name} [{kind}] shape={shape} dtype={dtype}")


def path_exists(h5_file: h5py.File, path: str) -> bool:
    try:
        h5_file[path]
        return True
    except KeyError:
        return False


def inspect_file(file_path: str, show_tree: bool) -> None:
    print(f"\n=== {file_path} ===")
    try:
        with h5py.File(file_path, "r") as h5_f:
            top_keys = list(h5_f.keys())
            print(f"top-level keys: {top_keys}")

            for key in top_keys:
                obj = h5_f[key]
                if isinstance(obj, h5py.Group):
                    print(f"{key}: Group, children={list(obj.keys())}")
                else:
                    print(
                        f"{key}: Dataset, shape={obj.shape}, dtype={obj.dtype}"
                    )

            print("required fields:")
            for path in REQUIRED_PATHS:
                if path_exists(h5_f, path):
                    obj = h5_f[path]
                    print(f"  OK   {path} shape={obj.shape} dtype={obj.dtype}")
                else:
                    print(f"  MISS {path}")

            if show_tree:
                print("full tree:")
                h5_f.visititems(print_node)
    except OSError as exc:
        print(f"failed to open: {exc}")


def resolve_targets(input_path: str):
    if os.path.isfile(input_path):
        return [input_path]

    recon_dir = os.path.join(input_path, "recon_release")
    target_dir = recon_dir if os.path.isdir(recon_dir) else input_path

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(
            f"input path is neither a file nor a directory: {input_path}"
        )

    files = [
        os.path.join(target_dir, name)
        for name in sorted(os.listdir(target_dir))
        if name.endswith((".h5", ".hdf5"))
    ]
    return files


def main(args: argparse.Namespace) -> None:
    files = resolve_targets(args.input_path)
    if not files:
        print("no .h5 or .hdf5 files found")
        return

    if args.num_files >= 0:
        files = files[: args.num_files]

    print(f"inspecting {len(files)} file(s)")
    for file_path in files:
        inspect_file(file_path, show_tree=args.show_tree)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect raw RECON HDF5 files and summarize usable fields."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help=(
            "path to a RECON dataset directory, its recon_release subdirectory, "
            "or a single .h5 file"
        ),
    )
    parser.add_argument(
        "--num-files",
        "-n",
        type=int,
        default=3,
        help="number of files to inspect when input_path is a directory (default: 3)",
    )
    parser.add_argument(
        "--show-tree",
        action="store_true",
        help="print the full HDF5 tree for each inspected file",
    )

    main(parser.parse_args())
