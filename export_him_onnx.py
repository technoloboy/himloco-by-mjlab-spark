"""Export a HIM velocity checkpoint to ONNX.

Usage:
    python export_him_onnx.py --checkpoint <path/to/model_N.pt> [--output <out.onnx>]

The script reconstructs the HIMActorModel from the checkpoint's weight shapes
(no environment required), wraps it with _HIMExportModel, and exports via
torch.onnx.  The resulting ONNX input is:

    obs: float32[1, S + H*S]  = concat([current_frame(47), history(282)])

Output:
    actions: float32[1, 12]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

# Make sure the repo root is on the path.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Export HIM checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        default="logs/rsl_rl/boying_him_velocity/2026-06-21_16-41-29/model_1180.pt",
        help="Path to the .pt checkpoint file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .onnx path (default: same dir as checkpoint, policy_1180.onnx)",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def infer_dims(actor_sd: dict) -> tuple[int, int, int, int]:
    """Infer (num_one_step_obs, history_length, latent_dim, action_dim) from weights."""
    # encoder input = history_dim = S * H
    history_dim: int = actor_sd["estimator.encoder.0.weight"].shape[1]
    # encoder output = vel(3) + latent_dim
    enc_out_dim: int = actor_sd[
        [k for k in actor_sd if "estimator.encoder" in k and "weight" in k][-1]
    ].shape[0]
    latent_dim = enc_out_dim - 3
    # mlp input = current(S) + vel(3) + latent(L)
    mlp_in: int = actor_sd["mlp.0.weight"].shape[1]
    num_one_step_obs = mlp_in - 3 - latent_dim
    history_length = history_dim // num_one_step_obs
    # mlp output = action_dim
    action_dim: int = actor_sd[
        [k for k in actor_sd if k.startswith("mlp.") and "weight" in k][-1]
    ].shape[0]
    return num_one_step_obs, history_length, latent_dim, action_dim


def build_him_export_model(actor_sd: dict) -> nn.Module:
    """Reconstruct _HIMExportModel from a checkpoint state-dict."""
    from src.algorithms.him.him_actor import HIMActorModelCfg
    from tensordict import TensorDict

    num_one_step_obs, history_length, latent_dim, action_dim = infer_dims(actor_sd)
    history_dim = num_one_step_obs * history_length

    print(f"  num_one_step_obs : {num_one_step_obs}")
    print(f"  history_length   : {history_length}")
    print(f"  latent_dim       : {latent_dim}")
    print(f"  action_dim       : {action_dim}")
    print(f"  model input size : {num_one_step_obs} + {history_dim} = {num_one_step_obs + history_dim}")

    # Build a dummy obs TensorDict so HIMActorModel can infer sizes.
    dummy_obs = TensorDict(
        {
            "proprio_history": torch.zeros(1, history_dim),
            "proprio_current": torch.zeros(1, num_one_step_obs),
        },
        batch_size=[1],
    )

    from src.algorithms.him.him_actor import HIMActorModel

    # Infer hidden_dims from mlp weight shapes.
    # mlp: Linear(latent,h1), act, ..., Linear(hN, action_dim)
    mlp_weights = [
        (k, actor_sd[k].shape)
        for k in sorted(actor_sd)
        if k.startswith("mlp.") and "weight" in k
    ]
    hidden_dims = tuple(s[0] for _, s in mlp_weights[:-1])

    # Infer estimator enc_hidden_dims = (h0, h1, ..., num_latent).
    # The encoder's final Linear outputs (num_latent + 3); the layers before it
    # are hidden.  So enc_hidden_dims[-1] = last_layer_out - 3.
    enc_weights = [
        (k, actor_sd[k].shape)
        for k in sorted(actor_sd)
        if "estimator.encoder" in k and "weight" in k
    ]
    # Hidden layers: all but the last; last layer outputs latent+3.
    enc_hidden = tuple(s[0] for _, s in enc_weights[:-1])
    enc_last_out = enc_weights[-1][1][0]   # e.g. 19 = 16 + 3
    enc_hidden_dims = enc_hidden + (enc_last_out - 3,)  # e.g. (128, 64, 16)

    # Infer estimator tar_hidden_dims.
    # Target network: for i in range(len(tar_hidden_dims)): Linear+act; then
    # final Linear(tar_hidden_dims[-1], enc_hidden_dims[-1]).
    # So tar_hidden_dims covers all layers except the final output layer.
    tar_weights = [
        (k, actor_sd[k].shape)
        for k in sorted(actor_sd)
        if "estimator.target" in k and "weight" in k
    ]
    # All hidden outputs except the last (output) layer.
    tar_hidden_dims = tuple(s[0] for _, s in tar_weights[:-1])

    # num_prototype from proto weight.
    num_prototype: int = actor_sd["estimator.proto.weight"].shape[0]

    actor = HIMActorModel(
        obs=dummy_obs,
        obs_groups={"actor": ["proprio_history"]},
        obs_set="actor",
        output_dim=action_dim,
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        history_length=history_length,
        history_group="proprio_history",
        current_group="proprio_current",
        estimator={
            "enc_hidden_dims": enc_hidden_dims,
            "tar_hidden_dims": tar_hidden_dims,
            "num_prototype": num_prototype,
        },
    )
    actor.load_state_dict(actor_sd)
    actor.eval()

    export_model = actor.as_onnx(verbose=False)
    export_model.to("cpu")
    export_model.eval()
    return export_model


def main():
    args = parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    # Default output: same directory, policy_<iter>.onnx
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        ckpt_dir = os.path.dirname(checkpoint_path)
        stem = os.path.splitext(os.path.basename(checkpoint_path))[0]  # e.g. model_1180
        output_path = os.path.join(ckpt_dir, f"policy_{stem.split('_', 1)[-1]}.onnx")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_sd = ckpt["actor_state_dict"]

    print("Building export model ...")
    export_model = build_him_export_model(actor_sd)

    print(f"Exporting to ONNX (opset {args.opset}) ...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        export_model,
        export_model.get_dummy_inputs(),
        output_path,
        export_params=True,
        opset_version=args.opset,
        verbose=args.verbose,
        input_names=export_model.input_names,
        output_names=export_model.output_names,
        dynamic_axes={},
        dynamo=False,
    )
    print(f"Saved: {output_path}")

    # Sanity check: verify the ONNX graph is well-formed.
    import onnx
    model_proto = onnx.load(output_path)
    onnx.checker.check_model(model_proto)
    print(f"ONNX graph check OK — {len(model_proto.graph.node)} nodes")

    # Optional runtime check if onnxruntime is available.
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, export_model.input_size), dtype=np.float32)
        actions = sess.run(None, {"obs": dummy_np})[0]
        print(f"ONNXRuntime inference check OK — actions shape: {actions.shape}")
    except ImportError:
        print("(onnxruntime not installed — skipping runtime inference check)")


if __name__ == "__main__":
    main()
