#!/usr/bin/env python3
"""
Utility helpers for saving and loading sbi posteriors.
"""

import json
from pathlib import Path

import torch
from sbi.inference import NPE
from sbi.neural_nets import embedding_nets, posterior_nn
from sbi.utils import BoxUniform


def load_posterior(path: str | Path, prior: BoxUniform):
    """Load posterior from a saved checkpoint."""
    path = Path(path)
    if path.is_dir():
        candidate = path / "posterior_cnn.pt"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"Directory provided but no posterior_cnn.pt inside: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    # Build actual neural net module from builder by providing shapes
    builder = posterior_nn(
        model="maf",
        z_score_x="none",
        z_score_y="none",
    )
    input_len = checkpoint.get("input_length", None)
    if input_len is None:
        raise KeyError("Checkpoint missing 'input_length' needed to rebuild network.")
    # y_shape is parameter dimensionality (theta), inferred from prior
    y_dim = prior.sample((1,)).shape[-1]
    # Build by providing example batches on the correct device as expected by sbi
    batch_x = torch.zeros(2, input_len)
    batch_theta = torch.zeros(2, y_dim)
    # sbi's builder may internally map (batch_x=batch_theta, batch_y=batch_x).
    # To match expected order, pass (batch_theta, batch_x) positionally.
    density_estimator = builder(batch_theta, batch_x)
    # Some sbi wrappers expect parameters under the 'net.' prefix.
    state_dict = checkpoint["state_dict"]
    needs_prefix = any(not k.startswith("net.") for k in state_dict.keys())
    if needs_prefix:
        # If none of the keys start with 'net.', prepend it to align with NFlowsFlow expectations.
        state_dict = {("net." + k): v for k, v in state_dict.items()}
    density_estimator.load_state_dict(state_dict, strict=False)
    # NPE expects a builder function for density_estimator in __init__.
    # Provide the builder separately and pass the loaded module to build_posterior.
    builder_for_npe = posterior_nn(
        model="maf",
        z_score_x="none",
        z_score_y="none",
    )
    inference = NPE(prior=prior, density_estimator=builder_for_npe)
    posterior = inference.build_posterior(density_estimator)
    print(f"[LOAD] Posterior loaded from {path}")
    return posterior


def save_posterior(
    path: str | Path, prior: BoxUniform, posterior, input_length: int | None = None
) -> Path:
    """Save posterior flow state_dict plus minimal metadata.

    If a directory path is provided, a default filename 'posterior_cnn.pt' is used.
    Returns final checkpoint path.
    """
    path = Path(path)
    if path.is_dir() or path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "posterior_cnn.pt"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Locate underlying network (nflows flow or wrapped module)
    net = getattr(posterior, "net", None)
    if net is None:
        net = getattr(posterior, "_flow", None)
    if net is None:
        net = getattr(posterior, "flow", None)
    if net is None:
        raise TypeError("Could not find underlying network on posterior (net/_flow/flow).")

    if input_length is None:
        input_length = getattr(posterior, "x_shape", None)
        if isinstance(input_length, (tuple, list)) and len(input_length) > 0:
            input_length = input_length[0]

    # Extract bounds safely (BoxUniform stores low/high tensors on prior.low/high)
    theta_min = getattr(prior, "low", None)
    theta_max = getattr(prior, "high", None)
    if theta_min is not None:
        theta_min = theta_min.detach().cpu().tolist()
    if theta_max is not None:
        theta_max = theta_max.detach().cpu().tolist()

    checkpoint = {
        "state_dict": net.state_dict(),
        "theta_min": theta_min,
        "theta_max": theta_max,
        "input_length": input_length,
        "model": "maf",
    }
    torch.save(checkpoint, path)
    meta_path = path.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump({k: v for k, v in checkpoint.items() if k != "state_dict"}, f, indent=2)
    print(f"\n[SAVE] Posterior saved to {path} (metadata: {meta_path})")
    return path


def load_posterior_transformer(path: str | Path, prior: BoxUniform):
    """Load transformer-based posterior and embedding from checkpoint.

    Expects a checkpoint saved by save_posterior_transformer containing:
    - flow_state_dict
    - embed_state_dict
    - trans_cfg
    - input_length (flattened xy length = 2*T)
    """
    path = Path(path)
    if path.is_dir():
        candidate = path / "posterior_trans.pt"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"Directory provided but no posterior_trans.pt inside: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    trans_cfg = checkpoint.get("trans_cfg")
    if trans_cfg is None:
        raise KeyError("Transformer checkpoint missing 'trans_cfg'.")

    # Rebuild transformer embedding
    base_trans = embedding_nets.TransformerEmbedding(trans_cfg)

    d_model = int(trans_cfg.get("d_model", trans_cfg.get("feature_space_dim", 192)))
    seq_len = int(trans_cfg.get("sequence_length", 100))
    # Note: input_length in checkpoint is flattened (seq*features), not used for seq_len

    class ProjectedTransformer(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.proj, self.transformer = torch.nn.Linear(3, d_model), transformer

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 2:
                x = x.unsqueeze(0)  # add batch dimension: [seq, feat] -> [1, seq, feat]
            x = self.proj(x)
            return self.transformer(x)

    embedding_trans = ProjectedTransformer(base_trans)
    # Load embedding weights
    embed_state = checkpoint.get("embed_state_dict")
    if embed_state is None:
        raise KeyError("Transformer checkpoint missing 'embed_state_dict'.")
    embedding_trans.load_state_dict(embed_state, strict=False)

    # Build flow with embedding
    builder = posterior_nn(
        model="maf",
        embedding_net=embedding_trans,
        z_score_x="independent",
        z_score_y="independent",
    )

    # Prepare example batches to build network (match sbi signature discovered earlier)
    y_dim = prior.sample((1,)).shape[-1]
    batch_x = torch.zeros(2, seq_len, 3)  # Must match projection input features
    batch_theta = torch.zeros(2, y_dim)
    density_estimator = builder(batch_theta, batch_x)

    # Load flow state dict (adjust prefix if needed)
    state_dict = checkpoint.get("flow_state_dict")
    if state_dict is None:
        raise KeyError("Transformer checkpoint missing 'flow_state_dict'.")
    needs_prefix = any(not k.startswith("net.") for k in state_dict.keys())
    if needs_prefix:
        state_dict = {("net." + k): v for k, v in state_dict.items()}
    density_estimator.load_state_dict(state_dict, strict=False)

    # Build posterior via NPE using a builder function
    builder_for_npe = posterior_nn(
        model="maf",
        embedding_net=embedding_trans,
        z_score_x="independent",
        z_score_y="independent",
    )
    inference = NPE(prior=prior, density_estimator=builder_for_npe)
    posterior = inference.build_posterior(density_estimator)
    print(f"[LOAD] Transformer posterior loaded from {path}")
    return posterior


def save_posterior_transformer(
    path: str | Path,
    prior: BoxUniform,
    posterior,
    embedding_module: torch.nn.Module,
    trans_cfg: dict,
    input_length: int | None = None,
) -> Path:
    """Save a transformer-based posterior.

    Persists:
    - flow state_dict (from posterior)
    - embedding/projection module state_dict (for transformer front-end)
    - minimal metadata: prior bounds, input length, transformer config

    If a directory path is given, writes to 'posterior_trans.pt'. Returns final checkpoint path.
    """
    path = Path(path)
    if path.is_dir() or path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "posterior_trans.pt"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Find posterior underlying flow/net
    net = getattr(posterior, "net", None)
    if net is None:
        net = getattr(posterior, "_flow", None)
    if net is None:
        net = getattr(posterior, "flow", None)
    if net is None:
        raise TypeError(
            "Could not find underlying network on transformer posterior (net/_flow/flow)."
        )

    if input_length is None:
        input_length = getattr(posterior, "x_shape", None)
        if isinstance(input_length, (tuple, list)) and len(input_length) > 0:
            input_length = input_length[0]

    theta_min = getattr(prior, "low", None)
    theta_max = getattr(prior, "high", None)
    if theta_min is not None:
        theta_min = theta_min.detach().cpu().tolist()
    if theta_max is not None:
        theta_max = theta_max.detach().cpu().tolist()

    checkpoint = {
        "flow_state_dict": net.state_dict(),
        "embed_state_dict": embedding_module.state_dict(),
        "theta_min": theta_min,
        "theta_max": theta_max,
        "input_length": input_length,
        "model": "maf+transformer",
        "trans_cfg": trans_cfg,
    }
    torch.save(checkpoint, path)
    meta_path = path.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump(
            {k: v for k, v in checkpoint.items() if k.endswith("_state_dict") is False}, f, indent=2
        )
    print(f"\n[SAVE] Transformer posterior saved to {path} (metadata: {meta_path})")
    return path
