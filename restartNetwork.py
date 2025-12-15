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
        # Try to find any posterior_trans_*.pt file
        candidates = list(path.glob("posterior_trans_*.pt"))
        if candidates:
            path = candidates[0]
            if len(candidates) > 1:
                print(f"[LOAD] Multiple checkpoints found, using: {path.name}")
        else:
            # Fallback to old naming convention
            candidate = path / "posterior_trans.pt"
            if candidate.exists():
                path = candidate
            else:
                raise FileNotFoundError(f"Directory provided but no posterior checkpoint inside: {path}")

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
    field_type = checkpoint.get("field_type", "linear")
    print(f"[LOAD] Transformer posterior loaded from {path} (trained on field_type='{field_type}')")
    return posterior


def save_posterior_transformer(
    path: str | Path,
    prior: BoxUniform,
    posterior,
    embedding_module: torch.nn.Module,
    trans_cfg: dict,
    input_length: int | None = None,
    field_type: str = "linear",
) -> Path:
    """Save a transformer-based posterior.

    Persists:
    - flow state_dict (from posterior)
    - embedding/projection module state_dict (for transformer front-end)
    - minimal metadata: prior bounds, input length, transformer config

    If a directory path is given, writes to 'posterior_trans.pt'. Returns final checkpoint path.
    
    Args:
        field_type: Type of field used for training ("linear" or "cancer")
    """
    path = Path(path)
    if path.is_dir() or path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"posterior_trans_{field_type}.pt"
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
        "field_type": field_type,
    }
    torch.save(checkpoint, path)
    meta_path = path.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump(
            {k: v for k, v in checkpoint.items() if k.endswith("_state_dict") is False}, f, indent=2
        )
    print(f"\n[SAVE] Transformer posterior saved to {path} (metadata: {meta_path})")
    return path
