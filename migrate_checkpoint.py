"""Migra checkpoints PPO renomeando keys no state_dict e na obs_space.

Renomeia ``global_map`` -> ``visited_pooled`` na observação serializada e
``global_cnn`` / ``global_proj`` -> ``visited_cnn`` / ``visited_proj`` nos
módulos do policy. Permite carregar com a versão atual do código sem precisar
retreinar (~3h).

Uso:
    python migrate_checkpoint.py data/in.zip data/out.zip
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import zipfile

import torch as th


KEY_MAP = {
    "global_cnn": "visited_cnn",
    "global_proj": "visited_proj",
}
OBS_KEY_MAP = {"global_map": "visited_pooled"}


def rename_state_dict(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        new_k = k
        for old, new in KEY_MAP.items():
            new_k = new_k.replace(old, new)
        out[new_k] = v
    return out


def rename_obs_in_data(data_bytes: bytes) -> bytes:
    """Replace 'global_map' with 'visited_pooled' in both the JSON spaces
    list and the base64-serialized observation_space."""
    obj = json.loads(data_bytes.decode("utf-8"))
    obs = obj.get("observation_space", {})
    # The spaces summary string has 'global_map' as a key — replace.
    if isinstance(obs.get("spaces"), str):
        obs["spaces"] = obs["spaces"].replace(
            "'global_map'", "'visited_pooled'")
    # The serialized blob is pickled bytes; decoding+modifying is risky.
    # Instead, after load, call PPO.load(custom_objects={'observation_space':
    # env.observation_space}) to override. The :serialized: blob is left as-is.
    return json.dumps(obj).encode("utf-8")


def migrate(in_path: str, out_path: str) -> None:
    print(f"Migrating {in_path} -> {out_path}")
    if in_path == out_path:
        raise SystemExit("in_path and out_path must differ")

    with zipfile.ZipFile(in_path, "r") as zin:
        names = zin.namelist()
        files = {n: zin.read(n) for n in names}

    # Patch policy.pth and policy.optimizer.pth state_dicts.
    for fname in ["policy.pth", "policy.optimizer.pth"]:
        if fname not in files:
            continue
        buf = io.BytesIO(files[fname])
        try:
            state = th.load(buf, weights_only=True)
        except Exception:
            buf.seek(0)
            state = th.load(buf, weights_only=False)

        if isinstance(state, dict):
            renamed = _maybe_rename(state)
            out_buf = io.BytesIO()
            th.save(renamed, out_buf)
            files[fname] = out_buf.getvalue()
            print(f"  patched {fname}: {len(state)} keys")

    # Patch JSON data (obs space spaces summary).
    if "data" in files:
        files["data"] = rename_obs_in_data(files["data"])
        print("  patched data: obs space summary")

    # Re-write zip.
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, data in files.items():
            zout.writestr(n, data)
    print(f"Wrote {out_path}")


def _maybe_rename(state):
    """Apply KEY_MAP to a (possibly nested) state_dict.

    SB3 saves policy.pth as a dict with substructures. For PPO the top-level
    is the policy state_dict. For optimizer it's a dict with 'state' and
    'param_groups'. We try to rename keys in any nested string-keyed dict.
    """
    if isinstance(state, dict):
        # Check if it's a flat state_dict (string keys only).
        if all(isinstance(k, str) for k in state.keys()):
            renamed = rename_state_dict(state)
            # Recurse into nested dicts.
            for k, v in list(renamed.items()):
                if isinstance(v, dict):
                    renamed[k] = _maybe_rename(v)
            return renamed
    return state


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("in_path")
    p.add_argument("out_path")
    args = p.parse_args()
    migrate(args.in_path, args.out_path)
