"""Logit distillation (kd): loss values, config checks, short real runs (finetune, and prune with a
separate output teacher)."""

import json

import pytest
import torch

from karna_train.model import ArchConfig, PrunableCTC
from karna_train.train import Trainer, kd_loss, load_config
from test_resume import pytestmark as needs_cuda_and_data
from test_resume import tiny_ckpt, write_cfg


def test_kd_loss_zero_for_identical_and_masks_padding():
    torch.manual_seed(0)
    s, t = torch.randn(2, 5, 7), torch.randn(2, 5, 7)
    valid = torch.ones(2, 5, dtype=torch.bool)
    assert kd_loss(t, t, valid, 1.0).abs() < 1e-6
    assert kd_loss(s, t, valid, 1.0) > 0
    # padded frames do not count
    s2 = s.clone()
    s2[:, 3:] = 100.0
    valid[:, 3:] = False
    assert torch.allclose(kd_loss(s2, t, valid, 1.0), kd_loss(s[:, :3], t[:, :3], valid[:, :3], 1.0))


def test_kd_needs_a_teacher(tmp_path):
    with pytest.raises(ValueError, match="kd.weight"):
        load_config(write_cfg(tmp_path, stage="finetune", teacher=None, **{"kd": {"weight": 1.0}}), [])
    # an output teacher alone is enough for finetune, and prune/distill may add logit KD
    cfg = load_config(write_cfg(tmp_path, stage="finetune", teacher=None,
                                **{"kd": {"weight": 1.0, "teacher": str(tmp_path / "big.pt")}}), [])
    assert cfg["kd"]["teacher"] == str(tmp_path / "big.pt") and cfg["kd"]["ctc_weight"] == 1.0
    assert load_config(write_cfg(tmp_path, **{"kd": {"weight": 1.0}}), [])["kd"]["teacher"] is None


@needs_cuda_and_data
def test_finetune_with_kd_runs(tmp_path):
    tiny_ckpt(tmp_path / "tiny.pt")
    cfg = load_config(write_cfg(tmp_path, stage="finetune", **{"kd": {"weight": 1.0, "temperature": 1.0},
                                                               "model.precision": "amp",
                                                               "optim.max_steps": 6}), [])
    tr = Trainer(cfg, "never")
    assert tr.teacher is not None
    assert tr.fit() == 0
    rows = [r for r in map(json.loads, open(tmp_path / "run/metrics.jsonl")) if r.get("event") == "train"]
    assert rows and all("kd" in r and "ctc" in r for r in rows)


def tiny_output_teacher(path):
    """A wider, deeper tiny model with the student's conv stack (so the same frames) and vocabulary."""
    torch.manual_seed(1)
    cfg = ArchConfig(conv_layers=[(32, 10, 5), (32, 3, 2), (32, 3, 2), (32, 3, 2), (32, 3, 2), (32, 2, 2), (32, 2, 2)],
                     num_heads=[3, 3, 3], ffn_dims=[96, 96, 96], model_dim=48, head_dim=16,
                     pos_conv_kernel=8, pos_conv_groups=2, vocab_size=9812)
    torch.save({"arch": cfg.to_dict(), "model": PrunableCTC(cfg).state_dict()}, path)


@needs_cuda_and_data
def test_prune_with_separate_output_teacher_runs(tmp_path):
    tiny_ckpt(tmp_path / "tiny.pt")
    tiny_output_teacher(tmp_path / "big.pt")
    cfg = load_config(write_cfg(tmp_path, **{"kd": {"weight": 1.0, "teacher": str(tmp_path / "big.pt")},
                                             "optim.max_steps": 6}), [])
    tr = Trainer(cfg, "never")
    assert tr.kd_teacher is not None and tr.kd_teacher is not tr.teacher
    alphas = {n: p.detach().clone() for n, p in tr.model.named_parameters() if n.endswith("log_alpha")}
    assert tr.fit() == 0
    rows = [r for r in map(json.loads, open(tmp_path / "run/metrics.jsonl")) if r.get("event") == "train"]
    assert rows and all("kd" in r and "distill" in r and "ctc" not in r for r in rows)
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in rows)
    moved = [n for n, p in tr.model.named_parameters() if n in alphas and not torch.equal(p.detach(), alphas[n])]
    assert moved, "the gates must get gradients"
