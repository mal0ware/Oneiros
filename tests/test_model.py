"""Model forward-shape tests and EMA / regularizer sanity checks."""

import torch

from oneiros.model import (
    JEPA,
    ModelConfig,
    covariance_loss,
    variance_loss,
)


def test_vector_encoder_forward_shape():
    cfg = ModelConfig(obs_mode="vector", latent_dim=16)
    model = JEPA(cfg)
    obs = torch.randn(8, cfg.vector_dim)
    z = model.encode(obs)
    assert z.shape == (8, 16)


def test_image_encoder_forward_shape():
    cfg = ModelConfig(obs_mode="image", latent_dim=16)
    model = JEPA(cfg)
    obs = torch.randn(4, cfg.img_channels, cfg.img_size, cfg.img_size)
    z = model.encode(obs)
    assert z.shape == (4, 16)


def test_predictor_forward_shape():
    cfg = ModelConfig(latent_dim=16, action_dim=2)
    model = JEPA(cfg)
    z = torch.randn(5, 16)
    a = torch.randn(5, 2)
    z_hat = model.predict(z, a)
    assert z_hat.shape == (5, 16)


def test_ema_target_moves_toward_online():
    cfg = ModelConfig(obs_mode="vector", ema_decay=0.5)
    model = JEPA(cfg)
    # perturb the online encoder, then EMA-update the target toward it.
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    before = next(model.target_encoder.parameters()).clone()
    model.update_target()
    after = next(model.target_encoder.parameters())
    assert not torch.allclose(before, after)


def test_variance_loss_zero_when_spread():
    z = torch.randn(512, 16) * 2.0  # std ~2 > target 1 -> hinge is zero
    assert float(variance_loss(z)) == 0.0


def test_variance_loss_positive_when_collapsed():
    z = torch.zeros(512, 16)  # fully collapsed -> std 0 -> penalty ~target
    assert float(variance_loss(z)) > 0.5


def test_covariance_loss_nonnegative():
    z = torch.randn(256, 16)
    assert float(covariance_loss(z)) >= 0.0
