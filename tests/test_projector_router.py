import torch
from OmniMoE.models.projector_bridge import ProjectorBridge


def test_projector_soft_routing_forward():
    B, V, H = 2, 16, 64
    T = 8
    proj = ProjectorBridge(hidden_size=H, num_queries=T, num_layers=2, num_heads=4, num_experts=4, ec_routing=False)
    img = torch.randn(B, V, H)
    txt = torch.randn(B, T, H)
    out = proj(image_embeddings=img, text_embeddings=txt, image_mask=torch.ones(B, 1))
    assert out.shape == (B, T, H)
    aux = proj.aux_loss()
    assert aux.ndim == 0


def test_projector_ec_routing_forward():
    B, V, H = 2, 16, 64
    T = 8
    proj = ProjectorBridge(hidden_size=H, num_queries=T, num_layers=2, num_heads=4, num_experts=4, ec_routing=True)
    img = torch.randn(B, V, H)
    txt = torch.randn(B, T, H)
    out = proj(image_embeddings=img, text_embeddings=txt, image_mask=torch.ones(B, 1))
    assert out.shape == (B, T, H)
    aux = proj.aux_loss()
    assert aux.ndim == 0
