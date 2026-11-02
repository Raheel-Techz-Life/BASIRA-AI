import torch

from src.models.cnn import SimpleCNN


def test_forward_pass_shape():
    model = SimpleCNN(num_classes=4)
    model.eval()
    batch = torch.rand(2, 1, 128, 128)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2, 4)


def test_param_count_is_small():
    model = SimpleCNN(num_classes=4)
    n_params = sum(p.numel() for p in model.parameters())
    # Sanity check: this is a deliberately small CNN, not a large backbone.
    assert n_params < 1_000_000
