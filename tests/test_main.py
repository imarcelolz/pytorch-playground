import torch

def test_tensor_shape():
    x = torch.rand(5, 3)
    assert x.shape == (5, 3)
