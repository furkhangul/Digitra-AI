from PIL import Image
import torch
from torch import nn

from src.cnn import Letterbox, tta_logits
from src.hand_crop import padded_square_box


class MeanModel(nn.Module):
    def forward(self, value):
        means = value.mean(dim=(2, 3))
        return torch.stack((means[:, 0], means[:, 1]), dim=1)


def test_letterbox_keeps_square_size():
    image = Image.new("RGB", (320, 120), "white")
    assert Letterbox(224)(image).size == (224, 224)


def test_mirror_tta_is_exactly_mirror_invariant():
    generator = torch.Generator().manual_seed(4)
    value = torch.rand((2, 3, 32, 32), generator=generator)
    model = MeanModel()
    original = tta_logits(model, value, scales=(1.0,), mirror=True)
    mirrored = tta_logits(model, torch.flip(value, dims=(-1,)), scales=(1.0,), mirror=True)
    torch.testing.assert_close(original, mirrored)


def test_hand_crop_box_is_square_and_clamped():
    points = torch.tensor([[2.0, 3.0], [18.0, 20.0]]).numpy()
    x1, y1, x2, y2 = padded_square_box(points, width=24, height=24, padding=0.4)
    assert 0 <= x1 < x2 <= 24
    assert 0 <= y1 < y2 <= 24
    assert abs((x2 - x1) - (y2 - y1)) <= 1


def test_hand_crop_can_keep_full_context_for_a_missed_second_hand():
    points = torch.tensor([[14.0, 10.0], [18.0, 15.0]]).numpy()
    x1, y1, x2, y2 = padded_square_box(
        points,
        width=32,
        height=24,
        padding=0.4,
        minimum_side_fraction=1.0,
    )
    assert x2 - x1 == 24
    assert y2 - y1 == 24
