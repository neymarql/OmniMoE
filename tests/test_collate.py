from OmniMoE.data.collate import multimodal_collate_fn
from OmniMoE.data.dataset import ConversationSample
import torch


def test_multimodal_collate_shapes():
    pad = 0
    s1 = ConversationSample(
        input_ids=torch.tensor([1, 2, 3]),
        attention_mask=torch.tensor([1, 1, 1]),
        labels=torch.tensor([-100, -100, 3]),
        pixel_values=torch.zeros(3, 224, 224),
        image_mask=torch.tensor(1),
        pad_token_id=pad,
    )
    s2 = ConversationSample(
        input_ids=torch.tensor([4, 5]),
        attention_mask=torch.tensor([1, 1]),
        labels=torch.tensor([-100, 5]),
        pixel_values=torch.zeros(3, 224, 224),
        image_mask=torch.tensor(0),
        pad_token_id=pad,
    )
    batch = multimodal_collate_fn([s1, s2])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert batch["pixel_values"].shape == (2, 3, 224, 224)
    assert batch["image_mask"].shape == (2,)
