import pytest

from dopest_model.fastserve import BUCKETS, bucket_for, padded_segments


def test_buckets_leave_room_for_the_padding_segment():
    assert bucket_for(255) == 256 and bucket_for(256) == 384 and bucket_for(700) == 768
    assert bucket_for(40959) == 40960 and bucket_for(40960) == 40961      # beyond the table: exact size plus one
    assert list(BUCKETS) == sorted(BUCKETS) and len(set(BUCKETS)) == len(BUCKETS)


def test_padding_fills_the_bucket_exactly_as_a_second_segment():
    segs = padded_segments([5, 6, 7], pad_id=0, bucket=8)
    assert segs == [[5, 6, 7], [0, 0, 0, 0, 0]] and sum(map(len, segs)) == 8
    with pytest.raises(ValueError):
        padded_segments([1, 2, 3], pad_id=0, bucket=3)


def test_lora_scale_is_relative_to_the_trained_strength():
    from types import SimpleNamespace
    from dopest_model.fastserve import set_lora_scale
    layer = SimpleNamespace(scaling={"default": 2.0}, lora_A={"default": object()}, modules=lambda: [layer])
    plain = SimpleNamespace(scaling="not a dict", modules=lambda: [plain])
    model = SimpleNamespace(modules=lambda: [layer, plain])
    assert set_lora_scale(model, 0.5) == 1 and layer.scaling["default"] == 1.0
    assert set_lora_scale(model, 0.25) == 1 and layer.scaling["default"] == 0.5     # relative to 2.0, not to 1.0
    assert set_lora_scale(model, 1.0) == 1 and layer.scaling["default"] == 2.0
