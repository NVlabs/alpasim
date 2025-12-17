import numpy as np
import pytest

from ..frame_cache import FrameCache


def _make_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(2, 2, 3), dtype=np.uint8)


def test_add_image_orders_and_prunes() -> None:
    cache = FrameCache(context_length=3)
    cache.add_image(30, _make_image(30))
    cache.add_image(10, _make_image(10))
    cache.add_image(20, _make_image(20))

    assert [entry.timestamp_us for entry in cache.entries] == [10, 20, 30]
    assert cache.frame_count() == 3

    # Buffer size = context_length * subsample_factor
    # With defaults (context_length=3, subsample_factor=1): 3 * 1 = 3
    # Adding a 4th frame should trigger pruning immediately
    cache.add_image(25, _make_image(25))
    assert [entry.timestamp_us for entry in cache.entries] == [20, 25, 30]
    assert cache.frame_count() == 3

    # Adding another frame continues to maintain max of 3
    cache.add_image(35, _make_image(35))
    assert [entry.timestamp_us for entry in cache.entries] == [25, 30, 35]
    assert cache.frame_count() == 3


def test_add_image_rejects_duplicate_timestamp() -> None:
    cache = FrameCache(context_length=2)
    cache.add_image(5, _make_image(5))

    with pytest.raises(ValueError):
        cache.add_image(5, _make_image(5))


def test_latest_frame_entries() -> None:
    """Test retrieving the latest N frame entries."""
    cache = FrameCache(context_length=3)
    cache.add_image(10, _make_image(10))
    cache.add_image(20, _make_image(20))
    cache.add_image(30, _make_image(30))

    # Get last 2 entries
    entries = cache.latest_frame_entries(2)
    assert len(entries) == 2
    assert entries[0].timestamp_us == 20  # oldest of the 2
    assert entries[1].timestamp_us == 30  # newest

    # Get all 3 entries
    entries = cache.latest_frame_entries(3)
    assert len(entries) == 3
    assert [e.timestamp_us for e in entries] == [10, 20, 30]


def test_latest_frame_entries_insufficient_frames() -> None:
    """Test that latest_frame_entries raises when not enough frames."""
    cache = FrameCache(context_length=3)
    cache.add_image(10, _make_image(10))

    with pytest.raises(ValueError, match="Insufficient frames"):
        cache.latest_frame_entries(2)


def test_latest_frame_entries_with_subsampling() -> None:
    """Test that latest_frame_entries respects subsample_factor."""
    # context_length=3, subsample_factor=2 means we select every other frame
    cache = FrameCache(context_length=3, subsample_factor=2)

    # Add 6 frames (timestamps 100-600)
    for i in range(6):
        cache.add_image((i + 1) * 100, _make_image(i))

    # Request 3 frames with subsample_factor=2
    # Should select: newest (600), newest-2 (400), newest-4 (200)
    # Returned oldest-first: [200, 400, 600]
    entries = cache.latest_frame_entries(3)
    assert len(entries) == 3
    assert [e.timestamp_us for e in entries] == [200, 400, 600]


def test_latest_frame_entries_subsampling_insufficient() -> None:
    """Test that subsampled selection fails with insufficient frames."""
    cache = FrameCache(context_length=3, subsample_factor=2)
    # Need at least (3-1)*2 + 1 = 5 frames for subsampled selection of 3

    for i in range(4):
        cache.add_image((i + 1) * 100, _make_image(i))

    with pytest.raises(ValueError, match="Insufficient frames.*need at least 5"):
        cache.latest_frame_entries(3)


def test_subsample_factor_min_frames_required() -> None:
    """Test min_frames_required calculation with different subsample factors."""
    # context_length=3, subsample_factor=1: min = (3-1)*1 + 1 = 3
    cache1 = FrameCache(context_length=3, subsample_factor=1)
    assert cache1.min_frames_required() == 3

    # context_length=3, subsample_factor=2: min = (3-1)*2 + 1 = 5
    cache2 = FrameCache(context_length=3, subsample_factor=2)
    assert cache2.min_frames_required() == 5

    # context_length=4, subsample_factor=3: min = (4-1)*3 + 1 = 10
    cache3 = FrameCache(context_length=4, subsample_factor=3)
    assert cache3.min_frames_required() == 10


def test_subsample_factor_buffer_size() -> None:
    """Test that buffer size accommodates subsampling."""
    cache = FrameCache(context_length=3, subsample_factor=2)
    # max_entries = context_length * subsample_factor = 3 * 2 = 6

    for i in range(10):
        cache.add_image(i * 100, _make_image(i))

    # Should keep max_entries = 6 frames
    assert cache.frame_count() == 6
    # Should keep the newest 6: [400, 500, 600, 700, 800, 900]
    assert [e.timestamp_us for e in cache.entries] == [
        400,
        500,
        600,
        700,
        800,
        900,
    ]


def test_has_enough_frames() -> None:
    """Test has_enough_frames helper method."""
    cache = FrameCache(context_length=3, subsample_factor=2)
    # Min required = 5

    for i in range(4):
        cache.add_image(i * 100, _make_image(i))
        assert not cache.has_enough_frames()

    cache.add_image(400, _make_image(4))
    assert cache.has_enough_frames()


def test_frame_entries_contain_images() -> None:
    """Test that frame entries properly store images."""
    cache = FrameCache(context_length=2)
    img1 = _make_image(1)
    img2 = _make_image(2)

    cache.add_image(100, img1)
    cache.add_image(200, img2)

    entries = cache.latest_frame_entries(2)
    assert np.array_equal(entries[0].image, img1)
    assert np.array_equal(entries[1].image, img2)
