from piecewise_dataset_pipeline.build_combined_dataset_descriptor import (
    _most_common_chunk_size,  # type: ignore[reportPrivateUsage]
)


def test_most_common_chunk_size():
    chunk_sizes = [
        {"time": 2, "lat": 3, "lon": 4},
        {"time": 2, "lat": 3, "lon": 4},
        {"time": 3, "lat": 4, "lon": 5},
    ]

    most_common_chunk_size = _most_common_chunk_size(chunk_sizes)

    assert most_common_chunk_size == {"time": 2, "lat": 3, "lon": 4}
