import itertools
from typing import Any


def roundrobin(*iterables: list[list[Any]]) -> Any:
    """
    Round-robin iterator for multiple input iterables.

    Returns:
        An iterator that yields elements from each iterable in turn.

    Yields:
        Elements from the input iterables.
    """
    iterators = [iter(it) for it in iterables]
    while iterators:
        try:
            for it in iterators:
                yield next(it)
        except StopIteration:
            iterators.remove(it)


def fill_gaps(old_sample: dict[str, Any]) -> dict[str, Any]:
    """
    Adds gaps in the audio sample where there is no speech.

    Args:
        old_sample: The original audio sample.

    Returns:
        The modified audio sample with gaps added.
    """
    new_sample = old_sample.copy()
    talking_periods = list(
        zip(
            new_sample["timestamps_start"],
            new_sample["timestamps_end"],
            new_sample["speakers"],
            strict=True,
        )
    )
    non_talking_periods = [
        (start, end, "None")
        for (_, start, _), (end, _, _) in itertools.pairwise(talking_periods)
    ]

    audio_duration = (
        len(new_sample["audio"]["array"]) / new_sample["audio"]["sampling_rate"]
    )

    combined_periods: list[tuple[float, float, str]] = list(
        roundrobin(talking_periods, non_talking_periods)
    )

    for start, end, speaker in combined_periods:
        if speaker == "None" and start >= end:
            combined_periods.remove((start, end, speaker))

    if combined_periods[-1][1] < audio_duration:
        combined_periods.append((combined_periods[-1][1], audio_duration, "None"))

    new_sample["timestamps_start"] = [start for start, _, _ in combined_periods]
    new_sample["timestamps_end"] = [end for _, end, _ in combined_periods]
    new_sample["speakers"] = [speaker for _, _, speaker in combined_periods]
    return new_sample


def call_home_preprocess(sample: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """
    Preprocess the CallHome data by combining consecutive speaking segments.
    This ensures labels alternate between 0 (silence) and 1 (speech).

    Args:
        old_sample: The original audio sample.

    Returns:
        The modified audio sample with combined speaking segments and gaps.
    """

    # First, fill gaps to get alternating speech/silence
    sample = fill_gaps(sample)

    # Extract segments and create initial labels
    speakers = [0 if spk == "None" else 1 for spk in sample["speakers"]]

    segments = list(
        zip(sample["timestamps_start"], sample["timestamps_end"], strict=True)
    )

    combined_labels = []
    combined_segments = []

    current_label = speakers[0]
    current_start = segments[0][0]

    for i, (label, (start, _)) in enumerate(zip(speakers, segments, strict=True)):
        if label != current_label:
            # Label changed, so save the previous segment
            combined_segments.append((current_start, segments[i - 1][1]))
            combined_labels.append(current_label)
            current_label = label
            current_start = start

    # Handle the final segment
    combined_segments.append((current_start, segments[-1][1]))
    combined_labels.append(current_label)

    sample["segments"] = combined_segments
    sample["labels"] = combined_labels

    return sample
