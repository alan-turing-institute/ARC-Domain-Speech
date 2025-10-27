"""CallHome dataset utilities.

This module provides functions for parsing CallHome, but are not required
for general usage of the DR-SAD framework.
"""

import itertools
from pathlib import Path
from typing import Any

from dr_sad.data.data_fetching import remove_overlap

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "callhome"

DOMAIN_LANGUAGES = {"eng": 0, "deu": 1, "spa": 2, "jpn": 3, "zho": 4}


def _roundrobin(*iterables: list[Any]) -> Any:
    """
    Round-robin iterator for multiple input iterables.

    Returns:
        An iterator that yields elements from each iterable in turn.

    Yields:
        Elements from the input iterables.
    """
    iterators = [iter(it) for it in iterables]
    while iterators:
        # Track which iterators to remove after this round
        to_remove = []

        # Go through each iterator and try to get the next item
        for i, it in enumerate(iterators):
            try:
                yield next(it)
            except StopIteration:
                # Mark this iterator for removal, but continue with others
                to_remove.append(i)

        # Remove exhausted iterators (in reverse order to maintain valid indices)
        for i in reversed(to_remove):
            iterators.pop(i)


def _fill_gaps(old_sample: dict[str, Any]) -> dict[str, Any]:
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
    annotations = [
        (talk_period[0], talk_period[1])
        for talk_period in talking_periods
        if talk_period[2] != "None"
    ]

    annotations = remove_overlap(annotations)

    non_talking_periods = [
        (start, end, "None")
        for (_, start, _), (end, _, _) in itertools.pairwise(talking_periods)
    ]

    audio_duration = (
        len(new_sample["audio"]["array"]) / new_sample["audio"]["sampling_rate"]
    )

    combined_periods: list[tuple[float, float, str]] = list(
        _roundrobin(talking_periods, non_talking_periods)
    )

    # Remove invalid non-talking periods without modifying the list during iteration
    combined_periods = [
        (start, end, speaker)
        for (start, end, speaker) in combined_periods
        if not (speaker == "None" and start >= end)
    ]
    if combined_periods[-1][1] < audio_duration:
        combined_periods.append((combined_periods[-1][1], audio_duration, "None"))

    new_sample["annotations"] = annotations
    new_sample["timestamps_start"] = [start for start, _, _ in combined_periods]
    new_sample["timestamps_end"] = [end for _, end, _ in combined_periods]
    new_sample["speakers"] = [speaker for _, _, speaker in combined_periods]
    return new_sample


def _call_home_preprocess(sample: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """
    Preprocess the CallHome data by combining consecutive speaking segments.
    This ensures labels alternate between 0 (silence) and 1 (speech).

    Args:
        old_sample: The original audio sample.

    Returns:
        The modified audio sample with combined speaking segments and gaps.
    """

    # First, fill gaps to get alternating speech/silence
    sample = _fill_gaps(sample)

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

    return sample
