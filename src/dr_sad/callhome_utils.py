import itertools


def roundrobin(*iterables: list) -> any:
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


def add_gaps(old_sample: dict) -> dict:
    """
    Adds gaps in the audio sample where there is no speech.

    Args:
        old_sample: The original audio sample.

    Returns:
        The modified audio sample with gaps added.
    """
    new_sample = old_sample.copy()
    talking_periods = [
        (start, end, speaker)
        for start, end, speaker in zip(
            new_sample["timestamps_start"],
            new_sample["timestamps_end"],
            new_sample["speakers"],
            strict=True,
        )
    ]
    non_talking_periods = [
        (start, end, "None")
        for (_, start, _), (end, _, _) in itertools.pairwise(talking_periods)
    ]
    audio_duration = (
        len(new_sample["audio"]["array"]) / new_sample["audio"]["sampling_rate"]
    )

    combined_periods = list(roundrobin(talking_periods, non_talking_periods))

    for start, end, speaker in combined_periods:
        if speaker == "None" and start >= end:
            combined_periods.remove((start, end, speaker))

    if combined_periods[-1][1] < audio_duration:
        combined_periods.append((combined_periods[-1][1], audio_duration, "None"))

    new_sample["timestamps_start"] = [start for start, _, _ in combined_periods]
    new_sample["timestamps_end"] = [end for _, end, _ in combined_periods]
    new_sample["speakers"] = [speaker for _, _, speaker in combined_periods]
    return new_sample
