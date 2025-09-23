
def roundrobin(*iterables):
    """Recipe from itertools documentation"""
    iterators = [iter(it) for it in iterables]
    while iterators:
        try:
            for it in iterators:
                yield next(it)
        except StopIteration:
            iterators.remove(it)

def add_gaps(old_sample):
    sample = old_sample.copy()
    talking_periods = [(start, end, speaker) for start, end, speaker in zip(sample['timestamps_start'], sample['timestamps_end'], sample['speakers'], strict=True)]
    non_talking_periods = [(start,end,"None") for (_, start,_), (end,_,_) in zip(talking_periods[:-1], talking_periods[1:], strict=True)]
    audio_duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']

    combined_periods = list(roundrobin(talking_periods, non_talking_periods))

    for (start,end,speaker) in combined_periods:
        if speaker == "None" and start >= end:
            combined_periods.remove((start,end,speaker))

    if combined_periods[-1][1] < audio_duration:
        combined_periods.append((combined_periods[-1][1], audio_duration, "None"))

    sample['timestamps_start'] = [start for start, _, _ in combined_periods]
    sample['timestamps_end'] = [end for _, end, _ in combined_periods]
    sample['speakers'] = [speaker for _, _, speaker in combined_periods]
    return sample