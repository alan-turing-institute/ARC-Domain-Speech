import csv
import itertools
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "callhome"

DOMAIN_LANGUAGES = {"eng": 0, "deu": 1, "spa": 2, "jpn": 3, "zho": 4}


def roundrobin(*iterables: list[Any]) -> Any:
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


def remove_overlap(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Remove overlapping segments by merging them.

    Args:
        segments: List of (start, end) tuples representing segments.

    Returns:
        List of non-overlapping (start, end) tuples.
    """

    # Sort segments by start time
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    merged_segments = [segments[0]]

    for current in segments[1:]:
        last = merged_segments[-1]
        if current[0] <= last[1]:  # Overlap
            merged_segments[-1] = (last[0], max(last[1], current[1]))  # Merge
        else:
            merged_segments.append(current)

    return merged_segments


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
        roundrobin(talking_periods, non_talking_periods)
    )

    for start, end, speaker in combined_periods:
        if speaker == "None" and start >= end:
            combined_periods.remove((start, end, speaker))

    if combined_periods[-1][1] < audio_duration:
        combined_periods.append((combined_periods[-1][1], audio_duration, "None"))

    new_sample["annotations"] = annotations
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

    return sample


def load_callhome(
    data_dir: Path = DATA_DIR,
    domains: list[str] | None = None,
) -> pd.DataFrame:
    if domains is None:
        domains = list(DOMAIN_LANGUAGES.keys())

    data = pd.DataFrame(columns=["waveforms", "annotations", "domains"])
    audio_dir = data_dir / "flac"
    rttm_dir = data_dir / "rttm"
    sources_df = pd.read_csv(data_dir / "sources.tbl", sep="\t", header=0, index_col=0)
    audio_files = list(audio_dir.glob("*.flac"))

    for sample_index, audio_file in tqdm(
        enumerate(sorted(audio_files)),
        total=len(audio_files),
        desc="Loading CallHome data:",
    ):
        file_id = Path(audio_file).stem
        rttm_file = rttm_dir / f"{file_id}.rttm"

        domain = sources_df.loc[file_id, "lang"]
        if domain not in domains:
            continue

        # Load audio
        waveform = soundfile.read(audio_file)[0]
        total_duration = len(waveform) / 16000.0

        # Load RTTM
        timestamps_start = []
        timestamps_end = []
        speakers = []
        with open(rttm_file) as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if row[0] == "SPEAKER":
                    start_time = float(row[3])
                    duration = float(row[4])
                    end_time = start_time + duration
                    speaker_id = row[7]

                    timestamps_start.append(start_time)
                    # times may be longer due to floating point issues
                    timestamps_end.append(min(end_time, total_duration))
                    speakers.append(speaker_id)

        annotations = remove_overlap(
            list(zip(timestamps_start, timestamps_end, strict=True))
        )

        data.loc[sample_index] = {
            "waveforms": waveform,
            "annotations": annotations,
            "domains": DOMAIN_LANGUAGES[domain],
        }

    return data
