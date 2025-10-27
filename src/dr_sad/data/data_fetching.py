import csv
from pathlib import Path

import pandas as pd
import soundfile
from tqdm import tqdm

__all__ = ("load_data", "remove_overlap")

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"

DOMAIN_SETTINGS = {
    "callhome": {
        "file_name": "callhome",
        "domain_column": "lang",
        "domains_idx": {"eng": 0, "deu": 1, "spa": 2, "jpn": 3, "zho": 4},
    },
    "dihard": {
        "file_name": "dihard",
        "domain_column": "domain",
        "domains_idx": {
            "audiobooks": 0,
            "restaurant": 1,
            "clinical": 2,
            "court": 3,
            "maptask": 4,
            "meeting": 5,
            "socio_field": 6,
            "socio_lab": 7,
            "webvideo": 8,
            "broadcast_interview": 9,
            "dinner": 10,
        },
    },
    "test": {
        "file_name": "test_dataset",
        "domain_column": "domain",
        "domains_idx": {"AAA": 0, "BBB": 1, "CCC": 2},
    },
}


def remove_overlap(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Remove overlapping segments by merging them. This converts a rttm-style list of
    (start, end) tuples into non-overlapping segments to a format suitable for
    speech activity detection.

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


def load_data(
    data_choice: str | None,
    data_set_path: str | Path | None = None,
    domain_column: str | None = None,
    domains_idx: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Load a dataset from the specified source. Supports predefined datasets
    ("callhome" and "dihard") or custom datasets by specifying the path and
    domain information.

    Args:
        data_choice (str | None): Predefined dataset choice. Currently supports
            "callhome" and "dihard" using default settings. If None, data_set_path,
            domain_column, and domains_idx must be provided.
        data_set_path (str | Path, optional): Path to the dataset directory.
            Required if data_choice is None.
        domain_column (str, optional): Column name in sources.tbl that contains domain
            information. Required if data_choice is None.
        domains_idx (dict[str, int], optional): Mapping from domain names to integer
            indices. Required if data_choice is None.

    Returns:
        pd.DataFrame: The loaded dataset. This will contain the columns
            "waveforms", "annotations", and "domains".
    """
    if data_choice is None:
        if data_set_path is None:
            msg = "Either data_choice or data_set_path must be provided."
            raise ValueError(msg)
        data_dir: Path = Path(data_set_path)

        if domain_column is None:
            msg = "domain_column must be provided if data_choice is None."
            raise ValueError(msg)
        d_column: str = domain_column

        if domains_idx is None:
            msg = "domains_idx must be provided if data_choice is None."
            raise ValueError(msg)
        d_idx: dict[str, int] = domains_idx

    else:
        if data_choice not in DOMAIN_SETTINGS:
            msg = f"Unknown data_choice: {data_choice}"
            raise ValueError(msg)
        settings = DOMAIN_SETTINGS[data_choice]
        data_dir = DATA_DIR / str(settings["file_name"])
        d_column = str(settings["domain_column"])
        d_idx = settings["domains_idx"]  # type: ignore[assignment]

    if not data_dir.exists():
        msg = f"Data directory {data_dir} does not exist."
        raise ValueError(msg)

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

        # Get domain
        domain = sources_df.loc[file_id, d_column]
        if domain not in d_idx:
            msg = f"Unknown domain '{domain}' for file '{file_id}'."
            raise ValueError(msg)

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

        # Store in DataFrame
        data.loc[sample_index] = {
            "waveforms": waveform,
            "annotations": annotations,
            "domains": d_idx[domain],
        }

    return data
