__all__ = ("load_data", "remove_overlap")

import csv
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TypedDict

import pandas as pd
import soundfile
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


class DomainSetting(TypedDict):
    file_name: str
    domain_column: str
    domains_idx: dict[str, int]


DOMAIN_SETTINGS: dict[str, DomainSetting] = {
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


def full_file_pull(file_id: str, d_idx: int, data_dir_loc: str) -> dict[str, dict]:  # type: ignore[type-arg]
    """This is intended to be a mapping function that takes a set of parameters
    and returns a dictionary with the audio data, annotations, and domain index.

    It is written this way to facilitate parallel processing.
    Args:
        file_id (str): The ID of the file to process.
        d_idx (int): The domain index.
        data_dir_loc (str): The location of the data directory.

    Returns:
        data (dict): A dictionary with the following structure:
            file_id: {
                "waveforms": waveform (np.ndarray),
                "annotations": annotations (list of tuples),
                "domains": d_idx (int),
            }
    """

    audio_file = Path(data_dir_loc) / "flac" / f"{file_id}.flac"
    rttm_file = Path(data_dir_loc) / "rttm" / f"{file_id}.rttm"

    # Load audio
    waveform = soundfile.read(audio_file)[0]
    total_duration = len(waveform) / 16000.0

    # Load RTTM
    timestamps_start = []
    timestamps_end = []
    # speakers = []  # Not used currently, needed for diarisation
    with open(rttm_file) as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            if row[0] == "SPEAKER":
                start_time = float(row[3])
                duration = float(row[4])
                end_time = start_time + duration
                # speaker_id = row[7]

                timestamps_start.append(start_time)
                # times may be longer due to floating point issues
                timestamps_end.append(min(end_time, total_duration))
                # speakers.append(speaker_id)

    annotations = remove_overlap(
        list(zip(timestamps_start, timestamps_end, strict=True))
    )

    return {
        file_id: {
            "waveforms": waveform,
            "annotations": annotations,
            "domains": d_idx,
        }
    }


def _wrapped_full_file_pull(args: tuple) -> dict[str, dict]:  # type: ignore[type-arg]
    return full_file_pull(*args)


def load_data(
    data_choice: str | None,
    data_set_path: str | Path | None = None,
    domain_column: str | None = None,
    domains_idx: dict[str, int] | None = None,
    num_workers: int = 1,
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
        num_workers (int): Number of parallel workers to use for loading data.
            Defaults to 1 which means no parallelism.

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
        d_idx_map: dict[str, int] = domains_idx

    else:
        if data_choice not in DOMAIN_SETTINGS:
            msg = f"Unknown data_choice: {data_choice}"
            raise ValueError(msg)
        settings = DOMAIN_SETTINGS[data_choice]
        data_dir = DATA_DIR / str(settings["file_name"])
        d_column = str(settings["domain_column"])
        d_idx_map = settings["domains_idx"]

    if not data_dir.exists():
        msg = f"Data directory {data_dir} does not exist."
        raise ValueError(msg)

    sources_df = pd.read_csv(data_dir / "sources.tbl", sep="\t", header=0, index_col=0)
    file_ids = sorted([af.stem for af in (data_dir / "flac").glob("*.flac")])

    domain_indexes = []
    for file_id in file_ids:
        domain = sources_df.loc[file_id, d_column]
        if domain not in d_idx_map:
            msg = f"Unknown domain '{domain}' for file '{file_id}'."
            raise ValueError(msg)
        domain_indexes.append(d_idx_map[domain])

    args_list = [
        (file_id, d_idx, str(data_dir))
        for file_id, d_idx in zip(file_ids, domain_indexes, strict=True)
    ]

    dataset_list = []
    if num_workers < 1:
        msg = f"num_workers must be at least 1, was {num_workers}"
        raise ValueError(msg)
    if num_workers == 1:
        print("Loading data without parallel workers.")
        for args in tqdm(
            args_list,
            total=len(file_ids),
            desc="Loading Audio data",
        ):
            dataset_list.append(full_file_pull(*args))
    else:
        print(f"Loading data with {num_workers} workers.")
        with ThreadPool(num_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_wrapped_full_file_pull, args_list),
                total=len(args_list),
                desc="Loading Audio data",
            ):
                dataset_list.append(result)

    dataset = {k: v for d in dataset_list for k, v in d.items()}
    data = pd.DataFrame.from_dict(dataset, orient="index")
    return data.sort_index()
