from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf


@pytest.fixture()  # type: ignore[misc]
def example_dataset(tmp_path: Path) -> Path:
    """
    Create a temporary test dataset with audio files, RTTM annotations,
    and sources table.

    Returns:
        Path: Path to the temporary dataset directory
    """
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()

    # Create subdirectories
    flac_dir = dataset_dir / "flac"
    rttm_dir = dataset_dir / "rttm"
    flac_dir.mkdir()
    rttm_dir.mkdir()

    # Generate 20 test files
    file_ids = [f"TEST_{i:04d}" for i in range(1, 21)]
    domains = ["AAA", "BBB", "CCC"]

    sources_data = []

    for i, file_id in enumerate(file_ids):
        # Generate 2.5 seconds of random noise at 16kHz
        sample_rate = 16000
        duration = 2.5
        samples = int(sample_rate * duration)
        audio = np.random.uniform(-0.1, 0.1, samples).astype(np.float32)

        # Save audio file
        audio_path = flac_dir / f"{file_id}.flac"
        sf.write(audio_path, audio, sample_rate)

        # Create RTTM annotation file
        # Speech segments at 0.5s for 0.5s and 1.5s for 0.5s
        rttm_path = rttm_dir / f"{file_id}.rttm"
        with open(rttm_path, "w") as f:
            # First speech segment: start at 0.5s, duration 0.5s
            f.write(f"SPEAKER {file_id} 1 0.500 0.500 <NA> <NA> speaker1 <NA> <NA>\n")
            # Second speech segment: start at 1.5s, duration 0.5s
            f.write(f"SPEAKER {file_id} 1 1.500 0.500 <NA> <NA> speaker1 <NA> <NA>\n")

        # Add to sources data
        domain = domains[i % len(domains)]
        sources_data.append(
            {"file_id": file_id, "lang": "eng", "domain": domain, "source": "Test"}
        )

    # Create sources.tbl file
    sources_df = pd.DataFrame(sources_data)
    sources_df.set_index("file_id", inplace=True)
    sources_path = dataset_dir / "sources.tbl"
    sources_df.to_csv(sources_path, sep="\t")

    return dataset_dir


@pytest.fixture()  # type: ignore[misc]
def noise_dataset(tmp_path: Path) -> Path:
    noise_path = tmp_path / "noise"
    noise_path.mkdir()
    noise_folder1 = noise_path / "noise_folder1"
    noise_folder1.mkdir()
    noise_folder2 = noise_path / "noise_folder2"
    noise_folder2.mkdir()

    sample_rate = 16000
    sf.write(
        noise_folder1 / "noise-0001.wav",
        np.random.uniform(-0.1, 0.1, int(sample_rate * 5.5)).astype(np.float32),
        sample_rate,
    )
    sf.write(
        noise_folder1 / "noise-0002.wav",
        np.random.normal(0, 0.1, int(sample_rate * 5)).astype(np.float32),
        sample_rate,
    )
    sf.write(
        noise_folder2 / "noise-0003.wav",
        np.random.uniform(-0.05, 0.05, int(sample_rate * 6)).astype(np.float32),
        sample_rate,
    )
    return tmp_path
