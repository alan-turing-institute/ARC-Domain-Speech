"""
Script to generate CallHome dataset files and RTTM annotations.

This is done by downloading the dataset using HuggingFace datasets library.
The generated files are stored in the `data/callhome` directory.

This script takes no arguments.
"""

import logging
from pathlib import Path

import pandas as pd
import soundfile
from datasets import load_dataset
from tqdm import tqdm

from dr_sad.data.utils import generate_rttm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAIN_LANGUAGES = {"eng": 0, "deu": 1, "spa": 2, "jpn": 3, "zho": 4}

sources_df = pd.DataFrame(columns=["file_id", "lang", "domain", "source"])
data_dir = Path(__file__).parent.parent / "data" / "callhome"
data_dir.mkdir(exist_ok=True)
flac_dir = data_dir / "flac"
flac_dir.mkdir(exist_ok=True)
rttm_dir = data_dir / "rttm"
rttm_dir.mkdir(exist_ok=True)

source = "callhome"
domain = "telephone"

data_row_idx = 1
for lang in DOMAIN_LANGUAGES:
    dataset = load_dataset("talkbank/callhome", lang)
    for item in tqdm(dataset["data"], desc=f"Processing {lang}"):
        # Get filename
        file_loc = flac_dir / f"CH_{data_row_idx:04d}.flac"
        # Audio data
        audio_samples = item["audio"].get_all_samples()
        audio_data = audio_samples.data  # This is the actual tensor
        max_timestamp = max(item["timestamps_start"])
        audio_length = audio_data.shape[1] / 16000  # Convert to seconds
        if audio_length < max_timestamp:
            log_msg = (
                f"Warning: Skipping {file_loc.stem} as audio length"
                f" {audio_length:.2f}s"
                f" is less than max timestamp {max_timestamp:.2f}s"
            )
            logger.warning(log_msg)
            continue
        # check rttm is valid
        valid_sample = generate_rttm(item, rttm_dir, file_loc.stem)
        if not valid_sample:
            continue

        # If sample is valid, write the data
        # Metadata
        row_data = {
            "file_id": file_loc.stem,
            "lang": lang,
            "domain": domain,
            "source": source,
        }
        sources_df.loc[data_row_idx] = row_data
        with soundfile.SoundFile(file_loc, "w", 16000, 1) as f:
            f.write(audio_data.numpy().T)

        # Increment index
        data_row_idx += 1


sources_df.to_csv(data_dir / "sources.tbl", index=False, sep="\t")
