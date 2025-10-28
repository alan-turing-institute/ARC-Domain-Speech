"""
Script to generate CallHome dataset files and RTTM annotations.

This is done by downloading the dataset using HuggingFace datasets library.
The generated files are stored in the `data/callhome` directory.

This script takes no arguments.
"""

from pathlib import Path

import pandas as pd
import soundfile
from datasets import load_dataset
from tqdm import tqdm

from dr_sad.data.utils import generate_rttm

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
        # get filename
        file_loc = flac_dir / f"CH_{data_row_idx:04d}.flac"
        # check rttm is valid
        valid_sample = generate_rttm(item, rttm_dir, file_loc.stem)
        if not valid_sample:
            continue

        # if sample valid write the data
        # metadata
        row_data = {
            "file_id": file_loc.stem,
            "lang": lang,
            "domain": domain,
            "source": source,
        }
        sources_df.loc[data_row_idx] = row_data
        # audio data
        file_loc = flac_dir / f"CH_{data_row_idx:04d}.flac"
        audio_samples = item["audio"].get_all_samples()
        audio_data = audio_samples.data  # This is the actual tensor
        with soundfile.SoundFile(file_loc, "w", 16000, 1) as f:
            f.write(audio_data.numpy().T)

        # increment index
        data_row_idx += 1


sources_df.to_csv(data_dir / "sources.tbl", index=False, sep="\t")
