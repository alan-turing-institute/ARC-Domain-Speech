from pathlib import Path

import pandas as pd
import soundfile
from datasets import load_dataset
from tqdm import tqdm

from dr_sad.data.callhome_utils import generate_rttm

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
        audio_samples = item["audio"].get_all_samples()
        audio_data = audio_samples.data  # This is the actual tensor
        file_loc = flac_dir / f"CH_{data_row_idx:04d}.flac"
        with soundfile.SoundFile(file_loc, "w", 16000, 1) as f:
            f.write(audio_data.numpy().T)
        row_data = {
            "file_id": file_loc.stem,
            "lang": lang,
            "domain": domain,
            "source": source,
        }
        sources_df.loc[data_row_idx] = row_data
        generate_rttm(item, rttm_dir, file_loc.stem)
        data_row_idx += 1


sources_df.to_csv(data_dir / "sources.tbl", index=False, sep="\t")
