from pathlib import Path
from shutil import copy

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from dr_sad.data.data_fetching import DOMAIN_SETTINGS, full_file_pull
from dr_sad.data.noise import BaseNoiseBuilder, NoiseBuilder
from dr_sad.data.synthetic import (
    ResampleNoiseBuilder,
    ReverbNoiseBuilder,
    VolumeNoiseBuilder,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NOISE_DIR = DATA_DIR / "musan"

NOISE_DOMAIN_MAP = {
    "restaurant": "babble",
    "socio_field": "reverb",
    "clinical": "noise",
    "meeting": "volume",
    "webvideo": "downsample",
}

DATA_NAME = "dihard"


def main() -> None:
    # load sources.tbl to get file ids and domain info for each file
    sources_tbl_pth = DATA_DIR / DATA_NAME / "sources.tbl"
    with open(sources_tbl_pth) as f:
        sources = pd.read_csv(f, delimiter="\t")

    # create synthetic data directories
    synthetic_data_dir = DATA_DIR / f"{DATA_NAME}_synthetic"
    synthetic_audio_dir = synthetic_data_dir / "flac"
    rttm_dir = synthetic_data_dir / "rttm"
    synthetic_audio_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    # create noise builder for each noise type
    noise_builder_dict: dict[str, BaseNoiseBuilder] = {
        "babble": NoiseBuilder(
            noise_dir=NOISE_DIR / "speech",
            simultaneous=10,
            snr_db=(5, 10),
            seed=67,
        ),
        "noise": NoiseBuilder(
            noise_dir=NOISE_DIR / "noise",
            simultaneous=1,
            snr_db=0,
            seed=1337,
        ),
        "reverb": ReverbNoiseBuilder(
            reverb_sample_filepath=DATA_DIR
            / "elveden-hall-suffolk-england"
            / "examples"
            / "1a_marble_hall.wav",
            sample_rate=16000,
        ),
        "volume": VolumeNoiseBuilder(
            volume_range=(0.03, 3.0),
            volume_change_time=1,
            sample_rate=16000,
            seed=42,
        ),
        "downsample": ResampleNoiseBuilder(downsample_factor=8),
    }

    # loop over files, add noise to each file, and save the synthetic data
    domain_sources = sources[sources["domain"].isin(NOISE_DOMAIN_MAP.keys())]
    for _, row in tqdm(
        domain_sources.iterrows(),
        total=len(domain_sources),
        desc="Processing all files",
    ):
        # get domain information and corresponding noise builder for the file
        domain_name = row["domain"]
        domain_index = DOMAIN_SETTINGS[DATA_NAME]["domains_idx"][domain_name]
        noise_builder = noise_builder_dict[NOISE_DOMAIN_MAP[domain_name]]

        # load data
        file_id = row["file_id"]
        file_data = full_file_pull(file_id, domain_index, str(DATA_DIR / DATA_NAME))
        file_audio = file_data[file_id]["waveforms"]

        # generate noisy sample
        noisy_sample = noise_builder.add_noise(file_audio)
        synthetic_audio_pth = synthetic_audio_dir / f"{file_id}.flac"

        # Save the noisy sample as a FLAC file and copy the corresponding RTTM file
        sf.write(synthetic_audio_pth, noisy_sample, samplerate=16000)
        copy(
            DATA_DIR / DATA_NAME / "rttm" / f"{file_id}.rttm",
            rttm_dir / f"{file_id}.rttm",
        )


if __name__ == "__main__":
    main()
