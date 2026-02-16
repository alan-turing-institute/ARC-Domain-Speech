from pathlib import Path
from shutil import copy

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from dr_sad.data.data_fetching import DOMAIN_SETTINGS, full_file_pull
from dr_sad.data.noise import NoiseBuilder
from dr_sad.data.synthetic import (
    BaseNoiseBuilder,
    IdentityNoiseBuilder,
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


def main(data_name: str, domains: list[str | int]):
    # get domain name to index mapping for the dataset
    domain_name_idx_map = DOMAIN_SETTINGS[data_name]["domains_idx"]
    domain_idx_name_map = {idx: name for name, idx in domain_name_idx_map.items()}

    # load sources.tbl to get file ids and domain info for each file
    sources_tbl_pth = DATA_DIR / data_name / "sources.tbl"
    with open(sources_tbl_pth) as f:
        sources = pd.read_csv(f, delimiter="\t")

    # create synthetic data directories
    synthetic_data_dir = DATA_DIR / f"{data_name}_synthetic"
    synthetic_audio_dir = synthetic_data_dir / "flac"
    rttm_dir = synthetic_data_dir / "rttm"
    synthetic_audio_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)

    # create noise builder for each noise type
    noise_builder_dict: dict[str, NoiseBuilder | BaseNoiseBuilder] = {
        "babble": NoiseBuilder(
            noise_dir=NOISE_DIR / "speech", simultaneous=10, snr_db=(5, 10)
        ),
        "noise": NoiseBuilder(noise_dir=NOISE_DIR / "noise", simultaneous=1, snr_db=0),
        "reverb": ReverbNoiseBuilder(ratio=2.5, reverb_volume=0.3),
        "volume": VolumeNoiseBuilder(volume_range=(0.2, 0.6), n_volume_changes=25),
        "downsample": ResampleNoiseBuilder(downsample_factor=4),
    }

    # loop over domains and files, add noise to each file, and save the synthetic data
    for domain in domains:
        # get domain name and index
        domain_name = domain_idx_name_map[domain] if isinstance(domain, int) else domain
        domain_index = domain_name_idx_map[domain_name]

        domain_sources = sources[sources["domain"] == domain_name]
        noise_builder = noise_builder_dict.get(
            NOISE_DOMAIN_MAP.get(domain_name, "noise"), noise_builder_dict["noise"]
        )

        for _, row in tqdm(
            domain_sources.iterrows(),
            total=len(domain_sources),
            desc=f"Processing domain {domain_name}",
        ):
            file_id = row["file_id"]
            file_data = full_file_pull(file_id, domain_index, str(DATA_DIR / data_name))
            file_audio = file_data[file_id]["waveforms"]
            noisy_sample = noise_builder.add_noise(file_audio)
            synthetic_audio_pth = synthetic_audio_dir / f"{file_id}.flac"

            # Save the noisy sample as a FLAC file and copy the corresponding RTTM file
            sf.write(synthetic_audio_pth, noisy_sample, samplerate=16000)
            copy(
                DATA_DIR / data_name / "rttm" / f"{file_id}.rttm",
                rttm_dir / f"{file_id}.rttm",
            )


if __name__ == "__main__":
    main(
        data_name="dihard",
        domains=[
            "socio_field",
            "meeting",
            "clinical",
            "restaurant",
            "webvideo",
        ],
    )
