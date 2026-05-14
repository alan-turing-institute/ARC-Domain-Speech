import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dr_sad.analysis import load_audio_and_annotations
from dr_sad.data.data_fetching import remove_overlap

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dihard"


def main() -> None:
    sources_path = DATA_DIR / "sources.tbl"
    sources = np.loadtxt(sources_path, dtype=str, delimiter="\t", skiprows=1)

    domain_file_id = {
        file_id.item(): domain.item() for file_id, _, domain, _ in sources
    }
    all_ratios = dict.fromkeys(domain_file_id, 0.0)

    for file_id in tqdm(all_ratios, desc="Calculating speech ratios"):
        audio, sample_rate, speech_segments = load_audio_and_annotations(
            file_id,
            DATA_DIR,
        )
        merged = remove_overlap(speech_segments)
        all_ratios[file_id] = (
            100 * sum(end - start for start, end in merged) / (len(audio) / sample_rate)
        )

    domain_ratios: dict[str, list[float]] = {}
    for file_id, ratio in all_ratios.items():
        domain = domain_file_id[file_id]
        domain_ratios.setdefault(domain, []).append(ratio)

    rows = []
    latex_domains = []
    latex_values = []
    for domain, ratios in sorted(domain_ratios.items()):
        arr = np.array(ratios)
        print(f"{domain}: {arr.mean():.2f}% +/- {arr.std():.2f}%")
        rows.append(f"{domain}\t{arr.mean():.2f}\t{arr.std():.2f}")
        latex_domains.append(domain.replace("_", " ").title())
        latex_values.append(f"${arr.mean():.2f}({round(arr.std() * 100)})$")

    out_path = DATA_DIR / "speech_ratios.tsv"
    with open(out_path, "w") as f:
        f.write("domain\tmean\tstd\n")
        f.write("\n".join(rows))
    print(f"\nSaved to {out_path}")

    data_rows = "\n".join(
        f"        {d} & {v} \\\\"
        for d, v in zip(latex_domains, latex_values, strict=True)
    )
    latex_table = (
        "\\begin{table}[h]\n"
        "    \\centering\n"
        "    \\caption{Speech ratio by domain in the DIHARD dataset.}\n"
        "    \\label{tab:speech_ratios}\n"
        "    \\begin{tabular}{lc}\n"
        "        \\hline\n"
        "        Domain & Speech Ratio (\\%) \\\\\n"
        "        \\hline\n"
        f"{data_rows}\n"
        "        \\hline\n"
        "    \\end{tabular}\n"
        "\\end{table}\n"
    )
    latex_path = DATA_DIR / "speech_ratios.tex"
    latex_path.write_text(latex_table)
    print(f"Saved LaTeX table to {latex_path}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Calculate speech ratios for each file in dataset."
    )
    main()
