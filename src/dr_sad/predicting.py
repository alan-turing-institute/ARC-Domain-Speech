from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from dr_sad.data.data_fetching import load_data
from dr_sad.data.dataloaders import (
    domain_split_dataloaders,
    from_keys_dataloaders,
    single_domain_dataloaders,
)
from dr_sad.training import create_model

MAIN_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = MAIN_DIR / "configs"


def _save_chunk_safetensors(
    chunk_predictions: dict[str, torch.Tensor], output_path: Path, chunk_idx: int
) -> None:
    """
    Save a chunk of predictions to a temporary safetensors file.

    Args:
        chunk_predictions (dict[str, torch.Tensor]): The predictions for the current
        chunk, mapping keys to tensors.
        output_path (Path): The base path for the output file. The chunk index will be
        appended to this path.
        chunk_idx (int): The index of the current chunk, used to differentiate chunk
        files.
    """

    # Save chunk to temporary file
    chunk_path = output_path.with_suffix(f".chunk_{chunk_idx}.safetensors")
    save_file(chunk_predictions, chunk_path)


def _combine_chunks_safetensors(
    output_path: Path, num_chunks: int
) -> dict[str, torch.Tensor]:
    """
    Combine all chunk files into a single safetensors file.

    Args:
        output_path (Path): The path where the final combined safetensors file will be
            saved.
        num_chunks (int): The number of chunk files to combine.
    """

    if num_chunks == 0:
        err_msg = "No chunks to combine."
        raise ValueError(err_msg)

    if num_chunks == 1:
        # If only one chunk, just rename it
        chunk_path = output_path.with_suffix(".chunk_0.safetensors")
        chunk_path.rename(output_path)
        return

    # Load and combine all chunks
    combined_predictions = {}

    for chunk_idx in range(num_chunks):
        chunk_path = output_path.with_suffix(f".chunk_{chunk_idx}.safetensors")
        chunk_predictions = load_file(chunk_path)

        combined_predictions.update(chunk_predictions)

        # Delete chunk file after loading
        chunk_path.unlink()

    # Save final combined file
    save_file(combined_predictions, output_path)


def save_predictions_chunked(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_path: Path,
    chunk_size: int = 50,
) -> None:
    """
    Save predictions in chunks using safetensors format to avoid memory issues.

    Args:
        model (torch.nn.Module): The model used to generate predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader providing input batches.
        output_path (Path): Path to the output file (without chunk suffix).
        chunk_size (int, optional): Number of predictions per chunk. Defaults to 50.

    Returns:
        dict[str, torch.Tensor]: Combined predictions from all chunks.
    """

    model.eval()

    chunk_predictions = {}
    chunk_count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Get predictions for this batch
        file_ids = batch["file_id"]
        prediction = model.predict_step(batch, batch_idx)

        # Store predictions for current batch
        for index, file_id in enumerate(file_ids):
            chunk_predictions[file_id] = prediction[index].cpu()

        # Save chunk when we reach chunk_size batches or at the end
        if len(chunk_predictions) >= chunk_size or batch_idx == len(dataloader) - 1:
            _save_chunk_safetensors(chunk_predictions, output_path, chunk_count)
            chunk_predictions.clear()  # Clear to free memory
            chunk_count += 1

            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Combine all chunks into final file
    _combine_chunks_safetensors(output_path, chunk_count)


def load_model_eval(
    model_path: Path | str,
    model_cfg: dict[str, str | int | float],
    trainer_cfg: dict[str, str | int | float],
    data_cfg: dict[str, str | int | float] | None = None,
) -> torch.nn.Module:
    """
    Loads a model from a safetensors file and prepares it for evaluation.

    Args:
        model_path (Path or str): Path to the safetensors file containing the model
         weights.
        model_cfg (dict): Configuration dictionary for the model architecture.
        trainer_cfg (dict): Configuration dictionary for the trainer settings.
        data_cfg (dict, optional): Configuration dictionary for the dataset, if needed
            for model creation. Defaults to None.

    Returns:
        torch.nn.Module: The model loaded with weights and set to evaluation mode.
    """
    weightless_model = create_model(
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        data_cfg=data_cfg,
    )

    # Load the model state dict from the safetensors file
    state_dict = load_file(model_path)
    weightless_model.load_state_dict(state_dict)

    # Set model to evaluation mode
    return weightless_model.eval()


def load_data_eval(
    data_cfg: dict[str, str],
    data_split: dict[str, list[str]],
    trainer_cfg: dict[str, str | int | float],
    exp_config: dict[str, str | int | float],
    exclude_domain: int | None = None,
    train_domain: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Load evaluation DataLoaders based on the provided configuration and data split.

    Args:
        data_cfg (dict[str, str]): Configuration dictionary for the dataset,
            must include 'name', 'split_name', and 'domain_type' keys.
        data_split (dict[str, list[str]]): Pre-loaded data split
            dictionary.
        trainer_cfg (dict[str, str | int | float]): Trainer configuration dictionary,
            must include 'num_workers' and 'batch_size' keys.
        exp_config (dict[str, str | int | float]): Experiment configuration dictionary,
            must include 'random_seed' key.
        exclude_domain (int | None, optional): Domain index to exclude from evaluation.
            Only used if 'domain_type' is 'exclude_one'. Defaults to None.
        train_domain (int | None, optional): Domain index to use for training when
            'domain_type' is 'single_domain'. Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader | None]: A tuple containing:
            - validation_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set or the included
                domain.
            - excluded_loader (DataLoader | None): DataLoader for the excluded domain if
                applicable,
              otherwise None.
    """
    # load data
    data = load_data(data_cfg["name"], num_workers=int(trainer_cfg["num_workers"]))

    if not isinstance(data_split, dict):
        msg = (  # type: ignore[unreachable]
            "data_split must be provided as a dictionary with "
            "'train', 'val', and 'test' keys."
        )
        raise ValueError(msg)

    if data_cfg["domain_type"] != "exclude_one":
        if exclude_domain is not None:
            err_msg = (
                "Cannot exclude domain when domain_type is set to 'all'"
                " or 'single_domain'."
            )
            raise ValueError(err_msg)

        if data_cfg["domain_type"] == "all" and train_domain is not None:
            err_msg = "Cannot specify train_domain when domain_type is set to 'all'."
            raise ValueError(err_msg)

        if data_cfg["domain_type"] == "single_domain":
            if train_domain is None:
                err_msg = (
                    "Must specify --train_domain when domain_type is 'single_domain'."
                )
                raise ValueError(err_msg)
            _, validation_loader, test_loader = single_domain_dataloaders(
                data,
                train_keys=data_split["train"],
                val_keys=data_split["val"],
                test_keys=data_split["test"],
                domain=train_domain,
                batch_size=int(trainer_cfg["batch_size"]),
                random_seed=int(exp_config["random_seed"]),
            )

        elif data_cfg["domain_type"] == "all":
            _, validation_loader, test_loader = from_keys_dataloaders(
                data,
                train_keys=data_split["train"],
                val_keys=data_split["val"],
                test_keys=data_split["test"],
                batch_size=int(trainer_cfg["batch_size"]),
                random_seed=int(exp_config["random_seed"]),
            )

        else:
            err_msg = f"Unknown domain_type option: {data_cfg['domain_type']}"
            raise ValueError(err_msg)

        return validation_loader, test_loader, None

    if data_cfg["domain_type"] == "exclude_one":
        if exclude_domain is None:
            err_msg = "Must specify --exclude_domain when domain_type is 'exclude_one'."
            raise ValueError(err_msg)
        # Use domain_split_dataloaders to exclude the specified domain
        _, validation_loader, test_loader, domain_loader = domain_split_dataloaders(
            data,
            train_keys=data_split["train"],
            val_keys=data_split["val"],
            test_keys=data_split["test"],
            domain=exclude_domain,
            batch_size=int(trainer_cfg["batch_size"]),
            random_seed=int(exp_config["random_seed"]),
        )

        return validation_loader, test_loader, domain_loader

    err_msg = f"Unknown domain_type option: {data_cfg['domain_type']}"
    raise ValueError(err_msg)
