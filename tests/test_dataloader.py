import numpy as np
import pandas as pd

from dr_sad.data.dataloaders import DrSadDataset


def create_test_dataframe(n_samples: int = 100) -> pd.DataFrame:
    """Create a test DataFrame that mimics the structure of CallHome data.

    Args:
        n_samples: Number of samples to create

    Returns:
        DataFrame with test data including domains for stratified splitting
    """

    # Create test data with different domains for stratified splitting
    domains = np.random.choice(["eng_1", "eng_2", "spa_1", "spa_2"], size=n_samples)

    data = {
        "audio_path": [f"test_audio_{i}.flac" for i in range(n_samples)],
        "domains": domains,
        "annotations": [
            [(0.0, 1.0), (2.0, 3.0)] for _ in range(n_samples)
        ],  # dummy annotations
        "duration": np.random.uniform(10.0, 30.0, n_samples),
        "speaker_count": np.random.randint(1, 4, n_samples),
    }

    return pd.DataFrame(data)


def test_dataset_split_proportions():
    """
    Test that DrSadDataset.train_test_split creates splits with approximately correct
    proportions.
    """
    # Create test data
    test_df = create_test_dataframe(n_samples=100)
    dataset = DrSadDataset(test_df)

    # Test default split ratios
    val_ratio, test_ratio = 0.1, 0.1
    train, val, test = dataset.train_test_split(
        val_ratio=val_ratio, test_ratio=test_ratio, random_state=42
    )

    # Check that splits have approximately correct sizes
    total_size = len(dataset)
    expected_val_size = int(total_size * val_ratio)
    expected_test_size = int(total_size * test_ratio)
    expected_train_size = total_size - expected_val_size - expected_test_size

    # Allow for larger deviations (±3 samples) due to stratification constraints
    tolerance = 3
    assert (
        abs(len(train) - expected_train_size) <= tolerance
    ), f"Train set size {len(train)} too far from expected {expected_train_size}"
    assert (
        abs(len(val) - expected_val_size) <= tolerance
    ), f"Validation set size {len(val)} too far from expected {expected_val_size}"
    assert (
        abs(len(test) - expected_test_size) <= tolerance
    ), f"Test set size {len(test)} too far from expected {expected_test_size}"

    # Check that all samples are accounted for (no overlap, no missing)
    total_split_size = len(train) + len(val) + len(test)
    assert (
        total_split_size == total_size
    ), f"Total split size {total_split_size} != original size {total_size}"


def test_dataset_split_stratification():
    """
    Test that DrSadDataset.train_test_split maintains domain distribution across splits.
    """
    # Create test data with known domain distribution
    test_df = create_test_dataframe(
        n_samples=120
    )  # Divisible by domain count for cleaner testing
    dataset = DrSadDataset(test_df)

    # Get original domain distribution
    original_domains = dataset.data["domains"].value_counts()

    # Split the dataset
    train, val, test = dataset.train_test_split(
        val_ratio=0.2, test_ratio=0.2, random_state=42
    )

    # Check that each split contains all domains (or at least most)
    train_domains = set(train.data["domains"].unique())
    val_domains = set(val.data["domains"].unique())
    test_domains = set(test.data["domains"].unique())
    all_domains = set(original_domains.index)

    # Each split should have most domains
    assert (
        len(train_domains) >= len(all_domains) * 0.7
    ), f"Train split missing too many domains: {train_domains} vs {all_domains}"
    assert (
        len(val_domains) >= len(all_domains) * 0.5
    ), f"Val split missing too many domains: {val_domains} vs {all_domains}"
    assert (
        len(test_domains) >= len(all_domains) * 0.5
    ), f"Test split missing too many domains: {test_domains} vs {all_domains}"


def test_dataset_split_reproducibility():
    """
    Test that DrSadDataset.train_test_split is reproducible with same random_state.
    """
    test_df = create_test_dataframe(n_samples=50)
    dataset = DrSadDataset(test_df)

    # Split twice with same random state
    train1, val1, test1 = dataset.train_test_split(
        val_ratio=0.2, test_ratio=0.2, random_state=42
    )
    train2, val2, test2 = dataset.train_test_split(
        val_ratio=0.2, test_ratio=0.2, random_state=42
    )

    # Check that splits are identical
    pd.testing.assert_frame_equal(train1.data.sort_index(), train2.data.sort_index())
    pd.testing.assert_frame_equal(val1.data.sort_index(), val2.data.sort_index())
    pd.testing.assert_frame_equal(test1.data.sort_index(), test2.data.sort_index())
