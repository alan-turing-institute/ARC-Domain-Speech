import pytest

from dr_sad.utils import get_experiment_name


class TestGetExperimentName:
    """Tests for get_experiment_name function."""

    def test_direct_path_inside_config(self, tmp_path):
        """Test with a direct absolute path to a file inside the config dir."""
        # Create experiment file inside config dir
        exp_config_dir = tmp_path / "configs"
        exp_config_dir.mkdir()
        exp_file = exp_config_dir / "myexp.yaml"
        exp_file.write_text("dummy: true")

        # Pass direct path (string) to the function
        name, path = get_experiment_name(str(exp_file), exp_config_dir)

        assert path == exp_file
        assert name == "myexp"

    def test_relative_in_config_dir(self, tmp_path):
        """Test with a relative path inside the config dir."""
        exp_config_dir = tmp_path / "configs"
        exp_config_dir.mkdir()
        # Create nested directory and file
        nested = exp_config_dir / "nested"
        nested.mkdir()
        nested_file = nested / "exp.yaml"
        nested_file.write_text("x: 1")

        # Pass a relative path inside the config dir
        name, path = get_experiment_name("nested/exp.yaml", exp_config_dir)

        assert path == nested_file
        assert name == "nested/exp"

    def test_external_path(self, tmp_path):
        """Test with a path outside the config dir."""
        # Create a file outside the config dir
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        other_file = other_dir / "outside.yaml"
        other_file.write_text("a: b")

        exp_config_dir = tmp_path / "configs"
        exp_config_dir.mkdir()

        name, path = get_experiment_name(str(other_file), exp_config_dir)

        # For external files the name should be the stem and path the original path
        assert path == other_file
        assert name == "outside"

    def test_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for non-existent files."""
        exp_config_dir = tmp_path / "configs"
        exp_config_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            get_experiment_name("does_not_exist.yaml", exp_config_dir)
