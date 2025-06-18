"""Tests for visualization functions."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

import mne
from mne.preprocessing import ICA
from autoclean.functions.visualization import (
    plot_raw_comparison,
    plot_ica_components,
    plot_psd_topography,
    generate_processing_report,
    create_processing_summary
)


@pytest.fixture
def mock_raw():
    """Create a mock raw object for testing."""
    raw = MagicMock(spec=mne.io.BaseRaw)
    raw.ch_names = [f"EEG_{i:03d}" for i in range(32)]
    raw.info = {"sfreq": 500.0}
    raw.times = np.linspace(0, 10, 5000)  # 10 seconds
    raw.get_data.return_value = np.random.randn(32, 5000) * 1e-6  # Realistic EEG scale
    return raw


@pytest.fixture
def mock_ica():
    """Create a mock ICA object for testing."""
    ica = MagicMock(spec=ICA)
    ica.n_components_ = 10
    ica.plot_components.return_value = plt.figure()
    return ica


@pytest.fixture
def processing_steps():
    """Create sample processing steps for testing."""
    return [
        {
            'step_name': 'Filtering',
            'parameters': {'low_freq': 0.1, 'high_freq': 50.0},
            'execution_time': 2.3,
            'description': 'Applied bandpass filter'
        },
        {
            'step_name': 'Bad Channel Detection',
            'parameters': {'method': 'correlation'},
            'execution_time': 1.5,
            'description': 'Detected and interpolated bad channels'
        }
    ]


class TestPlotRawComparison:
    """Test raw data comparison plotting."""

    def test_basic_functionality(self, mock_raw):
        """Test basic raw comparison plotting."""
        # Create second mock with same properties
        mock_raw_cleaned = MagicMock(spec=mne.io.BaseRaw)
        mock_raw_cleaned.ch_names = mock_raw.ch_names
        mock_raw_cleaned.info = mock_raw.info
        mock_raw_cleaned.times = mock_raw.times
        mock_raw_cleaned.get_data.return_value = mock_raw.get_data.return_value * 0.8  # Simulated cleaning
        
        fig = plot_raw_comparison(mock_raw, mock_raw_cleaned)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_input_validation(self):
        """Test input validation."""
        raw = MagicMock(spec=mne.io.BaseRaw)
        
        # Test non-Raw objects
        with pytest.raises(TypeError):
            plot_raw_comparison("not_raw", raw)
        
        with pytest.raises(TypeError):
            plot_raw_comparison(raw, "not_raw")


class TestPlotIcaComponents:
    """Test ICA component plotting."""

    def test_basic_functionality(self, mock_ica):
        """Test basic ICA component plotting."""
        fig = plot_ica_components(mock_ica)
        
        assert isinstance(fig, plt.Figure)
        mock_ica.plot_components.assert_called_once()
        plt.close(fig)

    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(TypeError):
            plot_ica_components("not_ica")


class TestPlotPsdTopography:
    """Test PSD topography plotting."""

    @patch('mne.io.BaseRaw.compute_psd')
    def test_basic_functionality(self, mock_compute_psd, mock_raw):
        """Test basic PSD topography plotting."""
        # Mock spectrum object
        mock_spectrum = MagicMock()
        mock_spectrum.plot_topomap.return_value = plt.figure()
        mock_compute_psd.return_value = mock_spectrum
        
        fig = plot_psd_topography(mock_raw)
        
        assert isinstance(fig, plt.Figure)
        mock_compute_psd.assert_called_once()
        mock_spectrum.plot_topomap.assert_called_once()
        plt.close(fig)

    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(TypeError):
            plot_psd_topography("not_raw")


class TestGenerateProcessingReport:
    """Test processing report generation."""

    def test_basic_functionality(self, mock_raw, processing_steps):
        """Test basic report generation."""
        mock_raw_cleaned = MagicMock(spec=mne.io.BaseRaw)
        mock_raw_cleaned.ch_names = mock_raw.ch_names
        mock_raw_cleaned.info = mock_raw.info
        mock_raw_cleaned.times = mock_raw.times
        mock_raw_cleaned.get_data.return_value = mock_raw.get_data.return_value
        mock_raw_cleaned.annotations = MagicMock()
        mock_raw_cleaned.annotations.__len__.return_value = 5
        
        # Mock annotations for both raw objects
        mock_raw.annotations = MagicMock()
        mock_raw.annotations.__len__.return_value = 0
        mock_raw_cleaned.annotations.__iter__.return_value = [
            {'description': 'BAD_segment1'},
            {'description': 'BAD_segment2'},
            {'description': 'other_annotation'}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            result_path = generate_processing_report(
                mock_raw, mock_raw_cleaned, processing_steps, output_path
            )
            
            assert Path(result_path).exists()
            assert result_path == str(output_path)

    def test_input_validation(self, processing_steps):
        """Test input validation."""
        raw = MagicMock(spec=mne.io.BaseRaw)
        
        # Test non-Raw objects
        with pytest.raises(TypeError):
            generate_processing_report("not_raw", raw, processing_steps, "output.html")
        
        with pytest.raises(TypeError):
            generate_processing_report(raw, "not_raw", processing_steps, "output.html")


class TestCreateProcessingSummary:
    """Test processing summary creation."""

    def test_basic_functionality(self, processing_steps):
        """Test basic summary creation."""
        summary = create_processing_summary(processing_steps)
        
        assert isinstance(summary, dict)
        assert summary['total_steps'] == 2
        assert summary['total_time'] == 3.8  # 2.3 + 1.5
        assert summary['steps'] == processing_steps
        assert 'generated_at' in summary