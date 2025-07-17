"""Tests for ICA functions."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import mne
from mne.preprocessing import ICA
from autoclean.functions.ica import (
    fit_ica,
    classify_ica_components,
    apply_ica_rejection
)


@pytest.fixture
def mock_raw():
    """Create a mock raw object for testing."""
    raw = MagicMock(spec=mne.io.BaseRaw)
    raw.info = {"sfreq": 500.0, "nchan": 64}
    raw.get_data.return_value = np.random.randn(64, 10000)
    raw.copy.return_value = raw
    return raw


@pytest.fixture
def mock_ica():
    """Create a mock ICA object for testing."""
    ica = MagicMock(spec=ICA)
    ica.n_components_ = 20
    ica.copy.return_value = ica
    ica.exclude = []
    
    # Mock ICLabel results
    ica.labels_ = {
        "iclabel": {
            "y_pred_proba": np.random.rand(20, 7)  # 20 components, 7 categories
        }
    }
    
    return ica


class TestFitIca:
    """Test ICA fitting functionality."""

    @patch('autoclean.functions.ica.ica_processing.ICA')
    def test_basic_functionality(self, mock_ica_class, mock_raw):
        """Test basic ICA fitting."""
        mock_ica_instance = MagicMock(spec=ICA)
        mock_ica_class.return_value = mock_ica_instance
        
        result = fit_ica(mock_raw)
        
        # Should create and fit ICA
        mock_ica_class.assert_called_once()
        mock_ica_instance.fit.assert_called_once()
        assert result == mock_ica_instance

    @patch('autoclean.functions.ica.ica_processing.ICA')
    def test_custom_parameters(self, mock_ica_class, mock_raw):
        """Test ICA fitting with custom parameters."""
        mock_ica_instance = MagicMock(spec=ICA)
        mock_ica_class.return_value = mock_ica_instance
        
        result = fit_ica(
            mock_raw,
            n_components=15,
            method="infomax",
            max_iter=1000,
            random_state=42,
            picks="eeg"
        )
        
        # Check that parameters were passed correctly
        mock_ica_class.assert_called_once()
        call_args = mock_ica_class.call_args[1]
        assert call_args["n_components"] == 15
        assert call_args["method"] == "infomax"
        assert call_args["max_iter"] == 1000
        assert call_args["random_state"] == 42
        
        mock_ica_instance.fit.assert_called_once()

    def test_input_validation(self):
        """Test input validation for ICA fitting."""
        # Test non-Raw object
        with pytest.raises(TypeError):
            fit_ica("not_raw")
        
        # Test invalid method
        raw = MagicMock(spec=mne.io.BaseRaw)
        with pytest.raises(ValueError):
            fit_ica(raw, method="invalid_method")
        
        # Test invalid n_components
        with pytest.raises(ValueError):
            fit_ica(raw, n_components=-5)

    @patch('autoclean.functions.ica.ica_processing.ICA')
    def test_fitting_failure(self, mock_ica_class, mock_raw):
        """Test handling of ICA fitting failures."""
        mock_ica_instance = MagicMock(spec=ICA)
        mock_ica_instance.fit.side_effect = Exception("Fitting failed")
        mock_ica_class.return_value = mock_ica_instance
        
        with pytest.raises(RuntimeError, match="Failed to fit ICA"):
            fit_ica(mock_raw)


class TestClassifyIcaComponents:
    """Test ICA component classification."""

    @patch('autoclean.functions.ica.ica_processing.mne_icalabel.label_components')
    def test_basic_functionality(self, mock_label_components, mock_raw, mock_ica):
        """Test basic component classification."""
        result = classify_ica_components(mock_raw, mock_ica)
        
        # Should call ICLabel
        mock_label_components.assert_called_once_with(
            mock_raw, mock_ica, method="iclabel", verbose=None
        )
        
        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)
        expected_columns = ["component", "ic_type", "confidence"]
        for col in expected_columns:
            assert col in result.columns

    def test_input_validation(self):
        """Test input validation for component classification."""
        # Test non-Raw object
        ica = MagicMock(spec=ICA)
        with pytest.raises(TypeError):
            classify_ica_components("not_raw", ica)
        
        # Test non-ICA object
        raw = MagicMock(spec=mne.io.BaseRaw)
        with pytest.raises(TypeError):
            classify_ica_components(raw, "not_ica")
        
        # Test unsupported method
        with pytest.raises(ValueError):
            classify_ica_components(raw, ica, method="unsupported")

    @patch('autoclean.functions.ica.ica_processing.mne_icalabel.label_components')
    def test_classification_failure(self, mock_label_components, mock_raw, mock_ica):
        """Test handling of classification failures."""
        mock_label_components.side_effect = Exception("Classification failed")
        
        with pytest.raises(RuntimeError, match="Failed to classify ICA components"):
            classify_ica_components(mock_raw, mock_ica)

    def test_dataframe_structure(self, mock_raw, mock_ica):
        """Test that returned DataFrame has correct structure."""
        # Mock the ICA labels structure to match real ICLabel output
        mock_ica.labels_ = {
            'brain': [0, 5],
            'eog': [1, 2], 
            'muscle': [3],
            'ecg': [],
            'line_noise': [4],
            'ch_noise': [],
            'other': [6, 7, 8, 9]
        }
        mock_ica.labels_scores_ = MagicMock()
        mock_ica.labels_scores_.max.return_value = np.array([0.8, 0.9, 0.7, 0.85, 0.95, 0.75, 0.6, 0.65, 0.7, 0.8])
        
        with patch('autoclean.functions.ica.ica_processing.mne_icalabel.label_components'):
            result = classify_ica_components(mock_raw, mock_ica)
            
            # Check DataFrame structure
            assert len(result) == mock_ica.n_components_
            assert "component" in result.columns
            assert "ic_type" in result.columns
            assert "confidence" in result.columns
            assert "annotator" in result.columns
            
            # Check that component types are correctly assigned
            assert result.loc[0, "ic_type"] == "brain"
            assert result.loc[1, "ic_type"] == "eog"
            assert result.loc[3, "ic_type"] == "muscle"


class TestApplyIcaRejection:
    """Test ICA component rejection."""

    def test_basic_functionality(self, mock_raw, mock_ica):
        """Test basic ICA component rejection."""
        components_to_reject = [0, 2, 5]
        
        result = apply_ica_rejection(mock_raw, mock_ica, components_to_reject)
        
        # Should copy ICA and set exclude
        mock_ica.copy.assert_called_once()
        
        # Should apply ICA
        mock_ica.copy.return_value.apply.assert_called_once()
        assert result == mock_ica.copy.return_value.apply.return_value

    def test_input_validation(self):
        """Test input validation for ICA rejection."""
        # Test non-Raw object
        ica = MagicMock(spec=ICA)
        with pytest.raises(TypeError):
            apply_ica_rejection("not_raw", ica, [0, 1])
        
        # Test non-ICA object
        raw = MagicMock(spec=mne.io.BaseRaw)
        with pytest.raises(TypeError):
            apply_ica_rejection(raw, "not_ica", [0, 1])

    def test_invalid_component_indices(self, mock_raw, mock_ica):
        """Test validation of component indices."""
        # Test negative index
        with pytest.raises(ValueError, match="Invalid component indices"):
            apply_ica_rejection(mock_raw, mock_ica, [-1, 0, 1])
        
        # Test index too high
        with pytest.raises(ValueError, match="Invalid component indices"):
            apply_ica_rejection(mock_raw, mock_ica, [0, 1, 25])  # mock_ica has 20 components

    def test_empty_component_list(self, mock_raw, mock_ica):
        """Test with empty component rejection list."""
        result = apply_ica_rejection(mock_raw, mock_ica, [])
        
        # Should still work with empty list
        mock_ica.copy.assert_called_once()
        mock_ica.copy.return_value.apply.assert_called_once()

    def test_copy_parameter(self, mock_raw, mock_ica):
        """Test copy parameter behavior."""
        components_to_reject = [0, 1]
        
        # Test with copy=True (default)
        apply_ica_rejection(mock_raw, mock_ica, components_to_reject, copy=True)
        mock_ica.copy.return_value.apply.assert_called_with(
            mock_raw, copy=True, verbose=None
        )
        
        # Test with copy=False
        apply_ica_rejection(mock_raw, mock_ica, components_to_reject, copy=False)
        mock_ica.copy.return_value.apply.assert_called_with(
            mock_raw, copy=False, verbose=None
        )

    def test_rejection_failure(self, mock_raw, mock_ica):
        """Test handling of ICA rejection failures."""
        mock_ica.copy.return_value.apply.side_effect = Exception("Rejection failed")
        
        with pytest.raises(RuntimeError, match="Failed to apply ICA rejection"):
            apply_ica_rejection(mock_raw, mock_ica, [0, 1])


class TestIntegration:
    """Integration tests for ICA functions."""

    @patch('autoclean.functions.ica.ica_processing.ICA')
    @patch('autoclean.functions.ica.ica_processing.mne_icalabel.label_components')
    def test_complete_ica_workflow(self, mock_label_components, mock_ica_class, mock_raw):
        """Test complete ICA workflow: fit -> classify -> reject."""
        # Mock ICA fitting
        mock_ica_instance = MagicMock(spec=ICA)
        mock_ica_instance.n_components_ = 10
        mock_ica_instance.labels_ = {
            'brain': [0, 3, 4, 6, 7, 9],
            'eog': [1, 8], 
            'muscle': [2],
            'ecg': [5],
            'line_noise': [],
            'ch_noise': [],
            'other': []
        }
        mock_ica_instance.labels_scores_ = MagicMock()
        mock_ica_instance.labels_scores_.max.return_value = np.array([0.9, 0.8, 0.8, 0.9, 0.9, 0.7, 0.9, 0.9, 0.8, 0.9])
        mock_ica_class.return_value = mock_ica_instance
        
        # Step 1: Fit ICA
        ica = fit_ica(mock_raw, n_components=10)
        assert ica == mock_ica_instance
        
        # Step 2: Classify components
        labels = classify_ica_components(mock_raw, ica)
        assert isinstance(labels, pd.DataFrame)
        assert len(labels) == 10
        
        # Step 3: Find artifact components
        artifacts = labels[
            (labels["ic_type"].isin(["eog", "muscle", "ecg"])) &
            (labels["confidence"] > 0.7)
        ]["component"].tolist()
        
        expected_artifacts = [1, 2, 5, 8]  # Based on mock labels: eog=[1,8], muscle=[2], ecg=[5]
        assert set(artifacts) == set(expected_artifacts)
        
        # Step 4: Apply rejection
        clean_raw = apply_ica_rejection(mock_raw, ica, artifacts)
        
        # Verify the complete workflow
        mock_ica_instance.copy.assert_called()
        mock_ica_instance.copy.return_value.apply.assert_called()

    def test_dataframe_helper_function(self):
        """Test the _icalabel_to_dataframe helper function."""
        from autoclean.functions.ica.ica_processing import _icalabel_to_dataframe
        
        # Create mock ICA with real ICLabel results structure
        mock_ica = MagicMock(spec=ICA)
        mock_ica.n_components_ = 3
        mock_ica.labels_ = {
            'brain': [0],
            'eog': [1],
            'muscle': [2],
            'ecg': [],
            'line_noise': [],
            'ch_noise': [],
            'other': []
        }
        mock_ica.labels_scores_ = MagicMock()
        mock_ica.labels_scores_.max.return_value = np.array([0.9, 0.8, 0.8])
        
        result = _icalabel_to_dataframe(mock_ica)
        
        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result["ic_type"]) == ["brain", "eog", "muscle"]
        assert list(result["confidence"]) == [0.9, 0.8, 0.8]
        
        # Check required columns
        assert "component" in result.columns
        assert "annotator" in result.columns
        assert "ic_type" in result.columns
        assert "confidence" in result.columns