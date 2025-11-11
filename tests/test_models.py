"""
Test suite for Battery RUL Prediction Models
Tests both Zenodo and NASA models for correct input/output behavior
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import model_inference_example
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_inference_example import BatteryRULPredictor


class TestZenodoModel:
    """Test cases for Zenodo model"""
    
    @pytest.fixture
    def predictor(self):
        """Initialize Zenodo predictor"""
        return BatteryRULPredictor(model_type='zenodo')
    
    @pytest.fixture
    def sample_data(self, predictor):
        """Create sample input data with correct features"""
        required_features = predictor.get_required_features()
        # Create realistic sample data (1 sample)
        data = {
            'protocol_id': [1],
            'discharge_capacity_ah_max': [2.42],
            'discharge_capacity_ah_mean': [0.85],
            'discharge_capacity_ah_min': [0.00003],
            'charge_capacity_ah_max': [2.39],
            'charge_capacity_ah_mean': [2.12],
            'charge_capacity_ah_min': [0.0033],
            'voltage_v_max': [4.20],
            'voltage_v_mean': [3.83],
            'voltage_v_min': [3.00],
            'voltage_v_std': [0.29],
            'current_a_mean': [-0.0034],
            'current_a_std': [0.67],
            'discharge_energy_wh_max': [9.16],
            'charge_energy_wh_max': [9.37],
            'aux_temperature_1_c_mean': [25.0],
            'aux_temperature_1_c_max': [25.44],
            'aux_temperature_1_c_min': [24.40],
            'aux_temperature_1_c_std': [0.21],
            'current_a_abs_mean': [0.56],
            'soh_percent': [100.0],
            'rolling_mean_discharge_capacity_ah_max': [2.42],
            'rolling_mean_charge_capacity_ah_max': [2.39],
            'rolling_mean_voltage_v_max': [4.20],
            'rolling_mean_current_a_mean': [-0.0034],
            'rolling_mean_discharge_energy_wh_max': [9.16],
            'rolling_mean_charge_energy_wh_max': [9.37],
            'rolling_mean_aux_temperature_1_c_mean': [25.0],
            'rolling_mean_current_a_abs_mean': [0.56],
            'rolling_mean_soh_percent': [100.0],
            'rolling_std_discharge_capacity_ah_max': [0.0],
            'rolling_std_charge_capacity_ah_max': [0.0],
            'rolling_std_voltage_v_max': [0.0],
            'rolling_std_current_a_mean': [0.0],
            'rolling_std_discharge_energy_wh_max': [0.0],
            'rolling_std_charge_energy_wh_max': [0.0],
            'rolling_std_aux_temperature_1_c_mean': [0.0],
            'rolling_std_current_a_abs_mean': [0.0],
            'rolling_std_soh_percent': [0.0]
        }
        return pd.DataFrame(data)
    
    def test_model_loads(self, predictor):
        """Test that Zenodo model loads successfully"""
        assert predictor.model is not None
        assert predictor.preprocessor is not None
        assert predictor.metadata is not None
        assert predictor.feature_info is not None
    
    def test_required_features_count(self, predictor):
        """Test that required features count is correct"""
        features = predictor.get_required_features()
        assert len(features) == 39  # Zenodo model has 39 features
    
    def test_predict_with_correct_input(self, predictor, sample_data):
        """Test prediction with correct input format"""
        predictions = predictor.predict(sample_data)
        
        # Check output shape
        assert predictions.shape == (1,)
        
        # Check output type
        assert isinstance(predictions, np.ndarray)
        
        # Check output is numeric
        assert np.isfinite(predictions).all()
        
        # Check output is positive (RUL should be positive)
        assert (predictions >= 0).all()
    
    def test_predict_multiple_samples(self, predictor, sample_data):
        """Test prediction with multiple samples"""
        # Create 5 samples
        multi_sample = pd.concat([sample_data] * 5, ignore_index=True)
        predictions = predictor.predict(multi_sample)
        
        # Check output shape
        assert predictions.shape == (5,)
        
        # Check all predictions are valid
        assert np.isfinite(predictions).all()
        assert (predictions >= 0).all()
    
    def test_predict_with_missing_features(self, predictor, sample_data):
        """Test that prediction fails with missing features"""
        # Remove a required feature
        incomplete_data = sample_data.drop(columns=['protocol_id'])
        
        with pytest.raises(ValueError, match="Missing required features"):
            predictor.predict(incomplete_data)
    
    def test_predict_with_uncertainty(self, predictor, sample_data):
        """Test prediction with uncertainty estimation"""
        result = predictor.predict_with_uncertainty(sample_data, return_std=True)
        
        # Check if model supports uncertainty (returns tuple)
        if isinstance(result, tuple):
            predictions, std = result
            
            # Check output shapes
            assert predictions.shape == (1,)
            assert std.shape == (1,)
            
            # Check uncertainty is non-negative
            assert (std >= 0).all()
        else:
            # XGBoost doesn't support uncertainty, verify it returns predictions
            predictions = result
            assert predictions.shape == (1,)
            assert isinstance(predictions, np.ndarray)
            assert np.isfinite(predictions).all()
    
    def test_feature_importance(self, predictor):
        """Test feature importance extraction"""
        importance_df = predictor.get_feature_importance(top_n=10)
        
        if importance_df is not None:
            # Check structure
            assert 'Feature' in importance_df.columns
            assert 'Importance' in importance_df.columns
            
            # Check number of features returned
            assert len(importance_df) <= 10
            
            # Check importance values are non-negative
            assert (importance_df['Importance'] >= 0).all()


class TestNASAModel:
    """Test cases for NASA model"""
    
    @pytest.fixture
    def predictor(self):
        """Initialize NASA predictor"""
        return BatteryRULPredictor(model_type='nasa')
    
    @pytest.fixture
    def sample_data(self, predictor):
        """Create sample input data with correct features"""
        required_features = predictor.get_required_features()
        # Create realistic sample data (1 sample)
        data = {
            'voltage_v_mean': [3.83],
            'current_a_mean': [-0.0034],
            'aux_temperature_1_c_mean': [25.0],
            'current_a_abs_mean': [0.56],
            'rolling_mean_current_a_mean': [-0.0034],
            'rolling_mean_aux_temperature_1_c_mean': [25.0],
            'rolling_mean_current_a_abs_mean': [0.56],
            'rolling_std_current_a_mean': [0.0],
            'rolling_std_aux_temperature_1_c_mean': [0.0],
            'rolling_std_current_a_abs_mean': [0.0]
        }
        return pd.DataFrame(data)
    
    def test_model_loads(self, predictor):
        """Test that NASA model loads successfully"""
        assert predictor.model is not None
        assert predictor.metadata is not None
        assert predictor.feature_info is not None
    
    def test_required_features_count(self, predictor):
        """Test that required features count is correct"""
        features = predictor.get_required_features()
        assert len(features) == 10  # NASA model has 10 common features
    
    def test_predict_with_correct_input(self, predictor, sample_data):
        """Test prediction with correct input format"""
        predictions = predictor.predict(sample_data)
        
        # Check output shape
        assert predictions.shape == (1,)
        
        # Check output type
        assert isinstance(predictions, np.ndarray)
        
        # Check output is numeric
        assert np.isfinite(predictions).all()
        
        # Check output is positive (RUL should be positive)
        assert (predictions >= 0).all()
    
    def test_predict_multiple_samples(self, predictor, sample_data):
        """Test prediction with multiple samples"""
        # Create 3 samples
        multi_sample = pd.concat([sample_data] * 3, ignore_index=True)
        predictions = predictor.predict(multi_sample)
        
        # Check output shape
        assert predictions.shape == (3,)
        
        # Check all predictions are valid
        assert np.isfinite(predictions).all()
        assert (predictions >= 0).all()
    
    def test_predict_with_missing_features(self, predictor, sample_data):
        """Test that prediction fails with missing features"""
        # Remove a required feature
        incomplete_data = sample_data.drop(columns=['voltage_v_mean'])
        
        with pytest.raises(ValueError, match="Missing required features"):
            predictor.predict(incomplete_data)
    
    def test_predict_with_uncertainty(self, predictor, sample_data):
        """Test prediction with uncertainty estimation"""
        result = predictor.predict_with_uncertainty(sample_data, return_std=True)
        
        # Check if model supports uncertainty (returns tuple)
        if isinstance(result, tuple):
            predictions, std = result
            
            # Check output shapes
            assert predictions.shape == (1,)
            assert std.shape == (1,)
            
            # Check uncertainty is non-negative
            assert (std >= 0).all()
        else:
            # Model doesn't support uncertainty, just returns predictions
            pytest.skip("Model does not support uncertainty estimation")
    
    def test_feature_importance(self, predictor):
        """Test feature importance extraction"""
        importance_df = predictor.get_feature_importance(top_n=5)
        
        if importance_df is not None:
            # Check structure
            assert 'Feature' in importance_df.columns
            assert 'Importance' in importance_df.columns
            
            # Check number of features returned
            assert len(importance_df) <= 5
            
            # Check importance values are non-negative
            assert (importance_df['Importance'] >= 0).all()


class TestModelComparison:
    """Test cases comparing both models"""
    
    def test_both_models_load(self):
        """Test that both models can be loaded"""
        zenodo_predictor = BatteryRULPredictor(model_type='zenodo')
        nasa_predictor = BatteryRULPredictor(model_type='nasa')
        
        assert zenodo_predictor.model is not None
        assert nasa_predictor.model is not None
    
    def test_models_have_different_features(self):
        """Test that models require different number of features"""
        zenodo_predictor = BatteryRULPredictor(model_type='zenodo')
        nasa_predictor = BatteryRULPredictor(model_type='nasa')
        
        zenodo_features = len(zenodo_predictor.get_required_features())
        nasa_features = len(nasa_predictor.get_required_features())
        
        # Zenodo has more features than NASA
        assert zenodo_features > nasa_features
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error"""
        with pytest.raises(ValueError, match="model_type must be"):
            BatteryRULPredictor(model_type='invalid')


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
