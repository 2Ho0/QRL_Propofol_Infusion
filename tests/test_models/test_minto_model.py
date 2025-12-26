"""
Unit Tests for Minto Pharmacokinetic Model
===========================================

Tests the Minto 3-compartment model for remifentanil pharmacokinetics.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.pharmacokinetics.minto_model import MintoModel, MintoParameters
from src.utils.exceptions import PKPDModelError


class TestMintoModel:
    """Test suite for MintoModel."""
    
    def test_initialization(self, sample_patient_params):
        """Test model initialization with valid parameters."""
        model = MintoModel(**sample_patient_params)
        
        assert model.age == 45
        assert model.weight == 70
        assert model.height == 170
        assert model.gender == 'M'
        
        # Check parameters are positive
        assert model.V1 > 0
        assert model.V2 > 0
        assert model.V3 > 0
        assert model.k10 > 0
        assert model.ke0 > 0
    
    def test_invalid_age(self):
        """Test that invalid age raises error."""
        with pytest.raises(PKPDModelError):
            MintoModel(age=150, weight=70, height=170, gender='M')
        
        with pytest.raises(PKPDModelError):
            MintoModel(age=-5, weight=70, height=170, gender='M')
    
    def test_invalid_weight(self):
        """Test that invalid weight raises error."""
        with pytest.raises(PKPDModelError):
            MintoModel(age=45, weight=250, height=170, gender='M')
        
        with pytest.raises(PKPDModelError):
            MintoModel(age=45, weight=20, height=170, gender='M')
    
    def test_invalid_height(self):
        """Test that invalid height raises error."""
        with pytest.raises(PKPDModelError):
            MintoModel(age=45, weight=70, height=300, gender='M')
    
    def test_initial_state(self, sample_patient_params):
        """Test initial state is zero."""
        model = MintoModel(**sample_patient_params)
        
        assert model.A1 == 0.0
        assert model.A2 == 0.0
        assert model.A3 == 0.0
        assert model.Ae == 0.0
        assert model.Cp == 0.0
        assert model.Ce == 0.0
    
    def test_set_infusion_rate(self, sample_patient_params):
        """Test setting infusion rate."""
        model = MintoModel(**sample_patient_params)
        
        rate = 0.5  # μg/kg/min
        model.set_infusion_rate(rate)
        
        # Infusion rate should be converted to μg/min
        expected_rate = rate * model.weight
        assert abs(model.infusion_rate - expected_rate) < 1e-6
    
    def test_step_increases_concentration(self, sample_patient_params):
        """Test that stepping with infusion increases concentration."""
        model = MintoModel(**sample_patient_params)
        model.set_infusion_rate(0.5)
        
        # Initial concentration
        ce_initial = model.Ce
        
        # Step forward
        model.step(dt=5.0)
        
        # Concentration should increase
        assert model.Ce > ce_initial
        assert model.Cp > 0
    
    def test_steady_state(self, sample_patient_params):
        """Test model reaches steady state."""
        model = MintoModel(**sample_patient_params)
        model.set_infusion_rate(0.5)
        
        # Simulate 60 minutes
        for _ in range(720):  # 60 min * 12 steps/min (dt=5s)
            model.step(dt=5.0)
        
        # Should approach steady state
        ce_ss = model.Ce
        assert 1.0 < ce_ss < 5.0  # ng/mL (reasonable range)
        
        # Plasma and effect-site should be close at steady state
        assert abs(model.Ce - model.Cp) < 0.5
    
    def test_recovery(self, sample_patient_params):
        """Test concentration decreases after stopping infusion."""
        model = MintoModel(**sample_patient_params)
        model.set_infusion_rate(0.5)
        
        # Reach steady state
        for _ in range(360):
            model.step(dt=5.0)
        
        ce_before = model.Ce
        
        # Stop infusion
        model.set_infusion_rate(0.0)
        
        # Wait 5 minutes
        for _ in range(60):
            model.step(dt=5.0)
        
        # Concentration should decrease
        assert model.Ce < ce_before
        assert model.Cp < ce_before
    
    def test_get_effect(self, sample_patient_params):
        """Test pharmacodynamic effect calculation."""
        model = MintoModel(**sample_patient_params)
        model.set_infusion_rate(0.5)
        
        # Initial effect should be near zero
        effect_initial = model.get_effect()
        assert 0 <= effect_initial < 0.1
        
        # After induction
        for _ in range(120):
            model.step(dt=5.0)
        
        effect_induced = model.get_effect()
        assert 0.5 < effect_induced < 1.0
    
    def test_state_management(self, sample_patient_params):
        """Test getting and setting state."""
        model = MintoModel(**sample_patient_params)
        model.set_infusion_rate(0.5)
        
        # Advance to some state
        for _ in range(60):
            model.step(dt=5.0)
        
        # Get state
        state = model.get_state()
        
        # Create new model
        model2 = MintoModel(**sample_patient_params)
        
        # Set state
        model2.set_state(state)
        
        # States should match
        assert abs(model.A1 - model2.A1) < 1e-6
        assert abs(model.A2 - model2.A2) < 1e-6
        assert abs(model.A3 - model2.A3) < 1e-6
        assert abs(model.Ae - model2.Ae) < 1e-6
        assert abs(model.Ce - model2.Ce) < 1e-6
    
    def test_get_info(self, sample_patient_params):
        """Test info dictionary."""
        model = MintoModel(**sample_patient_params)
        model.set_infusion_rate(0.5)
        
        for _ in range(60):
            model.step(dt=5.0)
        
        info = model.get_info()
        
        # Check required keys
        assert 'Cp' in info
        assert 'Ce' in info
        assert 'infusion_rate' in info
        assert 'V1' in info
        assert 'k10' in info
        
        # Check values
        assert info['Cp'] == model.Cp
        assert info['Ce'] == model.Ce
    
    def test_gender_difference(self):
        """Test that male and female have different LBM."""
        model_male = MintoModel(age=45, weight=70, height=170, gender='M')
        model_female = MintoModel(age=45, weight=70, height=170, gender='F')
        
        # LBM should be different
        assert model_male.lbm != model_female.lbm
        
        # Female typically has lower LBM
        assert model_female.lbm < model_male.lbm
    
    def test_age_effect(self):
        """Test that age affects parameters."""
        model_young = MintoModel(age=25, weight=70, height=170, gender='M')
        model_old = MintoModel(age=75, weight=70, height=170, gender='M')
        
        # Clearance typically decreases with age
        assert model_old.Cl1 < model_young.Cl1


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
