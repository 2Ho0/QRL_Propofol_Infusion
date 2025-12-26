"""
Unit Tests for Greco Interaction Model
=======================================

Tests the Greco response surface model for drug interaction.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.pharmacodynamics.interaction_model import (
    GrecoInteractionModel,
    MintoInteractionModel,
    AdditiveInteractionModel,
    InteractionType,
    DrugParameters
)


class TestGrecoInteractionModel:
    """Test suite for GrecoInteractionModel."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        model = GrecoInteractionModel()
        
        assert model.alpha == 1.2
        assert model.E0 == 100.0
        assert model.Emax == 100.0
        assert model.interaction_type == InteractionType.SYNERGISTIC
    
    def test_initialization_additive(self):
        """Test additive interaction (alpha=0)."""
        model = GrecoInteractionModel(alpha=0.0)
        
        assert model.alpha == 0.0
        assert model.interaction_type == InteractionType.ADDITIVE
    
    def test_initialization_antagonistic(self):
        """Test antagonistic interaction (alpha<0)."""
        model = GrecoInteractionModel(alpha=-0.5)
        
        assert model.alpha == -0.5
        assert model.interaction_type == InteractionType.ANTAGONISTIC
    
    def test_single_drug_ec50(self):
        """Test that EC50 gives BIS=50 for single drug."""
        model = GrecoInteractionModel()
        
        # Propofol at EC50
        bis_ppf = model.compute_bis(ce_propofol=3.4, ce_remifentanil=0.0)
        assert abs(bis_ppf - 50.0) < 1.0
        
        # Remifentanil at EC50
        bis_rftn = model.compute_bis(ce_propofol=0.0, ce_remifentanil=2.0)
        assert abs(bis_rftn - 50.0) < 1.0
    
    def test_awake_state(self):
        """Test that no drug gives BIS=100."""
        model = GrecoInteractionModel()
        
        bis = model.compute_bis(ce_propofol=0.0, ce_remifentanil=0.0)
        assert abs(bis - 100.0) < 0.1
    
    def test_synergy_benefit(self):
        """Test synergistic interaction gives lower BIS than additive."""
        model_synergistic = GrecoInteractionModel(alpha=1.2)
        model_additive = GrecoInteractionModel(alpha=0.0)
        
        ce_ppf = 2.0  # Below EC50
        ce_rftn = 1.0  # Below EC50
        
        bis_syn = model_synergistic.compute_bis(ce_ppf, ce_rftn)
        bis_add = model_additive.compute_bis(ce_ppf, ce_rftn)
        
        # Synergistic should give lower BIS (deeper anesthesia)
        assert bis_syn < bis_add
        
        # Difference should be noticeable
        assert (bis_add - bis_syn) > 3.0
    
    def test_deep_anesthesia(self):
        """Test high concentrations give low BIS."""
        model = GrecoInteractionModel()
        
        # High doses
        bis = model.compute_bis(ce_propofol=6.0, ce_remifentanil=4.0)
        
        # Should be deeply anesthetized
        assert bis < 30.0
    
    def test_compute_effect(self):
        """Test effect computation (0-1 scale)."""
        model = GrecoInteractionModel()
        
        # Awake
        effect_awake = model.compute_effect(0.0, 0.0)
        assert abs(effect_awake) < 0.01
        
        # At EC50s
        effect_ec50 = model.compute_effect(3.4, 2.0)
        assert 0.4 < effect_ec50 < 0.6
        
        # Deep anesthesia
        effect_deep = model.compute_effect(6.0, 4.0)
        assert effect_deep > 0.7
    
    def test_monotonicity(self):
        """Test that increasing concentration decreases BIS."""
        model = GrecoInteractionModel()
        
        bis_prev = 100.0
        for ce_ppf in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            bis = model.compute_bis(ce_ppf, 1.0)
            assert bis < bis_prev  # BIS should decrease
            bis_prev = bis
    
    def test_compute_isobol(self):
        """Test isobol computation."""
        model = GrecoInteractionModel()
        
        ce_ppf_array, ce_rftn_array = model.compute_isobol(
            target_bis=50.0,
            n_points=50
        )
        
        # Should have found some points
        assert len(ce_ppf_array) > 0
        assert len(ce_rftn_array) > 0
        
        # Verify points are on isobol
        for ce_ppf, ce_rftn in zip(ce_ppf_array, ce_rftn_array):
            bis = model.compute_bis(ce_ppf, ce_rftn)
            assert abs(bis - 50.0) < 1.0
    
    def test_get_info(self):
        """Test info dictionary."""
        model = GrecoInteractionModel(alpha=1.2)
        
        info = model.get_info()
        
        assert info['model'] == 'Greco Response Surface'
        assert info['interaction_type'] == 'synergistic'
        assert info['alpha'] == 1.2
        assert 'propofol' in info
        assert 'remifentanil' in info


class TestMintoInteractionModel:
    """Test suite for MintoInteractionModel."""
    
    def test_initialization(self):
        """Test initialization."""
        model = MintoInteractionModel()
        
        assert model.ppf_C50 == 3.4
        assert model.rftn_C50 == 2.0
        assert model.E0 == 100.0
    
    def test_awake_state(self):
        """Test no drug gives BIS=100."""
        model = MintoInteractionModel()
        
        bis = model.compute_bis(0.0, 0.0)
        assert abs(bis - 100.0) < 0.1
    
    def test_single_drug_ec50(self):
        """Test EC50 gives approximately BIS=50."""
        model = MintoInteractionModel()
        
        bis_ppf = model.compute_bis(3.4, 0.0)
        bis_rftn = model.compute_bis(0.0, 2.0)
        
        # Should be around 50 (not exact due to multiplicative model)
        assert 40 < bis_ppf < 60
        assert 40 < bis_rftn < 60


class TestAdditiveInteractionModel:
    """Test suite for AdditiveInteractionModel."""
    
    def test_initialization(self):
        """Test initialization."""
        model = AdditiveInteractionModel()
        
        assert model.ppf_C50 == 3.4
        assert model.rftn_C50 == 2.0
        assert model.E0 == 100.0
    
    def test_awake_state(self):
        """Test no drug gives BIS=100."""
        model = AdditiveInteractionModel()
        
        bis = model.compute_bis(0.0, 0.0)
        assert abs(bis - 100.0) < 0.1
    
    def test_additive_property(self):
        """Test that effects add linearly."""
        model = AdditiveInteractionModel()
        
        # Individual effects
        bis_ppf_only = model.compute_bis(2.0, 0.0)
        bis_rftn_only = model.compute_bis(0.0, 1.0)
        
        # Combined effect
        bis_combined = model.compute_bis(2.0, 1.0)
        
        # Combined should be approximately sum of individual effects
        # (converted back from BIS to effect)
        effect_ppf = (100 - bis_ppf_only) / 100
        effect_rftn = (100 - bis_rftn_only) / 100
        effect_combined = (100 - bis_combined) / 100
        
        # Should be close to additive
        assert abs(effect_combined - (effect_ppf + effect_rftn)) < 0.1


class TestDrugParameters:
    """Test DrugParameters dataclass."""
    
    def test_creation(self):
        """Test creating drug parameters."""
        params = DrugParameters(C50=3.4, gamma=1.47, E0=100.0, Emax=100.0)
        
        assert params.C50 == 3.4
        assert params.gamma == 1.47
        assert params.E0 == 100.0
        assert params.Emax == 100.0
    
    def test_defaults(self):
        """Test default values."""
        params = DrugParameters(C50=3.4, gamma=1.47)
        
        assert params.E0 == 100.0
        assert params.Emax == 100.0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
