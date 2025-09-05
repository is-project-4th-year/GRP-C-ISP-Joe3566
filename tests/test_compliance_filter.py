"""
Basic tests for the LLM Compliance Filter functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.compliance_filter import ComplianceFilter, ComplianceAction
from src.privacy_detector import PrivacyDetector, ViolationType
from src.feedback_system import FeedbackSystem, FeedbackType


class TestPrivacyDetector:
    """Test privacy violation detection."""
    
    def test_email_detection(self):
        """Test email address detection."""
        detector = PrivacyDetector()
        
        text = "Please contact me at john.doe@example.com for more information."
        violations = detector.detect_violations(text)
        
        assert len(violations) > 0
        assert any(v.violation_type == ViolationType.EMAIL for v in violations)
        assert any("john.doe@example.com" in v.text_span for v in violations)
    
    def test_phone_detection(self):
        """Test phone number detection."""
        detector = PrivacyDetector()
        
        text = "Call me at (555) 123-4567 tomorrow."
        violations = detector.detect_violations(text)
        
        assert len(violations) > 0
        assert any(v.violation_type == ViolationType.PHONE for v in violations)
    
    def test_ssn_detection(self):
        """Test SSN detection."""
        detector = PrivacyDetector()
        
        text = "My SSN is 123-45-6789."
        violations = detector.detect_violations(text)
        
        assert len(violations) > 0
        assert any(v.violation_type == ViolationType.SSN for v in violations)
    
    def test_no_violations(self):
        """Test text with no privacy violations."""
        detector = PrivacyDetector()
        
        text = "This is a perfectly normal message about the weather."
        violations = detector.detect_violations(text)
        
        assert len(violations) == 0
    
    def test_privacy_score_calculation(self):
        """Test privacy score calculation."""
        detector = PrivacyDetector()
        
        # High-risk violation
        text_high_risk = "My SSN is 123-45-6789 and credit card is 4532-1234-5678-9012."
        violations_high = detector.detect_violations(text_high_risk)
        score_high = detector.calculate_privacy_score(violations_high)
        
        # Low-risk violation
        text_low_risk = "Contact me at info@company.com"
        violations_low = detector.detect_violations(text_low_risk)
        score_low = detector.calculate_privacy_score(violations_low)
        
        assert score_high > score_low
        assert 0.0 <= score_high <= 1.0
        assert 0.0 <= score_low <= 1.0


class TestComplianceFilter:
    """Test main compliance filter functionality."""
    
    def test_initialization(self):
        """Test filter initialization."""
        filter = ComplianceFilter()
        
        assert filter is not None
        assert filter.privacy_detector is not None
        # Note: hate_speech_detector might be None if transformers not installed
    
    def test_safe_prompt(self):
        """Test filtering of safe prompt."""
        filter = ComplianceFilter()
        
        result = filter.check_compliance("What is the capital of France?")
        
        assert result.action == ComplianceAction.ALLOW
        assert result.overall_score < 0.3  # Should be low risk
        assert len(result.privacy_violations) == 0
    
    def test_privacy_violation_prompt(self):
        """Test filtering of prompt with privacy violations."""
        filter = ComplianceFilter()
        
        result = filter.check_compliance("My email is test@example.com and SSN is 123-45-6789")
        
        assert result.action in [ComplianceAction.WARN, ComplianceAction.BLOCK]
        assert result.privacy_score > 0.0
        assert len(result.privacy_violations) > 0
    
    def test_threshold_adjustment(self):
        """Test threshold adjustment functionality."""
        filter = ComplianceFilter()
        
        # Test with default thresholds
        result1 = filter.check_compliance("Contact me at user@example.com")
        
        # Make thresholds more permissive
        filter.update_thresholds({
            'block_threshold': 0.9,
            'warn_threshold': 0.7
        })
        
        # Test with adjusted thresholds
        result2 = filter.check_compliance("Contact me at user@example.com")
        
        # Second result should be more permissive
        assert result2.action.value <= result1.action.value or result1.action == ComplianceAction.ALLOW
    
    def test_weight_adjustment(self):
        """Test weight adjustment functionality."""
        filter = ComplianceFilter()
        
        # Adjust weights to prioritize privacy over hate speech
        filter.update_weights({
            'privacy': 0.8,
            'hate_speech': 0.2
        })
        
        # Test that weights are updated
        assert filter.weights['privacy'] == 0.8
        assert filter.weights['hate_speech'] == 0.2
    
    def test_batch_processing(self):
        """Test batch processing of multiple prompts."""
        filter = ComplianceFilter()
        
        prompts = [
            "What's the weather today?",
            "My email is test@example.com",
            "Tell me a joke"
        ]
        
        results = filter.batch_check_compliance(prompts)
        
        assert len(results) == len(prompts)
        assert all(hasattr(result, 'action') for result in results)
        assert all(hasattr(result, 'overall_score') for result in results)
    
    def test_performance_stats(self):
        """Test performance statistics collection."""
        filter = ComplianceFilter()
        
        # Process some requests
        filter.check_compliance("Test prompt 1")
        filter.check_compliance("Test prompt 2")
        
        stats = filter.get_performance_stats()
        
        assert stats['total_checks'] >= 2
        assert 'average_processing_time' in stats
        assert 'action_distribution' in stats


class TestFeedbackSystem:
    """Test feedback system functionality."""
    
    def test_initialization(self):
        """Test feedback system initialization."""
        feedback_system = FeedbackSystem()
        
        assert feedback_system is not None
        assert feedback_system.enable_feedback is True
    
    def test_submit_feedback(self):
        """Test feedback submission."""
        filter = ComplianceFilter()
        feedback_system = FeedbackSystem()
        
        # Create a compliance result
        result = filter.check_compliance("Test prompt")
        
        # Submit feedback
        feedback_id = feedback_system.submit_feedback(
            feedback_type=FeedbackType.CORRECT_POSITIVE,
            original_text="Test prompt",
            compliance_result=result,
            user_assessment="This was handled correctly",
            confidence=0.9
        )
        
        assert feedback_id != ""
        assert len(feedback_system.feedback_entries) > 0
    
    def test_feedback_statistics(self):
        """Test feedback statistics."""
        filter = ComplianceFilter()
        feedback_system = FeedbackSystem()
        
        # Submit some feedback
        result = filter.check_compliance("Test prompt")
        feedback_system.submit_feedback(
            FeedbackType.FALSE_POSITIVE,
            "Test prompt",
            result,
            "Incorrectly flagged",
            0.8
        )
        
        stats = feedback_system.get_statistics()
        
        assert stats['total_entries'] > 0
        assert 'by_type' in stats
        assert stats['by_type']['false_positive'] > 0
    
    def test_should_request_feedback(self):
        """Test feedback request logic."""
        filter = ComplianceFilter()
        feedback_system = FeedbackSystem(filter.config)
        
        # Test with borderline case
        result = filter.check_compliance("This might be questionable content")
        should_request = feedback_system.should_request_feedback(result)
        
        assert isinstance(should_request, bool)


class TestIntegration:
    """Test integration scenarios."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from prompt to decision."""
        filter = ComplianceFilter()
        feedback_system = FeedbackSystem(filter.config)
        
        # Test safe prompt
        safe_prompt = "What is machine learning?"
        result = filter.check_compliance(safe_prompt)
        
        assert result.action == ComplianceAction.ALLOW
        assert result.processing_time > 0
        assert result.timestamp is not None
        
        # Check if feedback is needed
        needs_feedback = feedback_system.should_request_feedback(result)
        
        # For a safe prompt, probably no feedback needed
        # (but this depends on threshold settings)
        assert isinstance(needs_feedback, bool)
    
    def test_configuration_loading(self):
        """Test configuration loading and validation."""
        # Test with minimal config
        minimal_config = {
            'compliance': {
                'thresholds': {
                    'block_threshold': 0.8,
                    'warn_threshold': 0.6
                }
            }
        }
        
        filter = ComplianceFilter(config_dict=minimal_config)
        
        assert filter.thresholds['block_threshold'] == 0.8
        assert filter.thresholds['warn_threshold'] == 0.6


@pytest.fixture
def sample_compliance_result():
    """Fixture providing a sample compliance result."""
    filter = ComplianceFilter()
    return filter.check_compliance("Sample text for testing")


def test_compliance_result_structure(sample_compliance_result):
    """Test that compliance result has expected structure."""
    result = sample_compliance_result
    
    assert hasattr(result, 'action')
    assert hasattr(result, 'overall_score')
    assert hasattr(result, 'hate_speech_score')
    assert hasattr(result, 'privacy_score')
    assert hasattr(result, 'privacy_violations')
    assert hasattr(result, 'processing_time')
    assert hasattr(result, 'reasoning')
    assert hasattr(result, 'timestamp')
    
    assert isinstance(result.action, ComplianceAction)
    assert 0.0 <= result.overall_score <= 1.0
    assert 0.0 <= result.hate_speech_score <= 1.0
    assert 0.0 <= result.privacy_score <= 1.0
    assert isinstance(result.privacy_violations, list)
    assert result.processing_time >= 0
    assert isinstance(result.reasoning, str)
    assert isinstance(result.timestamp, str)


if __name__ == "__main__":
    # Run basic tests if executed directly
    test_detector = TestPrivacyDetector()
    test_detector.test_email_detection()
    test_detector.test_phone_detection()
    test_detector.test_no_violations()
    
    test_filter = TestComplianceFilter()
    test_filter.test_initialization()
    test_filter.test_safe_prompt()
    test_filter.test_privacy_violation_prompt()
    
    print("Basic tests completed successfully!")
