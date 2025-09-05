"""
Main Compliance Filter Module

Combines hate speech and privacy violation detection into a unified compliance
scoring system with configurable thresholds and actions.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path

from .privacy_detector import PrivacyDetector, PrivacyViolation
from .hate_speech_detector import HateSpeechDetector, HateSpeechResult


class ComplianceAction(Enum):
    """Actions that can be taken based on compliance score."""
    ALLOW = "allow"
    WARN = "warn" 
    BLOCK = "block"


@dataclass
class ComplianceResult:
    """Result from compliance filtering."""
    action: ComplianceAction
    overall_score: float
    hate_speech_score: float
    privacy_score: float
    hate_speech_result: Optional[HateSpeechResult]
    privacy_violations: List[PrivacyViolation]
    processing_time: float
    reasoning: str
    timestamp: str


class ComplianceFilter:
    """
    Main compliance filter that combines multiple detection methods.
    
    Features:
    - Configurable scoring methods (weighted_average, max, product)
    - Adjustable thresholds for different risk levels
    - Comprehensive logging and audit trail
    - Performance monitoring
    - Feedback integration
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the compliance filter.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (takes precedence over config_path)
        """
        # Load configuration
        if config_dict:
            self.config = config_dict
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            # Use default config path
            default_config_path = Path(__file__).parent.parent / "config" / "default.yaml"
            self.config = self._load_config(str(default_config_path))
        
        # Extract configuration sections
        self.compliance_config = self.config.get('compliance', {})
        self.scoring_method = self.compliance_config.get('scoring_method', 'weighted_average')
        self.weights = self.compliance_config.get('weights', {'hate_speech': 0.6, 'privacy': 0.4})
        self.thresholds = self.compliance_config.get('thresholds', {
            'block_threshold': 0.7,
            'warn_threshold': 0.5,
            'pass_threshold': 0.2
        })
        
        # Initialize detectors
        self.privacy_detector = PrivacyDetector(self.config)
        
        try:
            self.hate_speech_detector = HateSpeechDetector(self.config)
            self.hate_speech_available = True
        except ImportError:
            logging.warning("Hate speech detection not available - transformers library not installed")
            self.hate_speech_detector = None
            self.hate_speech_available = False
        
        # Performance tracking
        self._total_checks = 0
        self._total_time = 0.0
        self._action_counts = {action.value: 0 for action in ComplianceAction}
        
        logging.info("ComplianceFilter initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_path}")
            # Return minimal default config
            return {
                'compliance': {
                    'scoring_method': 'weighted_average',
                    'weights': {'hate_speech': 0.6, 'privacy': 0.4},
                    'thresholds': {'block_threshold': 0.7, 'warn_threshold': 0.5, 'pass_threshold': 0.2}
                },
                'privacy': {'checks': {}},
                'hate_speech': {'threshold': 0.7}
            }
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise
    
    def check_compliance(self, text: str, user_context: Optional[Dict[str, Any]] = None) -> ComplianceResult:
        """
        Check compliance of input text against all filters.
        
        Args:
            text: Input text to check
            user_context: Optional user context for logging
            
        Returns:
            ComplianceResult with detailed analysis
        """
        start_time = time.time()
        
        # Run privacy detection
        privacy_violations = self.privacy_detector.detect_violations(text)
        privacy_score = self.privacy_detector.calculate_privacy_score(privacy_violations)
        
        # Run hate speech detection if available
        hate_speech_result = None
        hate_speech_score = 0.0
        
        if self.hate_speech_available:
            try:
                hate_speech_result = self.hate_speech_detector.detect_hate_speech(text)
                hate_speech_score = hate_speech_result.confidence if hate_speech_result.is_hate_speech else 0.0
            except Exception as e:
                logging.error(f"Hate speech detection failed: {e}")
                # Use conservative default
                hate_speech_score = 1.0
        
        # Calculate overall compliance score
        overall_score = self._calculate_overall_score(hate_speech_score, privacy_score)
        
        # Determine action
        action = self._determine_action(overall_score)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            overall_score, hate_speech_score, privacy_score, 
            hate_speech_result, privacy_violations
        )
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self._update_stats(action, processing_time)
        
        # Create result
        result = ComplianceResult(
            action=action,
            overall_score=overall_score,
            hate_speech_score=hate_speech_score,
            privacy_score=privacy_score,
            hate_speech_result=hate_speech_result,
            privacy_violations=privacy_violations,
            processing_time=processing_time,
            reasoning=reasoning,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Log the result
        self._log_compliance_check(text, result, user_context)
        
        return result
    
    def _calculate_overall_score(self, hate_speech_score: float, privacy_score: float) -> float:
        """
        Calculate overall compliance score using configured method.
        
        Args:
            hate_speech_score: Score from hate speech detection
            privacy_score: Score from privacy detection
            
        Returns:
            Overall compliance score between 0.0 and 1.0
        """
        if self.scoring_method == 'weighted_average':
            return (
                hate_speech_score * self.weights.get('hate_speech', 0.6) +
                privacy_score * self.weights.get('privacy', 0.4)
            )
        
        elif self.scoring_method == 'max':
            return max(hate_speech_score, privacy_score)
        
        elif self.scoring_method == 'product':
            # Using 1 - (1-a)(1-b) to combine probabilities
            return 1 - (1 - hate_speech_score) * (1 - privacy_score)
        
        else:
            logging.warning(f"Unknown scoring method: {self.scoring_method}, using weighted_average")
            return (
                hate_speech_score * self.weights.get('hate_speech', 0.6) +
                privacy_score * self.weights.get('privacy', 0.4)
            )
    
    def _determine_action(self, overall_score: float) -> ComplianceAction:
        """Determine action based on overall score and thresholds."""
        if overall_score >= self.thresholds.get('block_threshold', 0.7):
            return ComplianceAction.BLOCK
        elif overall_score >= self.thresholds.get('warn_threshold', 0.5):
            return ComplianceAction.WARN
        else:
            return ComplianceAction.ALLOW
    
    def _generate_reasoning(
        self, 
        overall_score: float, 
        hate_speech_score: float, 
        privacy_score: float,
        hate_speech_result: Optional[HateSpeechResult],
        privacy_violations: List[PrivacyViolation]
    ) -> str:
        """Generate human-readable reasoning for the compliance decision."""
        reasons = []
        
        if overall_score < self.thresholds.get('pass_threshold', 0.2):
            reasons.append("Content appears compliant with minimal risk.")
        
        if hate_speech_score > 0.3:
            if hate_speech_result:
                reasons.append(f"Potential hate speech detected (confidence: {hate_speech_score:.2f}, "
                             f"model: {hate_speech_result.model_used}).")
            else:
                reasons.append(f"Hate speech score elevated: {hate_speech_score:.2f}")
        
        if privacy_violations:
            violation_types = set(v.violation_type.value for v in privacy_violations)
            reasons.append(f"Privacy violations detected: {', '.join(violation_types)} "
                          f"(score: {privacy_score:.2f}).")
        
        if overall_score >= self.thresholds.get('block_threshold', 0.7):
            reasons.append("Content blocked due to high compliance risk.")
        elif overall_score >= self.thresholds.get('warn_threshold', 0.5):
            reasons.append("Content flagged for review due to moderate compliance risk.")
        
        return " ".join(reasons) if reasons else "No significant compliance issues detected."
    
    def _update_stats(self, action: ComplianceAction, processing_time: float):
        """Update performance statistics."""
        self._total_checks += 1
        self._total_time += processing_time
        self._action_counts[action.value] += 1
    
    def _log_compliance_check(
        self, 
        text: str, 
        result: ComplianceResult, 
        user_context: Optional[Dict[str, Any]]
    ):
        """Log compliance check details."""
        logging_config = self.config.get('logging', {})
        
        if not logging_config.get('audit_logs', True):
            return
        
        log_details = logging_config.get('log_details', {})
        
        log_entry = {
            'timestamp': result.timestamp,
            'action': result.action.value,
            'overall_score': result.overall_score,
            'hate_speech_score': result.hate_speech_score,
            'privacy_score': result.privacy_score,
            'processing_time': result.processing_time,
            'reasoning': result.reasoning
        }
        
        # Add optional details based on config
        if log_details.get('scores', True):
            if result.hate_speech_result:
                log_entry['hate_speech_details'] = {
                    'model': result.hate_speech_result.model_used,
                    'all_scores': result.hate_speech_result.all_scores
                }
        
        if log_details.get('violation_details', True):
            log_entry['privacy_violations'] = [
                {
                    'type': v.violation_type.value,
                    'confidence': v.confidence,
                    'description': v.description
                } for v in result.privacy_violations
            ]
        
        if log_details.get('user_context', True) and user_context:
            log_entry['user_context'] = user_context
        
        # Don't log prompt content by default for privacy
        if log_details.get('prompt_content', False):
            log_entry['prompt'] = text
        
        logging.info(f"Compliance check: {log_entry}")
    
    def batch_check_compliance(self, texts: List[str]) -> List[ComplianceResult]:
        """
        Check compliance for multiple texts.
        
        Args:
            texts: List of texts to check
            
        Returns:
            List of ComplianceResult objects
        """
        results = []
        for text in texts:
            result = self.check_compliance(text)
            results.append(result)
        return results
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Update compliance thresholds.
        
        Args:
            new_thresholds: Dictionary with new threshold values
        """
        for threshold_name, value in new_thresholds.items():
            if threshold_name in self.thresholds:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"Threshold {threshold_name} must be between 0.0 and 1.0")
                self.thresholds[threshold_name] = value
        
        logging.info(f"Updated thresholds: {new_thresholds}")
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update scoring weights.
        
        Args:
            new_weights: Dictionary with new weight values
        """
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            logging.warning(f"Weights sum to {total_weight}, not 1.0. Consider normalizing.")
        
        self.weights.update(new_weights)
        logging.info(f"Updated weights: {new_weights}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'total_checks': self._total_checks,
            'average_processing_time': self._total_time / max(self._total_checks, 1),
            'total_processing_time': self._total_time,
            'action_distribution': self._action_counts.copy()
        }
        
        # Add detector-specific stats
        if self.hate_speech_available:
            stats['hate_speech_detector'] = self.hate_speech_detector.get_performance_stats()
        
        return stats
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get current configuration summary."""
        return {
            'scoring_method': self.scoring_method,
            'weights': self.weights.copy(),
            'thresholds': self.thresholds.copy(),
            'hate_speech_available': self.hate_speech_available,
            'hate_speech_model': self.hate_speech_detector.model_name if self.hate_speech_available else None
        }
    
    def validate_prompt_safety(self, text: str) -> Tuple[bool, str]:
        """
        Quick safety validation for prompts.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_safe, reason)
        """
        result = self.check_compliance(text)
        
        is_safe = result.action == ComplianceAction.ALLOW
        reason = result.reasoning
        
        return is_safe, reason
    
    def cleanup(self):
        """Clean up resources."""
        if self.hate_speech_detector:
            self.hate_speech_detector.cleanup()
        
        logging.info("ComplianceFilter resources cleaned up")
