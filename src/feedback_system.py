"""
Feedback System Module

Handles feedback for compliance violations and provides mechanisms
to improve filter accuracy over time.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .compliance_filter import ComplianceResult, ComplianceAction


class FeedbackType(Enum):
    """Types of feedback that can be provided."""
    FALSE_POSITIVE = "false_positive"  # Filter incorrectly flagged safe content
    FALSE_NEGATIVE = "false_negative"  # Filter missed harmful content
    CORRECT_POSITIVE = "correct_positive"  # Filter correctly flagged harmful content
    CORRECT_NEGATIVE = "correct_negative"  # Filter correctly allowed safe content
    THRESHOLD_SUGGESTION = "threshold_suggestion"  # User suggests threshold adjustment
    MODEL_IMPROVEMENT = "model_improvement"  # General model improvement suggestion


@dataclass
class FeedbackEntry:
    """Represents a single feedback entry."""
    feedback_id: str
    feedback_type: FeedbackType
    original_text: str
    compliance_result: ComplianceResult
    user_assessment: str
    suggested_action: Optional[ComplianceAction]
    confidence: float  # User's confidence in their assessment
    context: Dict[str, Any]
    timestamp: str
    user_id: Optional[str] = None
    processed: bool = False
    notes: Optional[str] = None


class FeedbackSystem:
    """
    System for collecting and processing feedback to improve compliance filtering.
    
    Features:
    - Collect various types of feedback
    - Store feedback in structured format
    - Analyze feedback patterns
    - Generate improvement suggestions
    - Track feedback processing status
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the feedback system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feedback_config = self.config.get('feedback', {})
        
        # Configuration
        self.enable_feedback = self.feedback_config.get('enable_feedback', True)
        self.feedback_threshold = self.feedback_config.get('feedback_threshold', 0.1)
        self.store_feedback = self.feedback_config.get('store_feedback', True)
        self.feedback_file = self.feedback_config.get('feedback_file', './logs/feedback.jsonl')
        
        # In-memory feedback storage
        self.feedback_entries: List[FeedbackEntry] = []
        
        # Statistics
        self.feedback_stats = {
            'total_entries': 0,
            'by_type': {ftype.value: 0 for ftype in FeedbackType},
            'processed_count': 0,
            'average_confidence': 0.0
        }
        
        # Load existing feedback if file exists
        if self.store_feedback:
            self._load_feedback_from_file()
        
        logging.info("FeedbackSystem initialized")
    
    def submit_feedback(
        self,
        feedback_type: FeedbackType,
        original_text: str,
        compliance_result: ComplianceResult,
        user_assessment: str,
        confidence: float = 0.8,
        suggested_action: Optional[ComplianceAction] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Submit feedback about a compliance decision.
        
        Args:
            feedback_type: Type of feedback
            original_text: The original text that was checked
            compliance_result: The original compliance result
            user_assessment: User's assessment of the content
            confidence: User's confidence in their assessment (0.0-1.0)
            suggested_action: What action should have been taken
            user_id: Optional user identifier
            context: Additional context information
            notes: Optional notes from the user
            
        Returns:
            Feedback ID for tracking
        """
        if not self.enable_feedback:
            logging.warning("Feedback submission disabled")
            return ""
        
        feedback_id = f"fb_{int(time.time())}_{len(self.feedback_entries)}"
        
        feedback_entry = FeedbackEntry(
            feedback_id=feedback_id,
            feedback_type=feedback_type,
            original_text=original_text,
            compliance_result=compliance_result,
            user_assessment=user_assessment,
            suggested_action=suggested_action,
            confidence=confidence,
            context=context or {},
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            user_id=user_id,
            notes=notes
        )
        
        self.feedback_entries.append(feedback_entry)
        self._update_stats(feedback_entry)
        
        # Store to file if enabled
        if self.store_feedback:
            self._save_feedback_to_file(feedback_entry)
        
        logging.info(f"Feedback submitted: {feedback_id} ({feedback_type.value})")
        return feedback_id
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """
        Analyze feedback patterns to identify improvement opportunities.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.feedback_entries:
            return {"message": "No feedback data available for analysis"}
        
        analysis = {
            'summary': self.feedback_stats.copy(),
            'false_positive_rate': 0.0,
            'false_negative_rate': 0.0,
            'threshold_suggestions': [],
            'model_performance': {},
            'common_issues': []
        }
        
        total_judgments = len([f for f in self.feedback_entries 
                              if f.feedback_type in [FeedbackType.FALSE_POSITIVE, 
                                                   FeedbackType.FALSE_NEGATIVE,
                                                   FeedbackType.CORRECT_POSITIVE,
                                                   FeedbackType.CORRECT_NEGATIVE]])
        
        if total_judgments > 0:
            false_positives = len([f for f in self.feedback_entries 
                                 if f.feedback_type == FeedbackType.FALSE_POSITIVE])\n            false_negatives = len([f for f in self.feedback_entries \n                                 if f.feedback_type == FeedbackType.FALSE_NEGATIVE])\n            \n            analysis['false_positive_rate'] = false_positives / total_judgments\n            analysis['false_negative_rate'] = false_negatives / total_judgments\n        \n        # Analyze threshold suggestions\n        threshold_feedback = [f for f in self.feedback_entries \n                            if f.feedback_type == FeedbackType.THRESHOLD_SUGGESTION]\n        \n        if threshold_feedback:\n            # Group by suggested thresholds and find common suggestions\n            threshold_suggestions = {}\n            for feedback in threshold_feedback:\n                suggested = feedback.context.get('suggested_threshold', 'unknown')\n                if suggested not in threshold_suggestions:\n                    threshold_suggestions[suggested] = []\n                threshold_suggestions[suggested].append(feedback)\n            \n            analysis['threshold_suggestions'] = [\n                {\n                    'threshold': threshold,\n                    'count': len(entries),\n                    'average_confidence': sum(e.confidence for e in entries) / len(entries)\n                }\n                for threshold, entries in threshold_suggestions.items()\n            ]\n        \n        # Identify common issues\n        common_issues = self._identify_common_issues()\n        analysis['common_issues'] = common_issues\n        \n        return analysis\n    \n    def _identify_common_issues(self) -> List[Dict[str, Any]]:\n        \"\"\"Identify common issues from feedback.\"\"\"\n        issues = []\n        \n        # Analyze false positives by violation type\n        false_positives = [f for f in self.feedback_entries \n                         if f.feedback_type == FeedbackType.FALSE_POSITIVE]\n        \n        if false_positives:\n            privacy_fps = [f for f in false_positives \n                          if f.compliance_result.privacy_violations]\n            hate_speech_fps = [f for f in false_positives \n                             if f.compliance_result.hate_speech_result and \n                             f.compliance_result.hate_speech_result.is_hate_speech]\n            \n            if len(privacy_fps) > len(false_positives) * 0.3:  # More than 30% of FPs\n                issues.append({\n                    'type': 'privacy_detection_too_sensitive',\n                    'count': len(privacy_fps),\n                    'description': 'Privacy detection may be too sensitive',\n                    'suggestion': 'Consider adjusting privacy detection thresholds'\n                })\n            \n            if len(hate_speech_fps) > len(false_positives) * 0.3:\n                issues.append({\n                    'type': 'hate_speech_detection_too_sensitive',\n                    'count': len(hate_speech_fps),\n                    'description': 'Hate speech detection may be too sensitive',\n                    'suggestion': 'Consider adjusting hate speech model threshold'\n                })\n        \n        return issues\n    \n    def generate_improvement_suggestions(self) -> List[Dict[str, Any]]:\n        \"\"\"Generate specific suggestions for improving the filter.\"\"\"\n        suggestions = []\n        analysis = self.analyze_feedback_patterns()\n        \n        # Threshold adjustment suggestions\n        if analysis['false_positive_rate'] > 0.2:  # High false positive rate\n            suggestions.append({\n                'type': 'threshold_adjustment',\n                'priority': 'high',\n                'description': 'High false positive rate detected',\n                'action': 'Consider increasing block/warn thresholds',\n                'current_fp_rate': analysis['false_positive_rate']\n            })\n        \n        if analysis['false_negative_rate'] > 0.1:  # High false negative rate\n            suggestions.append({\n                'type': 'threshold_adjustment',\n                'priority': 'critical',\n                'description': 'High false negative rate detected',\n                'action': 'Consider decreasing block/warn thresholds',\n                'current_fn_rate': analysis['false_negative_rate']\n            })\n        \n        # Model-specific suggestions\n        for issue in analysis.get('common_issues', []):\n            suggestions.append({\n                'type': 'model_tuning',\n                'priority': 'medium',\n                'description': issue['description'],\n                'action': issue['suggestion'],\n                'affected_count': issue['count']\n            })\n        \n        return suggestions\n    \n    def get_feedback_by_type(self, feedback_type: FeedbackType) -> List[FeedbackEntry]:\n        \"\"\"Get all feedback entries of a specific type.\"\"\"\n        return [f for f in self.feedback_entries if f.feedback_type == feedback_type]\n    \n    def mark_feedback_processed(self, feedback_ids: List[str]):\n        \"\"\"Mark feedback entries as processed.\"\"\"\n        processed_count = 0\n        for feedback in self.feedback_entries:\n            if feedback.feedback_id in feedback_ids and not feedback.processed:\n                feedback.processed = True\n                processed_count += 1\n        \n        self.feedback_stats['processed_count'] += processed_count\n        logging.info(f\"Marked {processed_count} feedback entries as processed\")\n    \n    def export_feedback_for_training(self, output_file: str, include_processed: bool = False):\n        \"\"\"Export feedback data in format suitable for model training.\"\"\"\n        training_data = []\n        \n        for feedback in self.feedback_entries:\n            if not include_processed and feedback.processed:\n                continue\n            \n            # Convert to training format\n            training_entry = {\n                'text': feedback.original_text,\n                'original_prediction': {\n                    'action': feedback.compliance_result.action.value,\n                    'overall_score': feedback.compliance_result.overall_score,\n                    'hate_speech_score': feedback.compliance_result.hate_speech_score,\n                    'privacy_score': feedback.compliance_result.privacy_score\n                },\n                'user_label': feedback.user_assessment,\n                'feedback_type': feedback.feedback_type.value,\n                'confidence': feedback.confidence,\n                'suggested_action': feedback.suggested_action.value if feedback.suggested_action else None\n            }\n            training_data.append(training_entry)\n        \n        with open(output_file, 'w', encoding='utf-8') as f:\n            for entry in training_data:\n                f.write(json.dumps(entry) + '\\n')\n        \n        logging.info(f\"Exported {len(training_data)} feedback entries to {output_file}\")\n    \n    def _update_stats(self, feedback_entry: FeedbackEntry):\n        \"\"\"Update internal statistics.\"\"\"\n        self.feedback_stats['total_entries'] += 1\n        self.feedback_stats['by_type'][feedback_entry.feedback_type.value] += 1\n        \n        # Update average confidence\n        total_confidence = (self.feedback_stats['average_confidence'] * \n                          (self.feedback_stats['total_entries'] - 1) + \n                          feedback_entry.confidence)\n        self.feedback_stats['average_confidence'] = total_confidence / self.feedback_stats['total_entries']\n    \n    def _save_feedback_to_file(self, feedback_entry: FeedbackEntry):\n        \"\"\"Save feedback entry to file.\"\"\"\n        try:\n            # Create directory if it doesn't exist\n            Path(self.feedback_file).parent.mkdir(parents=True, exist_ok=True)\n            \n            # Convert to JSON-serializable format\n            entry_dict = asdict(feedback_entry)\n            entry_dict['compliance_result'] = asdict(feedback_entry.compliance_result)\n            \n            # Handle enum serialization\n            entry_dict['feedback_type'] = feedback_entry.feedback_type.value\n            entry_dict['compliance_result']['action'] = feedback_entry.compliance_result.action.value\n            if feedback_entry.suggested_action:\n                entry_dict['suggested_action'] = feedback_entry.suggested_action.value\n            \n            with open(self.feedback_file, 'a', encoding='utf-8') as f:\n                f.write(json.dumps(entry_dict) + '\\n')\n                \n        except Exception as e:\n            logging.error(f\"Failed to save feedback to file: {e}\")\n    \n    def _load_feedback_from_file(self):\n        \"\"\"Load existing feedback from file.\"\"\"\n        if not Path(self.feedback_file).exists():\n            return\n        \n        try:\n            with open(self.feedback_file, 'r', encoding='utf-8') as f:\n                for line_num, line in enumerate(f, 1):\n                    line = line.strip()\n                    if not line:\n                        continue\n                    \n                    try:\n                        entry_dict = json.loads(line)\n                        # Note: This is a simplified loading - full deserialization \n                        # would require more complex handling of nested objects\n                        self.feedback_stats['total_entries'] += 1\n                        feedback_type = entry_dict.get('feedback_type', 'unknown')\n                        if feedback_type in self.feedback_stats['by_type']:\n                            self.feedback_stats['by_type'][feedback_type] += 1\n                            \n                    except json.JSONDecodeError as e:\n                        logging.warning(f\"Failed to parse feedback line {line_num}: {e}\")\n                        \n            logging.info(f\"Loaded feedback statistics from {self.feedback_file}\")\n            \n        except Exception as e:\n            logging.error(f\"Failed to load feedback from file: {e}\")\n    \n    def get_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get current feedback statistics.\"\"\"\n        return self.feedback_stats.copy()\n    \n    def should_request_feedback(self, compliance_result: ComplianceResult) -> bool:\n        \"\"\"\n        Determine if feedback should be requested based on the compliance result.\n        \n        Args:\n            compliance_result: The compliance check result\n            \n        Returns:\n            True if feedback should be requested\n        \"\"\"\n        if not self.enable_feedback:\n            return False\n        \n        # Request feedback for borderline cases\n        threshold = self.feedback_threshold\n        score = compliance_result.overall_score\n        \n        # Cases where we should ask for feedback:\n        # 1. Score is close to thresholds\n        # 2. High uncertainty (score around 0.5)\n        # 3. Mixed signals (high hate speech but low privacy, or vice versa)\n        \n        if abs(score - 0.5) < threshold:  # High uncertainty\n            return True\n        \n        if abs(score - self.config.get('compliance', {}).get('thresholds', {}).get('block_threshold', 0.7)) < threshold:\n            return True\n            \n        if abs(score - self.config.get('compliance', {}).get('thresholds', {}).get('warn_threshold', 0.5)) < threshold:\n            return True\n        \n        # Mixed signals\n        hate_score = compliance_result.hate_speech_score\n        privacy_score = compliance_result.privacy_score\n        if abs(hate_score - privacy_score) > 0.4:  # Large difference\n            return True\n        \n        return False
