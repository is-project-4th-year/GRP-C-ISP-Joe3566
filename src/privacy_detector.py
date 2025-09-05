"""
Privacy Violation Detection Module

Detects various types of privacy violations including PII, sensitive data patterns,
and other privacy-sensitive content in text prompts.
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available. Some privacy detection features will be limited.")


class ViolationType(Enum):
    """Types of privacy violations that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    MEDICAL_INFO = "medical_info"
    FINANCIAL_INFO = "financial_info"
    API_KEY = "api_key"
    TOKEN = "token"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"


@dataclass
class PrivacyViolation:
    """Represents a detected privacy violation."""
    violation_type: ViolationType
    confidence: float
    text_span: str
    start_pos: int
    end_pos: int
    description: str


class PrivacyDetector:
    """
    Detects privacy violations in text using pattern matching and NLP.
    
    Features:
    - Comprehensive regex patterns for various PII types
    - Named Entity Recognition (if spaCy is available)
    - Configurable sensitivity levels
    - Custom pattern support
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the privacy detector.
        
        Args:
            config: Configuration dictionary with privacy detection settings
        """
        self.config = config or {}
        self.privacy_config = self.config.get('privacy', {})
        self.enabled_checks = self.privacy_config.get('checks', {})
        self.risk_levels = self.privacy_config.get('risk_levels', {})
        self.custom_patterns = self.privacy_config.get('custom_patterns', {})
        
        # Load spaCy model if available
        self.nlp = None
        if SPACY_AVAILABLE and self.enabled_checks.get('pii_detection', True):
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logging.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency."""
        self.patterns = {}
        
        # Email patterns
        if self.enabled_checks.get('email_detection', True):
            self.patterns[ViolationType.EMAIL] = re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            )
        
        # Phone number patterns (US format)
        if self.enabled_checks.get('phone_detection', True):
            self.patterns[ViolationType.PHONE] = re.compile(
                r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                re.IGNORECASE
            )
        
        # Social Security Number patterns
        if self.enabled_checks.get('ssn_detection', True):
            self.patterns[ViolationType.SSN] = re.compile(
                r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b'
            )
        
        # Credit card patterns (basic Luhn algorithm check can be added)
        if self.enabled_checks.get('credit_card_detection', True):
            self.patterns[ViolationType.CREDIT_CARD] = re.compile(
                r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[-\s]?(?:\d{4}[-\s]?){2,3}\d{4}\b'
            )
        
        # IP Address patterns
        self.patterns[ViolationType.IP_ADDRESS] = re.compile(
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        )
        
        # Date of birth patterns
        self.patterns[ViolationType.DATE_OF_BIRTH] = re.compile(
            r'\b(?:(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}|'
            r'(?:0[1-9]|[12][0-9]|3[01])[-/](?:0[1-9]|1[0-2])[-/](?:19|20)\d{2})\b'
        )
        
        # Driver's license patterns (simplified)
        self.patterns[ViolationType.DRIVER_LICENSE] = re.compile(
            r'\b[A-Z]{1,2}\d{6,8}\b|\b\d{8,9}\b',
            re.IGNORECASE
        )
        
        # Bank account patterns
        self.patterns[ViolationType.BANK_ACCOUNT] = re.compile(
            r'\b\d{8,17}\b'
        )
        
        # Medical information keywords
        if self.enabled_checks.get('medical_info', True):
            medical_keywords = [
                r'\bdiagnos(is|ed)\b', r'\bprescription\b', r'\bmedication\b',
                r'\bdoctor\b', r'\bphysician\b', r'\bhospital\b', r'\bpatient\b',
                r'\btreatment\b', r'\bsymptom\b', r'\bdisease\b', r'\billness\b',
                r'\bmedical record\b', r'\bhealth condition\b'
            ]
            self.patterns[ViolationType.MEDICAL_INFO] = re.compile(
                '|'.join(medical_keywords), re.IGNORECASE
            )
        
        # Financial information keywords
        if self.enabled_checks.get('financial_info', True):
            financial_keywords = [
                r'\bbank account\b', r'\brouting number\b', r'\bsort code\b',
                r'\biban\b', r'\bswift code\b', r'\bpayment\b', r'\bsalary\b',
                r'\bincome\b', r'\bcredit score\b', r'\bloan\b', r'\bmortgage\b'
            ]
            self.patterns[ViolationType.FINANCIAL_INFO] = re.compile(
                '|'.join(financial_keywords), re.IGNORECASE
            )
        
        # Custom patterns from config
        for pattern_name, pattern_string in self.custom_patterns.items():
            if pattern_name == 'api_keys':
                self.patterns[ViolationType.API_KEY] = re.compile(pattern_string, re.IGNORECASE)
            elif pattern_name == 'tokens':
                self.patterns[ViolationType.TOKEN] = re.compile(pattern_string, re.IGNORECASE)
    
    def detect_violations(self, text: str) -> List[PrivacyViolation]:
        """
        Detect privacy violations in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected privacy violations
        """
        violations = []
        
        # Pattern-based detection
        for violation_type, pattern in self.patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                confidence = self._calculate_confidence(violation_type, match.group())
                violation = PrivacyViolation(
                    violation_type=violation_type,
                    confidence=confidence,
                    text_span=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    description=f"Detected {violation_type.value}: {match.group()}"
                )
                violations.append(violation)
        
        # NLP-based entity detection
        if self.nlp:
            violations.extend(self._detect_entities_nlp(text))
        
        return violations
    
    def _detect_entities_nlp(self, text: str) -> List[PrivacyViolation]:
        """Use spaCy NER to detect privacy-sensitive entities."""
        violations = []
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                violation = PrivacyViolation(
                    violation_type=ViolationType.ADDRESS,  # Using ADDRESS as generic PII
                    confidence=0.8,
                    text_span=ent.text,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    description=f"Detected person name: {ent.text}"
                )
                violations.append(violation)
            
            elif ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity, location
                violation = PrivacyViolation(
                    violation_type=ViolationType.ADDRESS,
                    confidence=0.6,
                    text_span=ent.text,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    description=f"Detected location: {ent.text}"
                )
                violations.append(violation)
            
            elif ent.label_ == "ORG":  # Organization
                violation = PrivacyViolation(
                    violation_type=ViolationType.ADDRESS,
                    confidence=0.4,
                    text_span=ent.text,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    description=f"Detected organization: {ent.text}"
                )
                violations.append(violation)
        
        return violations
    
    def _calculate_confidence(self, violation_type: ViolationType, text: str) -> float:
        """
        Calculate confidence score for a detected violation.
        
        Args:
            violation_type: Type of violation detected
            text: The matched text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = {
            ViolationType.EMAIL: 0.95,
            ViolationType.PHONE: 0.85,
            ViolationType.SSN: 0.9,
            ViolationType.CREDIT_CARD: 0.8,
            ViolationType.IP_ADDRESS: 0.7,
            ViolationType.API_KEY: 0.8,
            ViolationType.TOKEN: 0.75,
            ViolationType.DATE_OF_BIRTH: 0.6,
            ViolationType.DRIVER_LICENSE: 0.7,
            ViolationType.BANK_ACCOUNT: 0.6,
            ViolationType.MEDICAL_INFO: 0.5,
            ViolationType.FINANCIAL_INFO: 0.5,
            ViolationType.ADDRESS: 0.6,
        }
        
        confidence = base_confidence.get(violation_type, 0.5)
        
        # Adjust confidence based on text length and context
        if len(text) < 3:
            confidence *= 0.5  # Very short matches are less reliable
        elif len(text) > 50:
            confidence *= 0.8  # Very long matches might be false positives
        
        return min(confidence, 1.0)
    
    def calculate_privacy_score(self, violations: List[PrivacyViolation]) -> float:
        """
        Calculate overall privacy violation score.
        
        Args:
            violations: List of detected violations
            
        Returns:
            Privacy score between 0.0 (no violations) and 1.0 (severe violations)
        """
        if not violations:
            return 0.0
        
        # Weight different violation types
        weights = {
            ViolationType.SSN: 1.0,
            ViolationType.CREDIT_CARD: 1.0,
            ViolationType.API_KEY: 1.0,
            ViolationType.TOKEN: 1.0,
            ViolationType.PHONE: 0.8,
            ViolationType.EMAIL: 0.7,
            ViolationType.ADDRESS: 0.6,
            ViolationType.DATE_OF_BIRTH: 0.8,
            ViolationType.DRIVER_LICENSE: 0.8,
            ViolationType.BANK_ACCOUNT: 0.9,
            ViolationType.IP_ADDRESS: 0.5,
            ViolationType.MEDICAL_INFO: 0.7,
            ViolationType.FINANCIAL_INFO: 0.7,
            ViolationType.PASSPORT: 0.9,
        }
        
        # Calculate weighted score
        total_score = 0.0
        max_possible_score = 0.0
        
        for violation in violations:
            weight = weights.get(violation.violation_type, 0.5)
            weighted_score = violation.confidence * weight
            total_score += weighted_score
            max_possible_score += weight
        
        if max_possible_score == 0:
            return 0.0
        
        # Normalize to 0-1 range
        normalized_score = min(total_score / max_possible_score, 1.0)
        
        return normalized_score
    
    def get_violation_summary(self, violations: List[PrivacyViolation]) -> Dict[str, Any]:
        """
        Get a summary of detected violations.
        
        Args:
            violations: List of detected violations
            
        Returns:
            Dictionary with violation summary
        """
        summary = {
            'total_violations': len(violations),
            'violation_types': {},
            'highest_confidence': 0.0,
            'privacy_score': self.calculate_privacy_score(violations)
        }
        
        for violation in violations:
            violation_type = violation.violation_type.value
            if violation_type not in summary['violation_types']:
                summary['violation_types'][violation_type] = {
                    'count': 0,
                    'max_confidence': 0.0,
                    'instances': []
                }
            
            summary['violation_types'][violation_type]['count'] += 1
            summary['violation_types'][violation_type]['max_confidence'] = max(
                summary['violation_types'][violation_type]['max_confidence'],
                violation.confidence
            )
            summary['violation_types'][violation_type]['instances'].append({
                'text': violation.text_span,
                'confidence': violation.confidence,
                'position': (violation.start_pos, violation.end_pos)
            })
            
            summary['highest_confidence'] = max(summary['highest_confidence'], violation.confidence)
        
        return summary
