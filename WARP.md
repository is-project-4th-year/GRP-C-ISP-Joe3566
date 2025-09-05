# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a Python-based LLM Compliance Filter that detects privacy violations and hate speech in prompts before they are processed by Large Language Models. The system provides configurable thresholds, comprehensive logging, and feedback mechanisms for continuous improvement.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install spaCy model for enhanced privacy detection (optional but recommended)
python -m spacy download en_core_web_sm
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_compliance_filter.py

# Run tests with verbose output
pytest tests/ -v

# Run specific test method
pytest tests/test_compliance_filter.py::TestPrivacyDetector::test_email_detection

# Run tests and see coverage
pytest tests/ --cov=src
```

### Running Examples
```bash
# Run the comprehensive example usage script
python example_usage.py

# Run basic compliance check
python -c "from src.compliance_filter import ComplianceFilter; print(ComplianceFilter().check_compliance('Test message'))"
```

### Development Tools
```bash
# Run individual test components
python tests/test_compliance_filter.py  # Basic smoke test

# Test privacy detection specifically
python -c "from src.privacy_detector import PrivacyDetector; d = PrivacyDetector(); print(d.detect_violations('test@email.com'))"

# Test hate speech detection (requires transformers)
python -c "from src.hate_speech_detector import HateSpeechDetector; d = HateSpeechDetector(); print(d.detect_hate_speech('test message'))"
```

## Architecture Overview

### Core Components

**ComplianceFilter (`src/compliance_filter.py`)**
- Main orchestrator that combines privacy and hate speech detection
- Configurable scoring methods: weighted_average, max, product
- Manages thresholds for ALLOW/WARN/BLOCK actions
- Handles batch processing and performance statistics
- Returns structured `ComplianceResult` with detailed analysis

**PrivacyDetector (`src/privacy_detector.py`)**  
- Pattern-based detection using compiled regex for PII types:
  - Email, phone, SSN, credit cards, IP addresses, dates of birth
  - Medical and financial keywords
  - API keys and tokens via custom patterns
- Optional spaCy NLP integration for entity recognition
- Configurable violation types and risk levels
- Returns `PrivacyViolation` objects with confidence scores

**HateSpeechDetector (`src/hate_speech_detector.py`)**
- Hugging Face transformers integration (requires transformers library)
- Supports multiple models: toxic-bert, dehatebert-mono-english, etc.
- Model caching for performance with configurable cache directory
- GPU acceleration when available
- Returns `HateSpeechResult` with confidence and label classifications

**LLMIntegration (`src/llm_integration.py`)**
- Async wrapper for LLM API calls with compliance filtering
- Pre/post-processing compliance checks
- Rate limiting and retry logic
- Built-in support for OpenAI, Anthropic, Azure OpenAI, Hugging Face
- Custom handler registration system
- Comprehensive statistics and audit logging

**FeedbackSystem (`src/feedback_system.py`)**
- Collects user feedback on compliance decisions
- Supports feedback types: false positives/negatives, threshold suggestions
- Persistent storage in JSONL format
- Pattern analysis for identifying improvement opportunities
- Training data export for model refinement

### Configuration System

The system uses YAML configuration (`config/default.yaml`) with the following structure:
- `hate_speech`: Model selection, thresholds, caching options
- `privacy`: Enable/disable specific checks, risk levels, custom patterns  
- `compliance`: Scoring method, weights, action thresholds
- `logging`: Levels, formats, audit trail settings
- `llm_integration`: Timeouts, retries, rate limiting
- `feedback`: Collection settings, storage options

### Data Flow

1. **Input Processing**: Text prompt received
2. **Privacy Analysis**: Pattern matching + optional NLP entity recognition
3. **Hate Speech Analysis**: Transformer model inference (if available)
4. **Score Calculation**: Weighted combination based on configuration
5. **Action Determination**: Compare against block/warn/pass thresholds
6. **Result Generation**: Create structured response with reasoning
7. **Logging/Feedback**: Audit trail and optional feedback collection

### Key Design Patterns

**Graceful Degradation**: System continues functioning even if optional components (spaCy, transformers) are unavailable

**Configurable Scoring**: Multiple scoring methods allow different risk profiles:
- `weighted_average`: Balanced approach using configurable weights
- `max`: Conservative approach using highest risk score
- `product`: Multiplicative penalty for multiple violations

**Extensible Detection**: Custom regex patterns can be added via configuration without code changes

**Async Support**: LLM integration supports concurrent request processing for better performance

## Configuration Guidelines

### Threshold Tuning
- `block_threshold` (default 0.7): High-confidence violations that should be blocked
- `warn_threshold` (default 0.5): Medium-confidence violations that should be flagged
- Lower thresholds = more permissive, higher thresholds = more restrictive

### Model Selection
- `martin-ha/toxic-bert`: Balanced general toxicity detection
- `Hate-speech-CNERG/dehatebert-mono-english`: Specialized hate speech detection
- Custom models can be specified via `model_name` configuration

### Privacy Detection Customization
Enable/disable specific checks in `privacy.checks` section. Add custom patterns in `privacy.custom_patterns` for domain-specific PII types.

### Performance Optimization
- Enable model caching for faster inference after initial load
- Adjust `max_length` to balance accuracy vs speed
- Use GPU acceleration when available for transformer models

## Development Notes

### Dependencies
- **Required**: PyYAML, regex, structlog, requests, pytest
- **Optional**: transformers + torch (hate speech), spaCy (enhanced privacy), openai/anthropic (LLM APIs)

### Testing Philosophy
- Unit tests for individual detectors with specific violation types
- Integration tests for end-to-end workflow
- Performance tests for response time requirements
- Configuration tests for various threshold combinations

### Error Handling
- Conservative fallbacks: assume violation when detection fails
- Comprehensive logging for debugging and monitoring
- Graceful degradation when optional dependencies unavailable

### Extension Points
- Custom LLM handlers via `LLMIntegration.register_custom_handler()`
- Custom privacy patterns via configuration
- Custom scoring methods by extending `ComplianceFilter._calculate_overall_score()`
- Feedback processors by extending `FeedbackSystem` methods
