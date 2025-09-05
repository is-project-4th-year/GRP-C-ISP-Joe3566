# LLM Compliance Filter

A comprehensive compliance filter for Large Language Models that detects and prevents privacy violations and hate speech in prompts before they are processed by LLMs.

## Features

- **Privacy Violation Detection**: Identifies PII, sensitive data patterns, and other privacy-sensitive content
- **Hate Speech Detection**: Uses Hugging Face transformers models to detect harmful content
- **Configurable Thresholds**: Adjustable risk tolerance levels for different use cases
- **Comprehensive Logging**: Structured logging for compliance audits and monitoring
- **LLM Integration**: Easy integration with various LLM APIs
- **Feedback System**: Mechanism to improve filter accuracy over time

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your settings in `config/default.yaml`

3. Run example usage:
   ```python
   from src.compliance_filter import ComplianceFilter
   
   filter = ComplianceFilter()
   result = filter.check_compliance("Your prompt here")
   ```

## Components

- `src/privacy_detector.py`: Privacy violation detection
- `src/hate_speech_detector.py`: Hate speech detection using transformers
- `src/compliance_filter.py`: Main compliance scoring and filtering
- `src/feedback_system.py`: Feedback and improvement mechanisms
- `src/llm_integration.py`: LLM API integration layer
- `config/`: Configuration files with adjustable thresholds
- `logs/`: Compliance audit logs

## Configuration

Adjust thresholds and models in `config/default.yaml`:
- Privacy detection sensitivity
- Hate speech model selection
- Risk tolerance levels
- Logging preferences

## Testing

Run tests with:
```bash
pytest tests/
```

## License

MIT License
