"""
Example Usage of LLM Compliance Filter

This script demonstrates how to use the compliance filter to check prompts
for privacy violations and hate speech before sending them to LLMs.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.compliance_filter import ComplianceFilter
from src.privacy_detector import PrivacyDetector
from src.hate_speech_detector import HateSpeechDetector
from src.feedback_system import FeedbackSystem, FeedbackType
from src.llm_integration import LLMIntegration, LLMRequest, LLMProvider

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_basic_usage():
    """Demonstrate basic compliance filtering."""
    print("=" * 60)
    print("BASIC COMPLIANCE FILTER USAGE")
    print("=" * 60)
    
    # Initialize the compliance filter
    filter = ComplianceFilter()
    
    # Test prompts with various compliance issues
    test_prompts = [
        "Tell me about machine learning algorithms",  # Safe prompt
        "My email is john.doe@example.com and my SSN is 123-45-6789",  # Privacy violation
        "I hate all people from that country, they are terrible",  # Hate speech
        "Generate a report using my credit card 4532-1234-5678-9012",  # Multiple violations
        "What's the weather like today?"  # Safe prompt
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        
        # Check compliance
        result = filter.check_compliance(prompt)
        
        print(f"Action: {result.action.value}")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Privacy Score: {result.privacy_score:.3f}")
        print(f"Hate Speech Score: {result.hate_speech_score:.3f}")
        print(f"Reasoning: {result.reasoning}")
        
        if result.privacy_violations:
            print(f"Privacy Violations: {len(result.privacy_violations)}")
            for violation in result.privacy_violations:
                print(f"  - {violation.violation_type.value}: {violation.text_span} (confidence: {violation.confidence:.2f})")


def example_privacy_detection():
    """Demonstrate privacy detection capabilities."""
    print("\\n" + "=" * 60)
    print("PRIVACY DETECTION EXAMPLES")
    print("=" * 60)
    
    detector = PrivacyDetector()
    
    test_texts = [
        "My name is John Smith, born on 01/15/1985",
        "Please send the report to mary@company.com",
        "My phone number is (555) 123-4567",
        "The API key is sk-abc123def456ghi789",
        "Patient was diagnosed with diabetes last month",
        "My credit card number is 4532-1234-5678-9012"
    ]
    
    for text in test_texts:
        print(f"\\nText: {text}")
        violations = detector.detect_violations(text)
        
        if violations:
            for violation in violations:
                print(f"  Found: {violation.violation_type.value} - '{violation.text_span}' (confidence: {violation.confidence:.2f})")
        else:
            print("  No violations detected")


def example_feedback_system():
    """Demonstrate feedback system usage."""
    print("\\n" + "=" * 60)
    print("FEEDBACK SYSTEM EXAMPLE")
    print("=" * 60)
    
    # Initialize components
    filter = ComplianceFilter()
    feedback_system = FeedbackSystem(filter.config)
    
    # Test a prompt
    prompt = "This content might be borderline inappropriate"
    result = filter.check_compliance(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Action: {result.action.value}")
    print(f"Should request feedback: {feedback_system.should_request_feedback(result)}")
    
    # Submit some example feedback
    feedback_id = feedback_system.submit_feedback(
        feedback_type=FeedbackType.FALSE_POSITIVE,
        original_text=prompt,
        compliance_result=result,
        user_assessment="This content is actually safe",
        confidence=0.9,
        user_id="example_user",
        notes="The filter was too aggressive on this one"
    )
    
    print(f"Submitted feedback: {feedback_id}")
    
    # Get statistics
    stats = feedback_system.get_statistics()
    print(f"Feedback statistics: {stats}")


async def example_llm_integration():
    """Demonstrate LLM integration with compliance filtering."""
    print("\\n" + "=" * 60)
    print("LLM INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Initialize components
    filter = ComplianceFilter()
    feedback_system = FeedbackSystem(filter.config)
    integration = LLMIntegration(filter, feedback_system=feedback_system)
    
    # Define a mock LLM handler (since we don't have real API keys)
    async def mock_llm_handler(request: LLMRequest) -> str:
        """Mock LLM handler for demonstration."""
        return f"This is a mock response to: '{request.prompt[:50]}...'"
    
    # Register the mock handler
    integration.register_custom_handler(LLMProvider.CUSTOM, mock_llm_handler)
    
    # Test requests
    test_requests = [
        LLMRequest(
            prompt="What are the benefits of renewable energy?",
            provider=LLMProvider.CUSTOM,
            model="mock-model",
            parameters={},
            user_id="user123"
        ),
        LLMRequest(
            prompt="My SSN is 123-45-6789, please help with my tax return",
            provider=LLMProvider.CUSTOM,
            model="mock-model", 
            parameters={},
            user_id="user123"
        ),
        LLMRequest(
            prompt="I hate those people, they should all be eliminated",
            provider=LLMProvider.CUSTOM,
            model="mock-model",
            parameters={},
            user_id="user123"
        )
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"\\n--- Request {i} ---")
        print(f"Prompt: {request.prompt}")
        
        response = await integration.process_request(request, check_input=True)
        
        print(f"Compliance Passed: {response.compliance_passed}")
        print(f"Response: {response.content}")
        
        if response.compliance_result:
            print(f"Compliance Action: {response.compliance_result.action.value}")
            print(f"Overall Score: {response.compliance_result.overall_score:.3f}")
    
    # Show statistics
    stats = integration.get_statistics()
    print(f"\\nIntegration Statistics: {stats}")


def example_configuration_adjustment():
    """Demonstrate configuration adjustment."""
    print("\\n" + "=" * 60)
    print("CONFIGURATION ADJUSTMENT EXAMPLE")
    print("=" * 60)
    
    # Initialize filter with default config
    filter = ComplianceFilter()
    
    # Test with default settings
    test_prompt = "This is a moderately suspicious prompt with some email@test.com"
    result1 = filter.check_compliance(test_prompt)
    print(f"Default config - Action: {result1.action.value}, Score: {result1.overall_score:.3f}")
    
    # Adjust thresholds to be more permissive
    filter.update_thresholds({
        'block_threshold': 0.9,  # Increased from 0.7
        'warn_threshold': 0.7    # Increased from 0.5
    })
    
    # Test with adjusted settings
    result2 = filter.check_compliance(test_prompt)
    print(f"Adjusted config - Action: {result2.action.value}, Score: {result2.overall_score:.3f}")
    
    # Show configuration summary
    config_summary = filter.get_configuration_summary()
    print(f"\\nConfiguration: {config_summary}")


def main():
    """Run all examples."""
    print("LLM Compliance Filter - Example Usage\\n")
    
    try:
        # Basic examples
        example_basic_usage()
        example_privacy_detection()
        example_feedback_system()
        example_configuration_adjustment()
        
        # Async example
        print("\\nRunning async LLM integration example...")
        asyncio.run(example_llm_integration())
        
    except ImportError as e:
        print(f"\\nNote: Some features require additional libraries: {e}")
        print("Install with: pip install -r requirements.txt")
    
    except Exception as e:
        print(f"\\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n" + "=" * 60)
    print("EXAMPLES COMPLETED")
    print("=" * 60)
    print("\\nTo use the compliance filter in your application:")
    print("1. Initialize ComplianceFilter with your config")
    print("2. Call check_compliance() on user prompts")
    print("3. Handle the returned ComplianceResult appropriately")
    print("4. Optionally integrate with FeedbackSystem for continuous improvement")
    print("5. Use LLMIntegration for seamless API integration")


if __name__ == "__main__":
    main()
