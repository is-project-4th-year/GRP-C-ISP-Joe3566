"""
LLM API Integration Module

Provides integration layer to connect the compliance filter with actual LLM API calls,
with before/after filtering hooks and comprehensive monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .compliance_filter import ComplianceFilter, ComplianceResult, ComplianceAction
from .feedback_system import FeedbackSystem


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    HUGGING_FACE = "hugging_face"
    CUSTOM = "custom"


@dataclass
class LLMRequest:
    """Represents an LLM API request."""
    prompt: str
    provider: LLMProvider
    model: str
    parameters: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Represents an LLM API response."""
    content: str
    provider: LLMProvider
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    compliance_passed: bool = True
    compliance_result: Optional[ComplianceResult] = None


class LLMIntegration:
    """
    Integration layer for LLM APIs with compliance filtering.
    
    Features:
    - Pre-processing compliance checks
    - Post-processing content filtering
    - Multiple LLM provider support
    - Rate limiting and monitoring
    - Comprehensive logging and audit trails
    - Async support for better performance
    """
    
    def __init__(
        self, 
        compliance_filter: ComplianceFilter,
        config: Optional[Dict[str, Any]] = None,
        feedback_system: Optional[FeedbackSystem] = None
    ):
        """
        Initialize LLM integration.
        
        Args:
            compliance_filter: The compliance filter instance
            config: Configuration dictionary
            feedback_system: Optional feedback system
        """
        self.compliance_filter = compliance_filter
        self.feedback_system = feedback_system
        self.config = config or {}
        
        # Integration configuration
        integration_config = self.config.get('llm_integration', {})
        self.timeout_seconds = integration_config.get('timeout_seconds', 30)
        self.max_retries = integration_config.get('max_retries', 3)
        self.retry_delay = integration_config.get('retry_delay', 1.0)
        self.requests_per_minute = integration_config.get('requests_per_minute', 60)
        
        # Rate limiting
        self._request_times: List[float] = []\n        self._last_cleanup = time.time()\n        \n        # Statistics\n        self._stats = {\n            'total_requests': 0,\n            'blocked_requests': 0,\n            'warned_requests': 0,\n            'allowed_requests': 0,\n            'failed_requests': 0,\n            'average_response_time': 0.0,\n            'total_response_time': 0.0,\n            'by_provider': {}\n        }\n        \n        # Custom LLM handlers\n        self._custom_handlers: Dict[LLMProvider, Callable] = {}\n        \n        logging.info(\"LLMIntegration initialized\")\n    \n    def register_custom_handler(self, provider: LLMProvider, handler: Callable):\n        \"\"\"\n        Register a custom handler for an LLM provider.\n        \n        Args:\n            provider: The LLM provider\n            handler: Async callable that takes (request, **kwargs) and returns response content\n        \"\"\"\n        self._custom_handlers[provider] = handler\n        logging.info(f\"Registered custom handler for {provider.value}\")\n    \n    async def process_request(\n        self, \n        request: LLMRequest,\n        check_input: bool = True,\n        check_output: bool = False\n    ) -> LLMResponse:\n        \"\"\"\n        Process an LLM request with compliance filtering.\n        \n        Args:\n            request: The LLM request to process\n            check_input: Whether to check input compliance\n            check_output: Whether to check output compliance\n            \n        Returns:\n            LLM response with compliance information\n        \"\"\"\n        start_time = time.time()\n        \n        try:\n            # Rate limiting check\n            if not self._check_rate_limit():\n                raise RuntimeError(\"Rate limit exceeded\")\n            \n            # Pre-processing compliance check\n            compliance_result = None\n            if check_input:\n                compliance_result = self.compliance_filter.check_compliance(\n                    request.prompt,\n                    user_context={\n                        'user_id': request.user_id,\n                        'session_id': request.session_id,\n                        'provider': request.provider.value,\n                        'model': request.model,\n                        **request.context or {}\n                    }\n                )\n                \n                # Handle compliance action\n                if compliance_result.action == ComplianceAction.BLOCK:\n                    self._update_stats('blocked', request.provider, time.time() - start_time)\n                    return self._create_blocked_response(request, compliance_result)\n                \n                elif compliance_result.action == ComplianceAction.WARN:\n                    logging.warning(f\"Compliance warning for request: {compliance_result.reasoning}\")\n                    self._update_stats('warned', request.provider, time.time() - start_time)\n            \n            # Make LLM API call\n            llm_response = await self._call_llm_api(request)\n            \n            # Post-processing compliance check\n            output_compliance_result = None\n            if check_output and llm_response:\n                output_compliance_result = self.compliance_filter.check_compliance(\n                    llm_response,\n                    user_context={\n                        'type': 'output_check',\n                        'user_id': request.user_id,\n                        'session_id': request.session_id,\n                        'provider': request.provider.value,\n                        'model': request.model\n                    }\n                )\n                \n                if output_compliance_result.action == ComplianceAction.BLOCK:\n                    logging.warning(\"LLM output blocked due to compliance violation\")\n                    llm_response = \"I apologize, but I cannot provide that response due to content policy violations.\"\n            \n            processing_time = time.time() - start_time\n            self._update_stats('allowed', request.provider, processing_time)\n            \n            # Create response\n            response = LLMResponse(\n                content=llm_response,\n                provider=request.provider,\n                model=request.model,\n                compliance_passed=True,\n                compliance_result=compliance_result or output_compliance_result\n            )\n            \n            # Request feedback if needed\n            if self.feedback_system and compliance_result:\n                if self.feedback_system.should_request_feedback(compliance_result):\n                    logging.info(\"Compliance result may benefit from feedback\")\n            \n            return response\n            \n        except Exception as e:\n            processing_time = time.time() - start_time\n            self._update_stats('failed', request.provider, processing_time)\n            logging.error(f\"Error processing LLM request: {e}\")\n            \n            return LLMResponse(\n                content=\"I apologize, but I encountered an error processing your request.\",\n                provider=request.provider,\n                model=request.model,\n                compliance_passed=False,\n                metadata={'error': str(e)}\n            )\n    \n    def _create_blocked_response(self, request: LLMRequest, compliance_result: ComplianceResult) -> LLMResponse:\n        \"\"\"Create a response for blocked requests.\"\"\"\n        blocked_message = (\n            \"I cannot process this request due to content policy violations. \"\n            f\"Reason: {compliance_result.reasoning}\"\n        )\n        \n        return LLMResponse(\n            content=blocked_message,\n            provider=request.provider,\n            model=request.model,\n            compliance_passed=False,\n            compliance_result=compliance_result\n        )\n    \n    async def _call_llm_api(self, request: LLMRequest) -> str:\n        \"\"\"\n        Make the actual LLM API call.\n        \n        Args:\n            request: The LLM request\n            \n        Returns:\n            Response content from the LLM\n        \"\"\"\n        # Check for custom handler\n        if request.provider in self._custom_handlers:\n            handler = self._custom_handlers[request.provider]\n            return await handler(request)\n        \n        # Built-in provider support\n        if request.provider == LLMProvider.OPENAI:\n            return await self._call_openai(request)\n        elif request.provider == LLMProvider.ANTHROPIC:\n            return await self._call_anthropic(request)\n        elif request.provider == LLMProvider.AZURE_OPENAI:\n            return await self._call_azure_openai(request)\n        elif request.provider == LLMProvider.HUGGING_FACE:\n            return await self._call_hugging_face(request)\n        else:\n            raise ValueError(f\"Unsupported LLM provider: {request.provider}\")\n    \n    async def _call_openai(self, request: LLMRequest) -> str:\n        \"\"\"Call OpenAI API.\"\"\"\n        try:\n            import openai\n            \n            response = await openai.ChatCompletion.acreate(\n                model=request.model,\n                messages=[\n                    {\"role\": \"user\", \"content\": request.prompt}\n                ],\n                **request.parameters\n            )\n            \n            return response.choices[0].message.content\n            \n        except ImportError:\n            raise ImportError(\"openai library not installed\")\n        except Exception as e:\n            logging.error(f\"OpenAI API call failed: {e}\")\n            raise\n    \n    async def _call_anthropic(self, request: LLMRequest) -> str:\n        \"\"\"Call Anthropic Claude API.\"\"\"\n        try:\n            import anthropic\n            \n            client = anthropic.AsyncAnthropic()\n            \n            response = await client.messages.create(\n                model=request.model,\n                max_tokens=request.parameters.get('max_tokens', 1000),\n                messages=[\n                    {\"role\": \"user\", \"content\": request.prompt}\n                ]\n            )\n            \n            return response.content[0].text\n            \n        except ImportError:\n            raise ImportError(\"anthropic library not installed\")\n        except Exception as e:\n            logging.error(f\"Anthropic API call failed: {e}\")\n            raise\n    \n    async def _call_azure_openai(self, request: LLMRequest) -> str:\n        \"\"\"Call Azure OpenAI API.\"\"\"\n        try:\n            import openai\n            \n            # Configure Azure OpenAI\n            openai.api_type = \"azure\"\n            openai.api_base = request.parameters.get('api_base')\n            openai.api_version = request.parameters.get('api_version', '2023-05-15')\n            openai.api_key = request.parameters.get('api_key')\n            \n            response = await openai.ChatCompletion.acreate(\n                engine=request.model,  # deployment name in Azure\n                messages=[\n                    {\"role\": \"user\", \"content\": request.prompt}\n                ],\n                **{k: v for k, v in request.parameters.items() \n                   if k not in ['api_base', 'api_version', 'api_key']}\n            )\n            \n            return response.choices[0].message.content\n            \n        except ImportError:\n            raise ImportError(\"openai library not installed\")\n        except Exception as e:\n            logging.error(f\"Azure OpenAI API call failed: {e}\")\n            raise\n    \n    async def _call_hugging_face(self, request: LLMRequest) -> str:\n        \"\"\"Call Hugging Face API.\"\"\"\n        try:\n            import aiohttp\n            \n            api_key = request.parameters.get('api_key')\n            if not api_key:\n                raise ValueError(\"Hugging Face API key required\")\n            \n            url = f\"https://api-inference.huggingface.co/models/{request.model}\"\n            headers = {\"Authorization\": f\"Bearer {api_key}\"}\n            payload = {\n                \"inputs\": request.prompt,\n                \"parameters\": {k: v for k, v in request.parameters.items() if k != 'api_key'}\n            }\n            \n            async with aiohttp.ClientSession() as session:\n                async with session.post(url, headers=headers, json=payload) as response:\n                    if response.status == 200:\n                        result = await response.json()\n                        if isinstance(result, list) and len(result) > 0:\n                            return result[0].get('generated_text', '')\n                        return str(result)\n                    else:\n                        error_text = await response.text()\n                        raise Exception(f\"HuggingFace API error: {error_text}\")\n                        \n        except ImportError:\n            raise ImportError(\"aiohttp library required for Hugging Face integration\")\n        except Exception as e:\n            logging.error(f\"Hugging Face API call failed: {e}\")\n            raise\n    \n    def _check_rate_limit(self) -> bool:\n        \"\"\"Check if request is within rate limits.\"\"\"\n        current_time = time.time()\n        \n        # Clean old requests (older than 1 minute)\n        if current_time - self._last_cleanup > 60:\n            cutoff_time = current_time - 60\n            self._request_times = [t for t in self._request_times if t > cutoff_time]\n            self._last_cleanup = current_time\n        \n        # Check rate limit\n        if len(self._request_times) >= self.requests_per_minute:\n            return False\n        \n        # Add current request time\n        self._request_times.append(current_time)\n        return True\n    \n    def _update_stats(self, status: str, provider: LLMProvider, processing_time: float):\n        \"\"\"Update internal statistics.\"\"\"\n        self._stats['total_requests'] += 1\n        self._stats[f'{status}_requests'] += 1\n        self._stats['total_response_time'] += processing_time\n        self._stats['average_response_time'] = (\n            self._stats['total_response_time'] / self._stats['total_requests']\n        )\n        \n        # Provider-specific stats\n        provider_name = provider.value\n        if provider_name not in self._stats['by_provider']:\n            self._stats['by_provider'][provider_name] = {\n                'total': 0, 'blocked': 0, 'warned': 0, 'allowed': 0, 'failed': 0\n            }\n        \n        self._stats['by_provider'][provider_name]['total'] += 1\n        self._stats['by_provider'][provider_name][status] += 1\n    \n    def batch_process_requests(\n        self, \n        requests: List[LLMRequest],\n        check_input: bool = True,\n        check_output: bool = False\n    ) -> List[LLMResponse]:\n        \"\"\"Process multiple requests concurrently.\"\"\"\n        async def process_batch():\n            tasks = [\n                self.process_request(req, check_input, check_output) \n                for req in requests\n            ]\n            return await asyncio.gather(*tasks, return_exceptions=True)\n        \n        return asyncio.run(process_batch())\n    \n    def get_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get integration statistics.\"\"\"\n        return self._stats.copy()\n    \n    def export_audit_log(self, output_file: str, start_date: Optional[str] = None, end_date: Optional[str] = None):\n        \"\"\"Export audit logs for compliance reporting.\"\"\"\n        # This would typically read from persistent logs\n        # For now, we'll export current statistics\n        audit_data = {\n            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),\n            'statistics': self.get_statistics(),\n            'compliance_stats': self.compliance_filter.get_performance_stats(),\n            'configuration': self.compliance_filter.get_configuration_summary()\n        }\n        \n        with open(output_file, 'w', encoding='utf-8') as f:\n            json.dump(audit_data, f, indent=2)\n        \n        logging.info(f\"Audit log exported to {output_file}\")\n    \n    def update_configuration(self, new_config: Dict[str, Any]):\n        \"\"\"Update integration configuration.\"\"\"\n        self.config.update(new_config)\n        \n        # Update specific settings\n        integration_config = self.config.get('llm_integration', {})\n        self.timeout_seconds = integration_config.get('timeout_seconds', self.timeout_seconds)\n        self.max_retries = integration_config.get('max_retries', self.max_retries)\n        self.retry_delay = integration_config.get('retry_delay', self.retry_delay)\n        self.requests_per_minute = integration_config.get('requests_per_minute', self.requests_per_minute)\n        \n        logging.info(\"LLM integration configuration updated\")\n    \n    def cleanup(self):\n        \"\"\"Clean up resources.\"\"\"\n        if self.compliance_filter:\n            self.compliance_filter.cleanup()\n        \n        logging.info(\"LLMIntegration resources cleaned up\")
