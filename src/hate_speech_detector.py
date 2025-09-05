"""
Hate Speech Detection Module

Uses Hugging Face transformers models to detect hate speech and toxic content
in text prompts with configurable models and thresholds.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
from pathlib import Path

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification, 
        pipeline,
        PreTrainedModel,
        PreTrainedTokenizer
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Hate speech detection will be disabled.")


@dataclass
class HateSpeechResult:
    """Result from hate speech detection."""
    is_hate_speech: bool
    confidence: float
    label: str
    all_scores: Dict[str, float]
    model_used: str
    processing_time: float


class HateSpeechDetector:
    """
    Detects hate speech using Hugging Face transformer models.
    
    Supported models:
    - martin-ha/toxic-bert: Specialized for toxicity detection
    - unitary/toxic-bert: Another toxicity detection model
    - Hate-speech-CNERG/dehatebert-mono-english: Dedicated hate speech detection
    - cardiffnlp/twitter-roberta-base-hate-multiclass-latest: Multi-class hate detection
    
    Features:
    - Multiple model support with configurable selection
    - Model caching for improved performance  
    - Configurable confidence thresholds
    - Detailed scoring with multiple categories
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the hate speech detector.
        
        Args:
            config: Configuration dictionary with hate speech detection settings
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for hate speech detection")
        
        self.config = config or {}
        self.hate_speech_config = self.config.get('hate_speech', {})
        
        # Model configuration
        self.model_name = self.hate_speech_config.get('model_name', 'martin-ha/toxic-bert')
        self.threshold = self.hate_speech_config.get('threshold', 0.7)
        self.use_cache = self.hate_speech_config.get('use_cache', True)
        self.max_length = self.hate_speech_config.get('max_length', 512)
        
        # Caching configuration
        caching_config = self.config.get('caching', {})
        self.cache_dir = caching_config.get('model_cache_dir', './models_cache')
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._model_loaded = False
        
        # Performance tracking
        self._total_predictions = 0
        self._total_time = 0.0
        
        # Load model if caching is enabled
        if self.use_cache:
            self._load_model()
    
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        try:
            start_time = time.time()
            
            # Create cache directory
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Loading hate speech model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            self._model_loaded = True
            load_time = time.time() - start_time
            
            logging.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logging.error(f"Failed to load hate speech model {self.model_name}: {str(e)}")
            self._model_loaded = False
            raise
    
    def detect_hate_speech(self, text: str) -> HateSpeechResult:
        """
        Detect hate speech in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            HateSpeechResult with detection results
        """
        if not self._model_loaded:
            if self.use_cache:
                raise RuntimeError("Model failed to load")
            else:
                self._load_model()
        
        start_time = time.time()
        
        try:
            # Truncate text if too long
            if len(text) > self.max_length:
                text = text[:self.max_length]
                logging.warning(f"Text truncated to {self.max_length} characters")
            
            # Get predictions
            results = self.pipeline(text)
            
            # Parse results based on model type
            parsed_results = self._parse_model_results(results)
            
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            
            # Determine if hate speech based on threshold
            is_hate_speech = parsed_results['max_toxic_score'] >= self.threshold
            
            return HateSpeechResult(
                is_hate_speech=is_hate_speech,
                confidence=parsed_results['max_toxic_score'],
                label=parsed_results['primary_label'],
                all_scores=parsed_results['all_scores'],
                model_used=self.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.error(f"Error during hate speech detection: {str(e)}")
            # Return safe default
            return HateSpeechResult(
                is_hate_speech=True,  # Err on the side of caution
                confidence=1.0,
                label="error",
                all_scores={"error": 1.0},
                model_used=self.model_name,
                processing_time=time.time() - start_time
            )
    
    def _parse_model_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse results from different model types into a consistent format.
        
        Args:
            results: Raw results from the transformer pipeline
            
        Returns:
            Parsed results with standardized format
        """
        all_scores = {}
        max_toxic_score = 0.0
        primary_label = "non_toxic"
        
        # Handle different model output formats
        if self.model_name in ['martin-ha/toxic-bert', 'unitary/toxic-bert']:
            # These models typically output TOXIC/NON_TOXIC labels
            for result in results:
                label = result['label'].lower()
                score = result['score']
                all_scores[label] = score
                
                if 'toxic' in label and score > max_toxic_score:
                    max_toxic_score = score
                    primary_label = label
        
        elif 'dehatebert' in self.model_name.lower():
            # DeHateBERT outputs HATE/NON_HATE
            for result in results:
                label = result['label'].lower()
                score = result['score']
                all_scores[label] = score
                
                if 'hate' in label and score > max_toxic_score:
                    max_toxic_score = score
                    primary_label = label
        
        elif 'twitter-roberta' in self.model_name.lower():
            # Multi-class hate detection (hate, offensive, neither)
            for result in results:
                label = result['label'].lower()
                score = result['score']
                all_scores[label] = score
                
                # Consider both 'hate' and 'offensive' as problematic
                if label in ['hate', 'offensive'] and score > max_toxic_score:
                    max_toxic_score = score
                    primary_label = label
        
        else:
            # Generic handling for unknown models
            for result in results:
                label = result['label'].lower()
                score = result['score']
                all_scores[label] = score
                
                # Look for common toxic/hate indicators
                toxic_keywords = ['toxic', 'hate', 'offensive', 'harmful', 'abusive']
                if any(keyword in label for keyword in toxic_keywords) and score > max_toxic_score:
                    max_toxic_score = score
                    primary_label = label
        
        return {
            'all_scores': all_scores,
            'max_toxic_score': max_toxic_score,
            'primary_label': primary_label
        }
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics."""
        self._total_predictions += 1
        self._total_time += processing_time
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self._total_predictions == 0:
            return {
                'total_predictions': 0,
                'average_time': 0.0,
                'total_time': 0.0
            }
        
        return {
            'total_predictions': self._total_predictions,
            'average_time': self._total_time / self._total_predictions,
            'total_time': self._total_time
        }
    
    def batch_detect(self, texts: List[str]) -> List[HateSpeechResult]:
        """
        Detect hate speech in multiple texts efficiently.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of HateSpeechResult objects
        """
        if not self._model_loaded:
            if self.use_cache:
                raise RuntimeError("Model failed to load")
            else:
                self._load_model()
        
        results = []
        
        # Process texts individually for now
        # TODO: Implement true batch processing for better performance
        for text in texts:
            result = self.detect_hate_speech(text)
            results.append(result)
        
        return results
    
    def update_threshold(self, new_threshold: float):
        """
        Update the detection threshold.
        
        Args:
            new_threshold: New threshold value between 0.0 and 1.0
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.threshold = new_threshold
        logging.info(f"Updated hate speech threshold to {new_threshold}")
    
    def switch_model(self, new_model_name: str):
        """
        Switch to a different model.
        
        Args:
            new_model_name: Name of the new model to use
        """
        if new_model_name == self.model_name:
            return  # No change needed
        
        logging.info(f"Switching from {self.model_name} to {new_model_name}")
        
        # Clear current model
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._model_loaded = False
        
        # Update config and reload
        self.model_name = new_model_name
        self._load_model()
    
    def get_supported_models(self) -> List[Dict[str, str]]:
        """Get list of supported models with descriptions."""
        return [
            {
                "name": "martin-ha/toxic-bert",
                "description": "BERT model fine-tuned for toxicity detection",
                "type": "binary_toxicity"
            },
            {
                "name": "unitary/toxic-bert", 
                "description": "Alternative BERT model for toxicity detection",
                "type": "binary_toxicity"
            },
            {
                "name": "Hate-speech-CNERG/dehatebert-mono-english",
                "description": "Specialized model for hate speech detection",
                "type": "binary_hate"
            },
            {
                "name": "cardiffnlp/twitter-roberta-base-hate-multiclass-latest",
                "description": "Multi-class hate detection model trained on Twitter data",
                "type": "multiclass_hate"
            }
        ]
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if self.pipeline is not None:
            del self.pipeline
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._model_loaded = False
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info("Model resources cleaned up")
