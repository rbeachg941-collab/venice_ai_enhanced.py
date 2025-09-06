import streamlit as st
import requests
import json
import time
import uuid
import os
import io
import base64
import logging
import hashlib
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from PIL import Image
from functools import wraps
from enum import Enum
import threading
from collections import deque

# ============================================================================
# INITIALIZATION AND CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Venice AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('venice_ai.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class VeniceAPIError(Exception):
    """Base exception for Venice API errors"""
    def __init__(self, message: str, code: str = None, details: Dict = None, status_code: int = None):
        self.message = message
        self.code = code
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)

class ValidationError(VeniceAPIError):
    """Raised when input validation fails"""
    pass

class AuthenticationError(VeniceAPIError):
    """Raised when authentication fails"""
    pass

class InsufficientBalanceError(VeniceAPIError):
    """Raised when account has insufficient balance"""
    pass

class RateLimitError(VeniceAPIError):
    """Raised when rate limit is exceeded"""
    pass

class ServerError(VeniceAPIError):
    """Raised when server encounters an error"""
    pass

class ServiceUnavailableError(VeniceAPIError):
    """Raised when service is unavailable"""
    pass

# ============================================================================
# CIRCUIT BREAKER PATTERN
# ============================================================================

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker pattern for API resilience"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise ServiceUnavailableError(
                        "Service temporarily unavailable due to repeated failures",
                        code="CIRCUIT_OPEN"
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info("Circuit breaker reset to CLOSED state")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                logger.warning("Circuit breaker opened due to failure in HALF_OPEN state")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

# ============================================================================
# DYNAMIC PROGRAMMING MEMOIZATION SYSTEM
# ============================================================================

class VeniceAIMemoizer:
    """Advanced memoization system for Venice AI API calls with dynamic programming optimization"""
    
    def __init__(self, max_cache_size: int = 50, default_ttl: int = 7200):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.quality_scores: Dict[str, float] = {}
        self.access_count: Dict[str, int] = {}  # Track access frequency
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
    
    def _generate_cache_key(self, prompt: str, **kwargs) -> str:
        """Create a unique hash key from prompt and parameters"""
        sorted_params = {k: v for k, v in sorted(kwargs.items()) if v is not None}
        param_string = json.dumps(sorted_params, sort_keys=True)
        key_string = f"{prompt}_{param_string}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid based on TTL"""
        current_time = time.time()
        return current_time - cache_entry['timestamp'] < cache_entry.get('ttl', self.default_ttl)
    
    def _manage_cache_size(self):
        """LFU (Least Frequently Used) cache eviction"""
        if len(self.cache) >= self.max_cache_size:
            # Find least frequently accessed entry
            if self.access_count:
                least_used_key = min(self.access_count.keys(), 
                                   key=lambda k: self.access_count[k] if k in self.cache else float('inf'))
                if least_used_key in self.cache:
                    del self.cache[least_used_key]
                    del self.access_count[least_used_key]
                    if least_used_key in self.quality_scores:
                        del self.quality_scores[least_used_key]
    
    def memoize_image_generation(self, func):
        """Decorator to memoize Venice AI image generation API calls"""
        @wraps(func)
        def wrapper(prompt: str, *args, **kwargs) -> Dict[str, Any]:
            if kwargs.pop('disable_cache', False):
                return func(prompt, *args, **kwargs)
            
            cache_key = self._generate_cache_key(prompt, **kwargs)
            
            with self._lock:
                if cache_key in self.cache:
                    cached_result = self.cache[cache_key]
                    if self._is_cache_valid(cached_result):
                        self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
                        logger.info(f"Cache hit for prompt: {prompt[:50]}...")
                        return cached_result['result']
                    else:
                        del self.cache[cache_key]
            
            result = func(prompt, *args, **kwargs)
            
            if result.get('success', False):
                with self._lock:
                    self._manage_cache_size()
                    self.cache[cache_key] = {
                        'result': result,
                        'timestamp': time.time(),
                        'prompt': prompt,
                        'params': kwargs,
                        'ttl': self.default_ttl
                    }
                    self.access_count[cache_key] = 1
            
            return result
        return wrapper
    
    def update_quality_score(self, prompt_hash: str, score: float):
        """Update quality score for a cached prompt"""
        with self._lock:
            if prompt_hash in self.quality_scores:
                self.quality_scores[prompt_hash] = (self.quality_scores[prompt_hash] + score) / 2
            else:
                self.quality_scores[prompt_hash] = score
    
    def get_best_variation(self, base_prompt: str, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """Find the best quality variation using optimized lookup"""
        with self._lock:
            # Create index for faster lookup
            prompt_words = set(base_prompt.lower().split())
            best_score = -1
            best_key = None
            
            for key, cache_entry in self.cache.items():
                cached_prompt = cache_entry['prompt']
                cached_words = set(cached_prompt.lower().split())
                
                # Quick similarity check using set operations
                if len(prompt_words & cached_words) / len(prompt_words | cached_words) >= threshold:
                    score = self.quality_scores.get(key, 0)
                    if score > best_score:
                        best_score = score
                        best_key = key
            
            return self.cache[best_key]['result'] if best_key else None
    
    def clear_cache(self):
        """Clear all cached results"""
        with self._lock:
            self.cache.clear()
            self.quality_scores.clear()
            self.access_count.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache"""
        with self._lock:
            total_size = len(self.cache)
            valid_count = sum(1 for entry in self.cache.values() if self._is_cache_valid(entry))
            avg_quality = sum(self.quality_scores.values()) / len(self.quality_scores) if self.quality_scores else 0
            
            # Estimate memory usage more accurately
            estimated_memory_mb = sum(
                len(json.dumps(entry).encode()) / (1024 * 1024)
                for entry in self.cache.values()
            )
            
            return {
                'total_entries': total_size,
                'valid_entries': valid_count,
                'average_quality': avg_quality,
                'memory_usage_mb': round(estimated_memory_mb, 2),
                'most_accessed': max(self.access_count.items(), key=lambda x: x[1])[0][:8] if self.access_count else None
            }

# ============================================================================
# ENHANCED SEED INCREMENTATION SYSTEM
# ============================================================================

class SeedManager:
    """Advanced seed management system for controlled image variations"""
    
    def __init__(self, max_history: int = 1000):
        self.seed_history = deque(maxlen=max_history)  # Limited history
        self.variation_patterns = {
            "linear": lambda base, step, i: base + (step * i),
            "geometric": lambda base, step, i: base + (step ** i),
            "alternating": lambda base, step, i: base + (step * (1 if i % 2 == 0 else -1)),
            "random_walk": lambda base, step, i: base + random.randint(-step, step),
            "fibonacci": self._fibonacci_sequence
        }
    
    def _fibonacci_sequence(self, base: int, step: int, i: int) -> int:
        """Generate Fibonacci-based seed sequence"""
        fib = [0, 1]
        for _ in range(i):
            fib.append(fib[-1] + fib[-2])
        return base + (step * fib[min(i, len(fib) - 1)])
    
    def generate_seeds(self, base_seed: Optional[int], num_images: int, 
                      increment_mode: str = "linear", step_size: int = 1) -> List[Optional[int]]:
        """Generate a sequence of seeds based on the selected mode"""
        if base_seed is None:
            return [None] * num_images
        
        pattern_func = self.variation_patterns.get(increment_mode, self.variation_patterns["linear"])
        seeds = []
        
        for i in range(num_images):
            seed = pattern_func(base_seed, step_size, i)
            # Ensure seed is within valid range
            seed = max(-999999999, min(999999999, seed))
            seeds.append(seed)
            self.seed_history.append(seed)
        
        return seeds
    
    def get_seed_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated seeds"""
        if not self.seed_history:
            return {}
        
        return {
            "total_generated": len(self.seed_history),
            "unique_seeds": len(set(self.seed_history)),
            "min_seed": min(self.seed_history),
            "max_seed": max(self.seed_history),
            "recent_seeds": list(self.seed_history)[-10:]
        }
    
    def clear_history(self):
        """Clear seed history"""
        self.seed_history.clear()

# ============================================================================
# VENICE AI CLIENT WITH OPTIMIZATIONS AND API COMPLIANCE
# ============================================================================

@dataclass
class VeniceSettings:
    """Venice AI API configuration"""
    base_url: str = "https://api.venice.ai/api/v1"
    image_path: str = "/image/generate"
    models_path: str = "/models"
    rate_limits_path: str = "/api_keys/rate_limits"
    image_styles_path: str = "/image/styles"
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    pool_connections: int = 10
    pool_maxsize: int = 10
    
    @property
    def image_endpoint(self) -> str:
        return f"{self.base_url}{self.image_path}"
    
    @property
    def models_endpoint(self) -> str:
        return f"{self.base_url}{self.models_path}"
    
    @property
    def rate_limits_endpoint(self) -> str:
        return f"{self.base_url}{self.rate_limits_path}"
    
    @property
    def image_styles_endpoint(self) -> str:
        return f"{self.base_url}{self.image_styles_path}"

class VeniceAIClient:
    """Production-ready Venice AI client with optimized image generation"""
    
    def __init__(self, api_key: str, config: VeniceSettings):
        if not api_key or not isinstance(api_key, str):
            raise ValueError("API Key must be a non-empty string")
        
        self.api_key = api_key
        self.settings = config
        self.session = self._create_session()
        self.memoizer = VeniceAIMemoizer(max_cache_size=50, default_ttl=7200)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.available_models = []
        self.available_styles = []
    
    def _create_session(self) -> requests.Session:
        """Create optimized requests session with connection pooling"""
        session = requests.Session()
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.settings.pool_connections,
            pool_maxsize=self.settings.pool_maxsize,
            max_retries=requests.adapters.Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504]
            ),
            pool_block=False
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Set headers
        session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'Venice-AI-Image-Generator/2.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
        
        return session
    
    def _handle_api_error(self, response: requests.Response) -> None:
        """Enhanced error handling with specific exceptions"""
        error_map = {
            400: ValidationError,
            401: AuthenticationError,
            402: InsufficientBalanceError,
            429: RateLimitError,
            500: ServerError,
            502: ServiceUnavailableError,
            503: ServiceUnavailableError,
            504: ServiceUnavailableError
        }
        
        try:
            error_data = response.json()
            error_message = error_data.get('error', f'HTTP {response.status_code}')
            error_code = error_data.get('code', 'UNKNOWN')
            error_details = error_data.get('details', {})
            
            # Add specific information for known error types
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    error_details['retry_after'] = retry_after
            
            exception_class = error_map.get(response.status_code, VeniceAPIError)
            
            raise exception_class(
                message=error_message,
                code=error_code,
                details=error_details,
                status_code=response.status_code
            )
        except json.JSONDecodeError:
            raise VeniceAPIError(
                f"Invalid response format",
                code="INVALID_RESPONSE",
                details={'response_text': response.text[:200]},
                status_code=response.status_code
            )
    
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        base_delay = min(2 ** attempt, 60)
        jitter = random.uniform(0, 0.1 * base_delay)
        return base_delay + jitter
    
    def validate_api_key(self) -> bool:
        """Validate the API key with circuit breaker protection"""
        try:
            def make_request():
                response = self.session.get(
                    self.settings.models_endpoint,
                    timeout=10
                )
                return response
            
            response = self.circuit_breaker.call(make_request)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API validation failed: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get available image models from the API"""
        try:
            def make_request():
                return self.session.get(
                    f"{self.settings.models_endpoint}?type=image",
                    timeout=10
                )
            
            response = self.circuit_breaker.call(make_request)
            
            if response.status_code == 200:
                data = response.json()
                models = [
                    model['id'] for model in data.get('data', [])
                    if model.get('type') == 'image' and 
                    not model.get('model_spec', {}).get('offline', False)
                ]
                self.available_models = models
                return models
            else:
                self._handle_api_error(response)
        except ServiceUnavailableError:
            logger.warning("Circuit breaker is open, using fallback models")
            return self._get_fallback_models()
        except Exception as e:
            logger.error(f"Failed to fetch models: {str(e)}")
            return self._get_fallback_models()
    
    def _get_fallback_models(self) -> List[str]:
        """Return fallback models when API is unavailable"""
        return [
            "hidream",
            "flux-dev",
            "flux-dev-uncensored",
            "stable-diffusion-3.5",
            "venice-sd35",
            "flux.1-krea",
            "lustify-sdxl",
            "pony-realism",
            "wai-Illustrious"
        ]
    
    def get_available_styles(self) -> List[str]:
        """Get available image styles from the API"""
        try:
            def make_request():
                return self.session.get(
                    self.settings.image_styles_endpoint,
                    timeout=10
                )
            
            response = self.circuit_breaker.call(make_request)
            
            if response.status_code == 200:
                data = response.json()
                styles = data.get('data', [])
                self.available_styles = styles
                return styles
            else:
                self._handle_api_error(response)
        except ServiceUnavailableError:
            logger.warning("Circuit breaker is open, using fallback styles")
            return self._get_fallback_styles()
        except Exception as e:
            logger.error(f"Failed to fetch styles: {str(e)}")
            return self._get_fallback_styles()
    
    def _get_fallback_styles(self) -> List[str]:
        """Return fallback styles when API is unavailable"""
        return [
            None,
            "3D Model",
            "Analog Film",
            "Anime",
            "Cinematic",
            "Comic Book",
            "Digital Art",
            "Fantasy",
            "Neon Punk",
            "Photographic"
        ]
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limits and balance information"""
        try:
            def make_request():
                return self.session.get(self.settings.rate_limits_endpoint)
            
            response = self.circuit_breaker.call(make_request)
            
            if response.status_code == 200:
                return response.json().get('data', {})
            else:
                self._handle_api_error(response)
        except Exception as e:
            logger.error(f"Failed to fetch rate limits: {str(e)}")
            return {}
    
    def _validate_image_params(self, prompt: str, width: int, height: int, 
                              steps: int, cfg_scale: float, seed: Optional[int],
                              negative_prompt: str) -> None:
        """Comprehensive parameter validation"""
        # Validate prompt
        if not prompt or not isinstance(prompt, str):
            raise ValidationError("Prompt must be a non-empty string")
        
        prompt = prompt.strip()
        if len(prompt) > 1500:
            raise ValidationError(f"Prompt too long: {len(prompt)}/1500 characters")
        
        # Validate negative prompt
        if negative_prompt and len(negative_prompt) > 1500:
            raise ValidationError(f"Negative prompt too long: {len(negative_prompt)}/1500 characters")
        
        # Validate dimensions
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValidationError("Width and height must be integers")
        
        if width < 256 or width > 1280:
            raise ValidationError(f"Width {width} must be between 256 and 1280")
        
        if height < 256 or height > 1280:
            raise ValidationError(f"Height {height} must be between 256 and 1280")
        
        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            logger.warning(f"Extreme aspect ratio {aspect_ratio:.2f} may cause issues")
        
        # Validate steps
        if not isinstance(steps, int):
            raise ValidationError("Steps must be an integer")
        
        if steps < 1 or steps > 50:
            raise ValidationError(f"Steps {steps} must be between 1 and 50")
        
        # Validate CFG scale
        if not isinstance(cfg_scale, (int, float)):
            raise ValidationError("CFG scale must be a number")
        
        if cfg_scale < 1.0 or cfg_scale > 20.0:
            raise ValidationError(f"CFG scale {cfg_scale} must be between 1.0 and 20.0")
        
        # Validate seed
        if seed is not None:
            if not isinstance(seed, int):
                raise ValidationError("Seed must be an integer or None")
            
            if seed < -999999999 or seed > 999999999:
                raise ValidationError(f"Seed {seed} must be between -999999999 and 999999999")
    
    @VeniceAIMemoizer().memoize_image_generation
    def generate_image(
        self,
        prompt: str,
        model: str = "hidream",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 7.5,
        seed: Optional[int] = None,
        negative_prompt: str = "",
        style_preset: Optional[str] = None,
        safe_mode: bool = True,
        format: str = "png",
        disable_cache: bool = False
    ) -> Dict[str, Any]:
        """Optimized image generation with comprehensive validation and error handling"""
        
        # Validate all parameters
        self._validate_image_params(prompt, width, height, steps, cfg_scale, seed, negative_prompt)
        
        # Validate format
        valid_formats = ["png", "jpeg", "webp"]
        if format not in valid_formats:
            raise ValidationError(f"Format must be one of {valid_formats}")
        
        payload = {
            'model': model,
            'prompt': prompt.strip(),
            'width': width,
            'height': height,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'format': format,
            'return_binary': False,
            'safe_mode': safe_mode
        }
        
        # Add optional parameters
        if negative_prompt:
            payload['negative_prompt'] = negative_prompt.strip()
        if seed is not None:
            payload['seed'] = seed
        if style_preset:
            payload['style_preset'] = style_preset
        
        # Implement retry logic with circuit breaker
        max_retries = self.settings.max_retries
        
        for attempt in range(max_retries):
            try:
                def make_request():
                    return self.session.post(
                        self.settings.image_endpoint,
                        json=payload,
                        timeout=self.settings.timeout
                    )
                
                response = self.circuit_breaker.call(make_request)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for content violation headers
                    is_blurred = response.headers.get('x-venice-is-blurred', 'false').lower() == 'true'
                    is_violation = response.headers.get('x-venice-is-content-violation', 'false').lower() == 'true'
                    
                    if 'images' in data and data['images']:
                        images = []
                        for img_b64 in data['images']:
                            if img_b64.startswith('data:'):
                                images.append(img_b64)
                            else:
                                images.append(f"data:image/{format};base64,{img_b64}")
                        
                        return {
                            'success': True,
                            'images': images,
                            'prompt': prompt,
                            'model': model,
                            'timestamp': datetime.now(),
                            'id': data.get('id', str(uuid.uuid4())),
                            'timing': data.get('timing', {}),
                            'is_blurred': is_blurred,
                            'is_violation': is_violation
                        }
                    else:
                        raise ValidationError("No images found in API response")
                
                elif response.status_code == 429:
                    # Rate limit - implement exponential backoff
                    if attempt < max_retries - 1:
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            wait_time = float(retry_after)
                        else:
                            wait_time = self._exponential_backoff(attempt)
                        
                        logger.warning(f"Rate limit hit, waiting {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise RateLimitError("Maximum retries exceeded for rate limit")
                else:
                    self._handle_api_error(response)
                    
            except ServiceUnavailableError as e:
                if attempt < max_retries - 1:
                    wait_time = self._exponential_backoff(attempt)
                    logger.warning(f"Service unavailable, retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                if attempt < max_retries - 1 and not isinstance(e, (ValidationError, AuthenticationError)):
                    wait_time = self._exponential_backoff(attempt)
                    logger.warning(f"Request failed, retrying in {wait_time:.1f} seconds: {str(e)}")
                    time.sleep(wait_time)
                    continue
                
                return {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'prompt': prompt,
                    'timestamp': datetime.now()
                }
        
        return {
            'success': False,
            'error': 'Maximum retries exceeded',
            'prompt': prompt,
            'timestamp': datetime.now()
        }
    
    def generate_image_approximation(
        self,
        prompt: str,
        model: str = "hidream",
        width: int = 512,
        height: int = 512,
        steps: int = 10,
        cfg_scale: float = 7.0,
        seed: Optional[int] = None,
        negative_prompt: str = "",
        style_preset: Optional[str] = None,
        safe_mode: bool = True
    ) -> Dict[str, Any]:
        """Generate a quick approximation with reduced parameters"""
        return self.generate_image(
            prompt=prompt,
            model=model,
            width=min(width, 512),
            height=min(height, 512),
            steps=min(steps, 10),
            cfg_scale=cfg_scale,
            seed=seed,
            negative_prompt=negative_prompt,
            style_preset=style_preset,
            safe_mode=safe_mode,
            disable_cache=True
        )

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def inject_custom_css():
    """Inject optimized CSS for better performance and readability"""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main {
            background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
            padding: 1rem;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            color: white !important;
            border: none !important;
            padding: 12px 24px !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 12px rgba(37, 99, 235, 0.3) !important;
        }
        
        .success-alert {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            color: #065f46;
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid #10b981;
            margin: 12px 0;
        }
        
        .error-alert {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            color: #991b1b;
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid #ef4444;
            margin: 12px 0;
        }
        
        .warning-alert {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            color: #92400e;
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid #f59e0b;
            margin: 12px 0;
        }
        
        .info-alert {
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            color: #1e40af;
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            margin: 12px 0;
        }
        
        .memory-warning {
            background-color: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

class SessionState:
    """Enhanced session state management with memory limits"""
    
    MAX_HISTORY_SIZE = 20  # Limit history to prevent memory leaks
    MAX_GENERATED_IMAGES = 100  # Maximum images to keep in memory
    
    @staticmethod
    def initialize():
        """Initialize all session state variables with proper defaults"""
        defaults = {
            'api_key': None,
            'api_validated': False,
            'generated_images': deque(maxlen=SessionState.MAX_GENERATED_IMAGES),
            'current_model': "hidream",
            'image_history': deque(maxlen=SessionState.MAX_HISTORY_SIZE),
            'rate_limit_info': {},
            'last_error': None,
            'image_cache': {},
            'use_approximation': False,
            'use_cache': True,
            'increment_seed': True,
            'increment_mode': "linear",
            'step_size': 1,
            'last_seed': None,
            'seed_manager': SeedManager(max_history=1000),
            'available_models': [],
            'available_styles': [],
            'format': 'png',
            'hide_watermark': False,
            'memory_warning_shown': False,
            'circuit_breaker_status': CircuitBreakerState.CLOSED
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Load API key from environment or secrets
        SessionState._load_api_key()
    
    @staticmethod
    def _load_api_key():
        """Securely load API key from various sources"""
        if st.session_state.api_key is None:
            # Try Streamlit secrets first
            try:
                if hasattr(st, 'secrets') and 'VENICE_API_KEY' in st.secrets:
                    st.session_state.api_key = st.secrets['VENICE_API_KEY']
                    logger.info("API key loaded from Streamlit secrets")
            except Exception as e:
                logger.debug(f"Could not load from secrets: {e}")
            
            # Try environment variable as fallback
            if st.session_state.api_key is None:
                env_key = os.getenv('VENICE_API_KEY')
                if env_key:
                    st.session_state.api_key = env_key
                    logger.info("API key loaded from environment variable")
    
    @staticmethod
    def add_to_history(item: Dict[str, Any]):
        """Add item to history with automatic size management"""
        # Optimize image data before storing
        optimized_item = SessionState._optimize_history_item(item)
        st.session_state.image_history.append(optimized_item)
        
        # Check memory usage
        SessionState._check_memory_usage()
    
    @staticmethod
    def _optimize_history_item(item: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize history item to reduce memory usage"""
        optimized = item.copy()
        
        # Limit stored image data
        if 'images' in optimized and len(optimized['images']) > 0:
            # Store only first image as thumbnail for history
            first_image = optimized['images'][0]
            if first_image.startswith('data:'):
                # Create a reference instead of storing full data
                optimized['images'] = [first_image[:100] + '...[truncated]']
                optimized['full_images_count'] = len(item['images'])
        
        return optimized
    
    @staticmethod
    def _check_memory_usage():
        """Monitor and warn about memory usage"""
        history_size = len(st.session_state.image_history)
        
        if history_size > SessionState.MAX_HISTORY_SIZE * 0.8 and not st.session_state.memory_warning_shown:
            st.warning(f"⚠️ Approaching history limit ({history_size}/{SessionState.MAX_HISTORY_SIZE}). "
                      "Older items will be automatically removed.")
            st.session_state.memory_warning_shown = True
        elif history_size < SessionState.MAX_HISTORY_SIZE * 0.5:
            st.session_state.memory_warning_shown = False
    
    @staticmethod
    def clear_history():
        """Clear history and reset related states"""
        st.session_state.image_history.clear()
        st.session_state.generated_images.clear()
        st.session_state.seed_manager.clear_history()
        st.session_state.memory_warning_shown = False
        logger.info("History cleared")

# ============================================================================
# IMAGE GENERATOR UI WITH ENHANCED API COMPLIANCE
# ============================================================================

class ImageGenerator:
    """Enhanced image generation interface with optimization features"""
    
    def __init__(self, client: VeniceAIClient):
        self.client = client
        
        # Initialize models and styles from API with caching
        if not st.session_state.available_models:
            with st.spinner("Loading available models..."):
                st.session_state.available_models = self.client.get_available_models()
        
        if not st.session_state.available_styles:
            with st.spinner("Loading available styles..."):
                st.session_state.available_styles = self.client.get_available_styles()
        
        self.venice_models = st.session_state.available_models
        self.image_styles = st.session_state.available_styles
    
    def render_api_validation(self):
        """Render API validation status with circuit breaker awareness"""
        if not hasattr(st.session_state, 'api_validated') or not st.session_state.api_validated:
            with st.sidebar:
                # Check circuit breaker status
                if hasattr(self.client, 'circuit_breaker'):
                    breaker_state = self.client.circuit_breaker.state
                    if breaker_state == CircuitBreakerState.OPEN:
                        st.warning("⚠️ API temporarily unavailable (Circuit breaker open)")
                        return False
                    elif breaker_state == CircuitBreakerState.HALF_OPEN:
                        st.info("🔄 Testing API connection...")
                
                with st.spinner("Validating API key..."):
                    is_valid = self.client.validate_api_key()
                    st.session_state.api_validated = is_valid
                    
                    if not is_valid:
                        st.error("❌ Invalid API key. Please check your API key in the sidebar.")
                        return False
                    else:
                        st.success("✅ API key validated successfully!")
        
        return True
    
    def render_sidebar(self):
        """Render the sidebar with enhanced configuration options"""
        with st.sidebar:
            st.markdown("""
                <div style='text-align: center; padding: 20px;'>
                    <h1>🎨 Venice AI</h1>
                    <p style='color: #6b7280;'>Enhanced Image Generator v2.0</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Circuit Breaker Status
            if hasattr(self.client, 'circuit_breaker'):
                breaker_state = self.client.circuit_breaker.state
                state_emoji = {
                    CircuitBreakerState.CLOSED: "🟢",
                    CircuitBreakerState.OPEN: "🔴",
                    CircuitBreakerState.HALF_OPEN: "🟡"
                }
                st.markdown(f"**API Status:** {state_emoji.get(breaker_state, '⚪')} {breaker_state.value.title()}")
            
            st.markdown("### 🔑 API Configuration")
            
            # Secure API key display
            current_key = st.session_state.api_key
            if current_key and len(current_key) > 12:
                masked_key = f"{current_key[:6]}...{current_key[-4:]}"
            else:
                masked_key = "Not Set"
            
            st.markdown(f"**API Key:** `{masked_key}`")
            
            with st.expander("Change API Key", expanded=False):
                new_api_key = st.text_input(
                    "Enter new API key",
                    type="password",
                    placeholder="Enter your Venice AI API key...",
                    key="new_api_key_input",
                    help="Your API key is stored securely in the session"
                )
                
                if st.button("Update Key", key="update_api_key_btn"):
                    if new_api_key and new_api_key.strip():
                        st.session_state.api_key = new_api_key.strip()
                        st.session_state.api_validated = False
                        st.success("✅ API key updated!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")
            
            st.markdown("---")
            st.markdown("### ⚙️ Generation Settings")
            
            # Model selection with description
            model = st.selectbox(
                "Model",
                self.venice_models,
                index=0,
                help="Different models have different artistic styles and capabilities"
            )
            st.session_state.current_model = model
            
            # Dimension presets with aspect ratio display
            dimension_presets = {
                "Square (1024×1024)": (1024, 1024),
                "Portrait (832×1216)": (832, 1216),
                "Landscape (1216×832)": (1216, 832),
                "Mobile (720×1280)": (720, 1280),
                "Desktop (1280×720)": (1280, 720),
                "Custom": (1024, 1024)
            }
            
            dimension_choice = st.selectbox(
                "Dimensions",
                list(dimension_presets.keys()),
                help="Choose from presets or set custom dimensions"
            )
            
            if dimension_choice == "Custom":
                col_w, col_h = st.columns(2)
                with col_w:
                    width = st.number_input(
                        "Width",
                        min_value=256,
                        max_value=1280,
                        value=1024,
                        step=64,
                        help="Width in pixels (256-1280)"
                    )
                with col_h:
                    height = st.number_input(
                        "Height",
                        min_value=256,
                        max_value=1280,
                        value=1024,
                        step=64,
                        help="Height in pixels (256-1280)"
                    )
            else:
                width, height = dimension_presets[dimension_choice]
                st.caption(f"Aspect Ratio: {width/height:.2f}:1")
            
            # Advanced settings
            with st.expander("🎛️ Advanced Settings", expanded=False):
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    steps = st.slider(
                        "Steps",
                        1, 50, 20,
                        help="More steps = higher quality but slower generation"
                    )
                    cfg_scale = st.slider(
                        "CFG Scale",
                        1.0, 20.0, 7.5, 0.5,
                        help="How closely to follow the prompt (higher = stricter)"
                    )
                
                with col_adv2:
                    use_seed = st.checkbox("Use Custom Seed", help="Set a specific seed for reproducible results")
                    seed = st.number_input(
                        "Seed",
                        min_value=-999999999,
                        max_value=999999999,
                        value=42,
                        disabled=not use_seed
                    ) if use_seed else None
                    
                    safe_mode = st.checkbox(
                        "Safe Mode",
                        value=True,
                        help="Blur potentially inappropriate content"
                    )
                
                # Format selection
                format_option = st.selectbox(
                    "Output Format",
                    ["png", "jpeg", "webp"],
                    index=0,
                    help="PNG: Best quality, JPEG: Smaller size, WebP: Modern format"
                )
                st.session_state.format = format_option
                
                # Style preset
                style_preset = st.selectbox(
                    "Style Preset",
                    self.image_styles,
                    help="Apply a predefined artistic style"
                )
                
                # Negative prompt with character counter
                negative_prompt = st.text_area(
                    "Negative Prompt (Optional)",
                    height=60,
                    max_chars=1500,
                    placeholder="blurry, low quality, distorted, watermark, text",
                    help="Describe what you DON'T want in the image"
                )
                
                if negative_prompt:
                    st.caption(f"Characters: {len(negative_prompt)}/1500")
            
            # Seed incrementation options
            st.markdown("---")
            st.markdown("#### 🌱 Seed Variation Options")
            
            increment_seed = st.checkbox(
                "Auto-vary Seeds",
                value=st.session_state.increment_seed,
                help="Automatically modify seed for each image to create variations",
                disabled=not use_seed
            )
            st.session_state.increment_seed = increment_seed
            
            if increment_seed and use_seed:
                increment_mode = st.selectbox(
                    "Variation Pattern",
                    ["linear", "geometric", "alternating", "random_walk", "fibonacci"],
                    index=0,
                    help="How to vary the seed between images"
                )
                st.session_state.increment_mode = increment_mode
                
                step_size = st.slider(
                    "Variation Strength",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.step_size,
                    help="How much to change the seed between images"
                )
                st.session_state.step_size = step_size
            
            # Optimization settings
            st.markdown("---")
            st.markdown("#### 🚀 Performance Options")
            
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                use_approximation = st.checkbox(
                    "Quick Preview",
                    value=st.session_state.use_approximation,
                    help="Generate low-res preview for faster iteration"
                )
                st.session_state.use_approximation = use_approximation
            
            with col_opt2:
                use_cache = st.checkbox(
                    "Enable Cache",
                    value=st.session_state.use_cache,
                    help="Cache results to avoid redundant API calls"
                )
                st.session_state.use_cache = use_cache
            
            # Cache and memory statistics
            st.markdown("---")
            st.markdown("### 📊 System Status")
            
            # Cache stats
            cache_stats = self.client.memoizer.get_cache_stats()
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Cache Entries", cache_stats['total_entries'])
                st.metric("Valid Hits", cache_stats['valid_entries'])
            with col_stat2:
                st.metric("Avg Quality", f"{cache_stats['average_quality']:.2f}")
                st.metric("Memory (MB)", f"{cache_stats['memory_usage_mb']:.1f}")
            
            if cache_stats.get('most_accessed'):
                st.caption(f"Most used: {cache_stats['most_accessed']}...")
            
            # Seed statistics
            seed_stats = st.session_state.seed_manager.get_seed_statistics()
            if seed_stats:
                st.markdown("#### 🌱 Seed History")
                col_seed1, col_seed2 = st.columns(2)
                with col_seed1:
                    st.metric("Total Seeds", seed_stats['total_generated'])
                with col_seed2:
                    st.metric("Unique Seeds", seed_stats['unique_seeds'])
                
                if seed_stats.get('recent_seeds'):
                    recent = seed_stats['recent_seeds'][:5]
                    st.caption(f"Recent: {', '.join(map(str, recent))}")
            
            # History size indicator
            history_size = len(st.session_state.image_history)
            history_percentage = (history_size / SessionState.MAX_HISTORY_SIZE) * 100
            
            st.markdown("#### 📚 History Status")
            st.progress(history_percentage / 100)
            st.caption(f"{history_size}/{SessionState.MAX_HISTORY_SIZE} items stored")
            
            # Clear cache and history buttons
            col_clear1, col_clear2 = st.columns(2)
            
            with col_clear1:
                if st.button("🗑️ Clear Cache", use_container_width=True):
                    self.client.memoizer.clear_cache()
                    st.success("Cache cleared!")
                    st.rerun()
            
            with col_clear2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    SessionState.clear_history()
                    st.success("History cleared!")
                    st.rerun()
            
            # Rate limits and balance
            st.markdown("---")
            st.markdown("### 💰 Account Balance")
            
            if st.button("🔄 Refresh Balance", key="refresh_balance_btn", use_container_width=True):
                with st.spinner("Fetching balance..."):
                    rate_limits = self.client.get_rate_limits()
                    st.session_state.rate_limit_info = rate_limits
            
            if st.session_state.rate_limit_info:
                balances = st.session_state.rate_limit_info.get('balances', {})
                if balances:
                    for currency, amount in balances.items():
                        if currency == "USD":
                            st.metric(f"{currency} Balance", f"${amount:.2f}")
                        else:
                            st.metric(f"{currency} Balance", f"{amount:.2f}")
            
            return {
                'model': model,
                'width': width,
                'height': height,
                'steps': steps,
                'cfg_scale': cfg_scale,
                'seed': seed,
                'safe_mode': safe_mode,
                'style_preset': style_preset,
                'negative_prompt': negative_prompt,
                'format': format_option,
                'hide_watermark': st.session_state.hide_watermark
            }
    
    def render_main_interface(self, params: Dict[str, Any]):
        """Render the main image generation interface"""
        st.markdown("# 🎨 Venice AI Image Generator")
        st.markdown("*Enhanced with Circuit Breaker, Optimized Caching, and Memory Management*")
        
        # Check for warnings
        if hasattr(self.client, 'circuit_breaker'):
            if self.client.circuit_breaker.state == CircuitBreakerState.OPEN:
                st.error("⚠️ API is temporarily unavailable. Please wait before retrying.")
            elif self.client.circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                st.warning("🔄 API is recovering. Limited functionality available.")
        
        # Prompt input with enhanced UI
        col1, col2 = st.columns([3, 1])
        
        with col1:
            prompt = st.text_area(
                "Image Prompt",
                height=100,
                max_chars=1500,
                placeholder="A serene Japanese garden with cherry blossoms, koi pond reflecting the sunset, ultra realistic, 8k, masterpiece",
                help="Describe your image in detail. Be specific about style, lighting, composition, and quality."
            )
            
            # Character counter with color coding
            char_count = len(prompt) if prompt else 0
            if char_count > 1400:
                st.error(f"⚠️ Characters: {char_count}/1500 - Approaching limit!")
            elif char_count > 1200:
                st.warning(f"Characters: {char_count}/1500")
            else:
                st.caption(f"Characters: {char_count}/1500")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            num_images = st.selectbox(
                "Number of Images",
                [1, 2, 3, 4],
                index=0,
                help="Generate multiple variations at once"
            )
            
            # Estimate generation time
            estimated_time = num_images * (params['steps'] * 0.5 + 5)
            st.caption(f"Est. time: ~{estimated_time:.0f}s")
        
        # Generate button with disabled state handling
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            generate_btn = st.button(
                "🎨 Generate Image" if not st.session_state.use_approximation else "🚀 Quick Preview",
                type="primary",
                disabled=not prompt or not st.session_state.api_validated,
                use_container_width=True
            )
        
        with col_btn2:
            if st.button("🎲 Random Prompt", use_container_width=True):
                # Generate a random creative prompt
                random_prompts = [
                    "A cyberpunk city at night with neon lights reflecting in puddles, flying cars",
                    "A magical forest with bioluminescent plants and ethereal creatures",
                    "A steampunk laboratory with brass machinery and glowing experiments",
                    "An underwater palace made of coral and pearls, mermaids swimming",
                    "A floating island in the sky with waterfalls cascading into clouds"
                ]
                st.session_state.random_prompt = random.choice(random_prompts)
                st.rerun()
        
        # Display random prompt if generated
        if hasattr(st.session_state, 'random_prompt'):
            st.info(f"💡 Suggested: {st.session_state.random_prompt}")
        
        # Display generation results
        if generate_btn and prompt:
            self._generate_images(prompt, num_images, params)
        
        # Display image history
        self._display_image_history()
    
    def _generate_images(self, prompt: str, num_images: int, params: Dict[str, Any]):
        """Generate images with enhanced progress tracking and error handling"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        generated_images = []
        failed_generations = []
        generation_times = []
        
        # Determine seeds
        base_seed = params['seed']
        
        if st.session_state.increment_seed and base_seed is not None:
            seeds = st.session_state.seed_manager.generate_seeds(
                base_seed,
                num_images,
                st.session_state.increment_mode,
                st.session_state.step_size
            )
        else:
            seeds = [base_seed] * num_images
        
        # Generation loop
        for i in range(num_images):
            progress_bar.progress((i) / num_images)
            
            if st.session_state.use_approximation:
                status_text.text(f"🚀 Generating quick preview {i+1}/{num_images}...")
            else:
                status_text.text(f"🎨 Generating image {i+1}/{num_images}...")
            
            start_time = time.time()
            
            try:
                current_seed = seeds[i]
                
                # Generate image
                if st.session_state.use_approximation:
                    result = self.client.generate_image_approximation(
                        prompt=prompt,
                        model=params['model'],
                        width=params['width'] // 2,
                        height=params['height'] // 2,
                        steps=10,
                        cfg_scale=params['cfg_scale'] - 1,
                        seed=current_seed,
                        negative_prompt=params['negative_prompt'],
                        style_preset=params['style_preset'],
                        safe_mode=params['safe_mode']
                    )
                else:
                    result = self.client.generate_image(
                        prompt=prompt,
                        model=params['model'],
                        width=params['width'],
                        height=params['height'],
                        steps=params['steps'],
                        cfg_scale=params['cfg_scale'],
                        seed=current_seed,
                        negative_prompt=params['negative_prompt'],
                        style_preset=params['style_preset'],
                        safe_mode=params['safe_mode'],
                        format=params['format'],
                        disable_cache=not st.session_state.use_cache
                    )
                
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                if result['success']:
                    generated_images.extend(result['images'])
                    
                    # Add to history with optimization
                    history_item = {
                        'prompt': prompt,
                        'negative_prompt': params['negative_prompt'],
                        'model': params['model'],
                        'images': result['images'],
                        'timestamp': result['timestamp'],
                        'settings': {
                            'width': params['width'],
                            'height': params['height'],
                            'steps': params['steps'],
                            'cfg_scale': params['cfg_scale'],
                            'seed': current_seed,
                            'style_preset': params['style_preset'],
                            'safe_mode': params['safe_mode'],
                            'format': params['format'],
                            'approximation': st.session_state.use_approximation,
                            'increment_mode': st.session_state.increment_mode if st.session_state.increment_seed else None,
                            'step_size': st.session_state.step_size if st.session_state.increment_seed else None
                        },
                        'timing': result.get('timing', {'total': generation_time}),
                        'is_blurred': result.get('is_blurred', False),
                        'is_violation': result.get('is_violation', False)
                    }
                    
                    SessionState.add_to_history(history_item)
                    st.session_state.last_seed = current_seed
                    
                else:
                    error_type = result.get('error_type', 'Unknown')
                    error_msg = result.get('error', 'Generation failed')
                    failed_generations.append(f"Image {i+1} ({error_type}): {error_msg}")
                    st.session_state.last_error = error_msg
                    logger.error(f"Generation {i+1} failed: {error_msg}")
                    
            except Exception as e:
                generation_time = time.time() - start_time
                error_msg = str(e)
                failed_generations.append(f"Image {i+1}: {error_msg}")
                st.session_state.last_error = error_msg
                logger.error(f"Generation {i+1} failed: {error_msg}", exc_info=True)
        
        progress_bar.progress(1.0)
        
        # Display results summary
        if generated_images:
            avg_time = sum(generation_times) / len(generation_times) if generation_times else 0
            status_text.success(
                f"✅ Generated {len(generated_images)} image(s) successfully! "
                f"(Avg time: {avg_time:.1f}s)"
            )
            
            # Display seed information
            if base_seed is not None and st.session_state.increment_seed and num_images > 1:
                seed_info = f"🌱 Seeds used: {seeds[0]} to {seeds[-1]}"
                if st.session_state.increment_mode != "linear":
                    seed_info += f" (Pattern: {st.session_state.increment_mode}, Step: {st.session_state.step_size})"
                st.info(seed_info)
        else:
            status_text.error("❌ All image generations failed")
        
        # Display any failures
        if failed_generations:
            with st.expander(f"⚠️ Generation Errors ({len(failed_generations)})", expanded=len(generated_images) == 0):
                for failure in failed_generations:
                    st.error(failure)
                
                # Provide helpful suggestions based on error types
                if any("RateLimitError" in f for f in failed_generations):
                    st.info("💡 Tip: Wait a moment before retrying or reduce the number of images")
                if any("ValidationError" in f for f in failed_generations):
                    st.info("💡 Tip: Check your prompt and parameters are within allowed limits")
                if any("AuthenticationError" in f for f in failed_generations):
                    st.info("💡 Tip: Verify your API key is correct and has sufficient permissions")
        
        # Display generated images
        if generated_images:
            with results_container:
                st.markdown("### 🖼️ Generated Images")
                
                # Create columns for image display
                cols = st.columns(min(len(generated_images), 4))
                
                for idx, image_data in enumerate(generated_images):
                    col_idx = idx % len(cols)
                    with cols[col_idx]:
                        # Get seed and metadata
                        seed_used = seeds[idx] if idx < len(seeds) else "Unknown"
                        generation_time = generation_times[idx] if idx < len(generation_times) else 0
                        
                        # Check for content warnings
                        if st.session_state.image_history:
                            latest_history = st.session_state.image_history[-1]
                            is_blurred = latest_history.get('is_blurred', False)
                            is_violation = latest_history.get('is_violation', False)
                        else:
                            is_blurred = False
                            is_violation = False
                        
                        # Build caption
                        caption_parts = [f"Image {idx+1}"]
                        if seed_used != "Unknown":
                            caption_parts.append(f"Seed: {seed_used}")
                        if generation_time > 0:
                            caption_parts.append(f"{generation_time:.1f}s")
                        
                        caption = " | ".join(caption_parts)
                        
                        # Add warning badges
                        if is_blurred:
                            caption += " 🔞"
                        if is_violation:
                            caption += " ⚠️"
                        
                        # Display image
                        st.image(image_data, caption=caption, use_column_width=True)
                        
                        # Download button
                        if image_data.startswith('data:'):
                            try:
                                # Extract base64 data
                                if image_data.startswith('data:image/'):
                                    format_str = image_data.split(';')[0].split('/')[1]
                                    base64_data = image_data.split(',')[1]
                                else:
                                    format_str = params['format']
                                    base64_data = image_data
                                
                                image_bytes = base64.b64decode(base64_data)
                                
                                # Optimize image if needed
                                if len(image_bytes) > 5 * 1024 * 1024:  # If larger than 5MB
                                    try:
                                        img = Image.open(io.BytesIO(image_bytes))
                                        optimized_buffer = io.BytesIO()
                                        img.save(optimized_buffer, format=format_str.upper(), optimize=True, quality=85)
                                        image_bytes = optimized_buffer.getvalue()
                                        st.caption("📦 Image optimized for download")
                                    except Exception as e:
                                        logger.warning(f"Could not optimize image: {e}")
                                
                                st.download_button(
                                    f"⬇️ Download",
                                    data=image_bytes,
                                    file_name=f"venice_ai_{int(time.time())}_{idx}.{format_str}",
                                    mime=f"image/{format_str}",
                                    key=f"download_{idx}_{time.time()}"
                                )
                            except Exception as e:
                                st.error(f"Download error: {str(e)}")
                                logger.error(f"Download preparation failed: {e}")
                
                # Display warnings if present
                if any(h.get('is_blurred', False) for h in [st.session_state.image_history[-1]] if st.session_state.image_history):
                    st.warning("🔞 Some content was blurred due to safety filters")
                if any(h.get('is_violation', False) for h in [st.session_state.image_history[-1]] if st.session_state.image_history):
                    st.error("⚠️ Some content violated policy guidelines")
        
        # Clean up UI elements
        progress_bar.empty()
        status_text.empty()
    
    def _display_image_history(self):
        """Display image generation history with memory management"""
        if not st.session_state.image_history:
            return
        
        st.markdown("---")
        st.markdown("### 📚 Generation History")
        
        # History controls
        col_hist1, col_hist2, col_hist3 = st.columns([1, 1, 2])
        
        with col_hist1:
            if st.button("🗑️ Clear History", key="clear_history_btn"):
                SessionState.clear_history()
                st.success("History cleared!")
                st.rerun()
        
        with col_hist2:
            history_count = len(st.session_state.image_history)
            st.metric("Total", f"{history_count} items")
        
        with col_hist3:
            # Memory usage indicator
            memory_usage = history_count / SessionState.MAX_HISTORY_SIZE * 100
            st.progress(memory_usage / 100)
            st.caption(f"Memory: {memory_usage:.0f}% used")
        
        # Display history items (newest first)
        history_items = list(reversed(st.session_state.image_history))
        
        # Pagination for history
        items_per_page = 5
        if 'history_page' not in st.session_state:
            st.session_state.history_page = 0
        
        total_pages = (len(history_items) - 1) // items_per_page + 1
        
        # Page navigation
        if total_pages > 1:
            col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
            with col_nav1:
                if st.button("◀ Previous", disabled=st.session_state.history_page == 0):
                    st.session_state.history_page -= 1
                    st.rerun()
            with col_nav2:
                st.markdown(f"<center>Page {st.session_state.history_page + 1} of {total_pages}</center>", unsafe_allow_html=True)
            with col_nav3:
                if st.button("Next ▶", disabled=st.session_state.history_page >= total_pages - 1):
                    st.session_state.history_page += 1
                    st.rerun()
        
        # Display current page items
        start_idx = st.session_state.history_page * items_per_page
        end_idx = min(start_idx + items_per_page, len(history_items))
        
        for idx, item in enumerate(history_items[start_idx:end_idx], start=start_idx):
            # Format timing information
            timing_info = item.get('timing', {})
            timing_text = f" | ⏱️ {timing_info.get('total', 0):.1f}s" if timing_info.get('total') else ""
            
            # Build expander title with warnings
            warning_icons = ""
            if item.get('is_blurred', False):
                warning_icons += " 🔞"
            if item.get('is_violation', False):
                warning_icons += " ⚠️"
            
            # Truncate prompt for display
            display_prompt = item['prompt'][:60] + '...' if len(item['prompt']) > 60 else item['prompt']
            
            with st.expander(
                f"🎨 {display_prompt} | {item['model']} | "
                f"{item['timestamp'].strftime('%H:%M:%S')}{timing_text}{warning_icons}",
                expanded=False
            ):
                # Check if we have actual image data
                if item.get('images') and item['images'][0] and not item['images'][0].endswith('[truncated]'):
                    # Display images
                    image_count = item.get('full_images_count', len(item['images']))
                    st.info(f"Generated {image_count} image(s)")
                    
                    # Note about truncated storage
                    if item['images'][0].endswith('[truncated]'):
                        st.warning("⚠️ Full image data not available in history (memory optimization)")
                else:
                    st.info(f"Generated {item.get('full_images_count', 1)} image(s) - Data optimized for memory")
                
                # Display generation details in a structured format
                col_detail1, col_detail2, col_detail3 = st.columns(3)
                
                with col_detail1:
                    st.markdown("**Model Settings:**")
                    st.text(f"Model: {item['model']}")
                    st.text(f"Dimensions: {item['settings']['width']}×{item['settings']['height']}")
                    st.text(f"Format: {item['settings'].get('format', 'png')}")
                
                with col_detail2:
                    st.markdown("**Generation Parameters:**")
                    st.text(f"Steps: {item['settings']['steps']}")
                    st.text(f"CFG Scale: {item['settings']['cfg_scale']}")
                    st.text(f"Safe Mode: {'✅' if item['settings']['safe_mode'] else '❌'}")
                
                with col_detail3:
                    st.markdown("**Seed Information:**")
                    if item['settings']['seed'] is not None:
                        st.text(f"Seed: {item['settings']['seed']}")
                        if item['settings'].get('increment_mode'):
                            st.text(f"Pattern: {item['settings']['increment_mode']}")
                            st.text(f"Step: {item['settings'].get('step_size', 1)}")
                    else:
                        st.text("Seed: Random")
                
                # Display prompts
                st.markdown("**Prompt:**")
                st.code(item['prompt'], language=None)
                
                if item.get('negative_prompt'):
                    st.markdown("**Negative Prompt:**")
                    st.code(item['negative_prompt'], language=None)
                
                # Display style if used
                if item['settings'].get('style_preset'):
                    st.markdown(f"**Style:** {item['settings']['style_preset']}")
                
                # Display warnings
                if item.get('is_blurred', False):
                    st.warning("🔞 This image was blurred due to content safety filters")
                if item.get('is_violation', False):
                    st.error("⚠️ This image violated content policy guidelines")
                
                # Copy prompt button
                if st.button(f"📋 Copy Prompt", key=f"copy_{idx}_{item['timestamp']}"):
                    st.code(item['prompt'], language=None)
                    st.success("Prompt displayed above - copy it manually!")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def handle_error_recovery():
    """Provide error recovery options"""
    st.markdown("### 🔧 Recovery Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset Application", key="reset_app_btn", use_container_width=True):
            # Preserve API key but reset everything else
            api_key = st.session_state.get('api_key')
            for key in list(st.session_state.keys()):
                if key != 'api_key':
                    del st.session_state[key]
            if api_key:
                st.session_state.api_key = api_key
            st.success("Application reset!")
            st.rerun()
    
    with col2:
        if st.button("🔑 Reset API Key", key="reset_api_btn", use_container_width=True):
            st.session_state.api_key = None
            st.session_state.api_validated = False
            st.success("API key reset!")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All Data", key="clear_all_btn", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("All data cleared!")
            st.rerun()

def main():
    """Main application entry point with error handling"""
    try:
        # Inject custom CSS
        inject_custom_css()
        
        # Initialize session state
        SessionState.initialize()
        
        # Check if API key is set
        if not st.session_state.api_key:
            st.error("⚠️ No API key found!")
            st.markdown("""
                ### 🔑 Setting up your API Key
                
                Please set your Venice AI API key using one of these methods:
                
                1. **Enter it in the sidebar** (Recommended for testing)
                2. **Set environment variable** `VENICE_API_KEY` (Recommended for deployment)
                3. **Add to Streamlit secrets** (For Streamlit Cloud deployment)
                
                Get your API key from [Venice AI](https://venice.ai)
            """)
            
            # Provide manual input option
            with st.sidebar:
                st.markdown("### 🔑 Quick Setup")
                manual_key = st.text_input(
                    "Enter API Key",
                    type="password",
                    placeholder="Your Venice AI API key",
                    help="This will be stored only for this session"
                )
                if st.button("Set API Key", type="primary"):
                    if manual_key and manual_key.strip():
                        st.session_state.api_key = manual_key.strip()
                        st.success("API key set!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key")
            return
        
        # Initialize Venice AI client
        settings = VeniceSettings()
        client = VeniceAIClient(st.session_state.api_key, settings)
        
        # Initialize image generator
        image_generator = ImageGenerator(client)
        
        # Validate API key
        if not image_generator.render_api_validation():
            st.error("❌ API validation failed. Please check your API key.")
            handle_error_recovery()
            return
        
        # Render sidebar and get parameters
        params = image_generator.render_sidebar()
        
        # Render main interface
        image_generator.render_main_interface(params)
        
    except AuthenticationError as e:
        st.error(f"🔐 Authentication Error: {e.message}")
        st.info("Please check your API key and ensure it has the correct permissions")
        handle_error_recovery()
        
    except InsufficientBalanceError as e:
        st.error(f"💰 Insufficient Balance: {e.message}")
        st.info("Please add credits to your Venice AI account")
        
    except RateLimitError as e:
        st.error(f"⏱️ Rate Limit Exceeded: {e.message}")
        if e.details.get('retry_after'):
            st.info(f"Please wait {e.details['retry_after']} seconds before retrying")
        
    except ServiceUnavailableError as e:
        st.error(f"🔧 Service Unavailable: {e.message}")
        st.info("The Venice AI service is temporarily unavailable. Please try again later.")
        handle_error_recovery()
        
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        
        # Show debug information in expander
        with st.expander("🐛 Debug Information"):
            st.code(f"""
Error Type: {type(e).__name__}
Error Message: {str(e)}
Timestamp: {datetime.now().isoformat()}
Session State Keys: {list(st.session_state.keys())}
            """)
        
        handle_error_recovery()

if __name__ == "__main__":
    main()