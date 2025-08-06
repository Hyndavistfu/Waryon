"""
WARYON Enhanced AI Engine with TRUE Gemma 3n Mix'n'Match
Real multiple model implementation for competition
"""

import requests
import json
import time
import base64
import io
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Literal
import numpy as np
import cv2
import threading

PerformanceLevel = Literal["2b_efficient", "4b_full", "3b_balanced", "auto", "mixnmatch"]

class GemmaMixNMatchAI:
    def __init__(self, config_manager):
        self.config = config_manager
        self.ollama_url = "http://localhost:11434"
        
        # TRUE Mix'n'Match - Multiple Gemma models
        self.models = {
            "quick_audio": "gemma3n:2b",      # Fast audio-only analysis
            "multimodal_full": "gemma3n:e4b", # Full multimodal analysis
            "text_analysis": "gemma3n:2b",    # Quick text processing
            "primary": "gemma3n:e4b"          # Main model
        }
        
        # Performance monitoring for each model
        self.model_stats = {
            "gemma3n:2b": {"uses": 0, "avg_time": 0.0, "total_time": 0.0},
            "gemma3n:e4b": {"uses": 0, "avg_time": 0.0, "total_time": 0.0}
        }
        
        # Mix'n'Match decision thresholds
        self.mixnmatch_thresholds = {
            "audio_urgency_for_quick": 0.8,      # Use quick 2B for urgent audio
            "escalation_confidence": 0.7,        # Escalate to 4B if confidence low
            "complex_scene_threshold": 0.6       # Use 4B for complex scenes
        }
        
        # Enhanced threat detection prompts for different models
        self.prompts = self._initialize_mixnmatch_prompts()
        
        # Test all model connections
        self.connection_status = self.test_all_models()
        
        print(f"🤖 Gemma 3n Mix'n'Match AI Engine initialized")
        print(f"   Models Available: {list(self.models.keys())}")
        print(f"   Connection Status: {'✅ All Ready' if self.connection_status else '❌ Some Failed'}")
    
    def _initialize_mixnmatch_prompts(self) -> Dict[str, str]:
        """Initialize specialized prompts for different models"""
        return {
            'quick_audio_screening': """You are WARYON's rapid audio threat detector using Gemma 3n 2B.

Analyze this audio description for IMMEDIATE threats requiring escalation:

AUDIO: {audio_description}

Look for URGENT keywords: help, emergency, fire, police, attack, hurt, stop, don't, scared

Respond in this EXACT format:
URGENT: [YES/NO]
CONFIDENCE: [0.0-1.0]
KEYWORDS: [List any urgent words found]
ESCALATE: [YES/NO - should this go to full multimodal analysis?]""",

            'multimodal_full_analysis': """You are WARYON's comprehensive threat analyzer using Gemma 3n 4B multimodal.

This case was escalated from quick screening. Perform COMPLETE analysis:

VISUAL CONTEXT: {visual_analysis}
AUDIO CONTEXT: {audio_analysis}
PREVIOUS SCREENING: {screening_result}

Analyze BOTH visual and audio for:
- Violence, medical emergencies, distress
- Context correlation between visual and audio
- Threat confirmation or false positive identification

THREAT: [YES/NO]
CONFIDENCE: [0.0-1.0]
TYPE: [Violence/Emergency Call/Medical/Distress/False Positive]
MULTIMODAL_CORRELATION: [How visual and audio relate]
REASONING: [Detailed multimodal analysis]
URGENCY: [Low/Medium/High/Critical]""",

            'fallback_analysis': """You are WARYON using standard Gemma 3n analysis.

VISUAL_CONTEXT: {visual_analysis}
AUDIO_CONTEXT: {audio_analysis}

Standard threat detection analysis:

THREAT: [YES/NO]
CONFIDENCE: [0.0-1.0]
TYPE: [Violence/Fall/Normal Activity]
REASONING: [Analysis explanation]
URGENCY: [Low/Medium/High]"""
        }
    
    def test_all_models(self) -> bool:
        """Test connection to all Gemma models"""
        try:
            print("🧪 Testing all Gemma 3n models...")
            
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            available_models = []
            missing_models = []
            
            for purpose, model_name in self.models.items():
                if model_name in model_names:
                    available_models.append(f"{purpose}: {model_name}")
                else:
                    missing_models.append(f"{purpose}: {model_name}")
            
            print(f"✅ Available Models:")
            for model in available_models:
                print(f"   {model}")
            
            if missing_models:
                print(f"❌ Missing Models:")
                for model in missing_models:
                    print(f"   {model}")
                print("   Download with: ollama pull gemma3n:2b")
            
            # Test basic queries on available models
            for model_name in set(self.models.values()):
                if model_name in model_names:
                    test_response = self._test_model_query(model_name)
                    if not test_response:
                        print(f"❌ Model {model_name} failed test query")
                        return False
            
            return len(missing_models) == 0
            
        except Exception as e:
            print(f"❌ Model connection test failed: {e}")
            return False
    
    def _test_model_query(self, model_name: str) -> bool:
        """Test individual model with simple query"""
        try:
            query_data = {
                'model': model_name,
                'prompt': 'Test query. Respond with: WARYON READY',
                'stream': False,
                'options': {'num_predict': 10}
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=query_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                return 'ready' in response_text.lower() or 'waryon' in response_text.lower()
            
            return False
            
        except Exception as e:
            print(f"❌ Model test error for {model_name}: {e}")
            return False
    
    def analyze_multimodal_threat(self, image, audio_description: str = "", performance_level: PerformanceLevel = "mixnmatch") -> Dict[str, Any]:
        """Enhanced Mix'n'Match multimodal threat analysis"""
        start_time = time.time()
        
        try:
            if performance_level == "mixnmatch":
                return self._mixnmatch_analysis(image, audio_description, start_time)
            else:
                # Fallback to single model analysis
                return self._single_model_analysis(image, audio_description, performance_level, start_time)
                
        except Exception as e:
            print(f"❌ Mix'n'Match analysis error: {e}")
            return self._create_error_response(f"Mix'n'Match analysis error: {e}")
    
    def _mixnmatch_analysis(self, image, audio_description: str, start_time: float) -> Dict[str, Any]:
        """TRUE Mix'n'Match analysis using multiple Gemma models"""
        print(f"🔀 Starting Mix'n'Match analysis...")
        
        analysis_steps = []
        
        # STEP 1: Quick Audio Screening with 2B model
        audio_screening = None
        if audio_description:
            print(f"🎤 Step 1: Quick audio screening with Gemma 2B...")
            audio_screening = self._quick_audio_screening(audio_description)
            analysis_steps.append(f"Audio screening: {audio_screening.get('escalate', False)}")
            
            # If urgent audio detected, escalate immediately
            if audio_screening.get('escalate', False):
                print(f"⚡ URGENT AUDIO DETECTED - Escalating to full multimodal analysis!")
                return self._full_multimodal_analysis(image, audio_description, audio_screening, start_time)
        
        # STEP 2: Scene Complexity Assessment
        print(f"🔍 Step 2: Assessing scene complexity...")
        scene_complexity = self._assess_scene_complexity(image)
        analysis_steps.append(f"Scene complexity: {scene_complexity:.2f}")
        
        # STEP 3: Model Selection Based on Complexity
        if scene_complexity > self.mixnmatch_thresholds["complex_scene_threshold"]:
            print(f"🧠 Complex scene detected - Using Gemma 4B for full analysis...")
            return self._full_multimodal_analysis(image, audio_description, audio_screening, start_time)
        else:
            print(f"📊 Normal scene - Using efficient analysis...")
            return self._efficient_analysis(image, audio_description, audio_screening, start_time)
    
    def _quick_audio_screening(self, audio_description: str) -> Dict[str, Any]:
        """Quick audio screening using Gemma 2B"""
        screening_start = time.time()
        
        try:
            # Use quick audio model
            prompt = self.prompts['quick_audio_screening'].format(audio_description=audio_description)
            
            response = self._query_specific_model(
                prompt=prompt,
                model_name=self.models['quick_audio'],
                max_tokens=100
            )
            
            screening_time = time.time() - screening_start
            self._update_model_stats(self.models['quick_audio'], screening_time)
            
            # Parse screening result
            result = self._parse_screening_response(response)
            result['screening_time'] = screening_time
            result['model_used'] = self.models['quick_audio']
            
            print(f"🎤 Audio screening complete: {screening_time:.2f}s - Escalate: {result.get('escalate', False)}")
            
            return result
            
        except Exception as e:
            print(f"❌ Quick audio screening error: {e}")
            return {'escalate': False, 'error': str(e)}
    
    def _full_multimodal_analysis(self, image, audio_description: str, screening_result: Optional[Dict], start_time: float) -> Dict[str, Any]:
        """Full multimodal analysis using Gemma 4B"""
        multimodal_start = time.time()
        
        try:
            print(f"🧠 Full multimodal analysis with Gemma 4B...")
            
            # Prepare comprehensive visual analysis
            visual_analysis = self._analyze_visual_content(image)
            audio_analysis = self._analyze_audio_content(audio_description)
            
            # Create enhanced multimodal prompt
            prompt = self.prompts['multimodal_full_analysis'].format(
                visual_analysis=visual_analysis,
                audio_analysis=audio_analysis,
                screening_result=screening_result or "No prior screening"
            )
            
            # Query full multimodal model
            response = self._query_model_with_image(
                prompt=prompt,
                image=image,
                model_name=self.models['multimodal_full'],
                max_tokens=500
            )
            
            multimodal_time = time.time() - multimodal_start
            total_time = time.time() - start_time
            
            self._update_model_stats(self.models['multimodal_full'], multimodal_time)
            
            # Parse comprehensive result
            result = self._parse_multimodal_response(response)
            result.update({
                'mixnmatch_analysis': True,
                'screening_result': screening_result,
                'multimodal_time': multimodal_time,
                'total_analysis_time': total_time,
                'models_used': [
                    self.models['quick_audio'] if screening_result else None,
                    self.models['multimodal_full']
                ],
                'analysis_path': 'quick_screening -> full_multimodal' if screening_result else 'direct_multimodal'
            })
            
            print(f"🧠 Full analysis complete: {multimodal_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Full multimodal analysis error: {e}")
            return self._create_error_response(f"Full multimodal analysis error: {e}")
    
    def _efficient_analysis(self, image, audio_description: str, screening_result: Optional[Dict], start_time: float) -> Dict[str, Any]:
        """Efficient analysis for normal scenes"""
        efficient_start = time.time()
        
        try:
            print(f"📊 Efficient analysis for normal scene...")
            
            # Use primary model with efficient settings
            visual_analysis = self._analyze_visual_content(image)
            audio_analysis = self._analyze_audio_content(audio_description)
            
            prompt = self.prompts['fallback_analysis'].format(
                visual_analysis=visual_analysis,
                audio_analysis=audio_analysis
            )
            
            response = self._query_model_with_image(
                prompt=prompt,
                image=image,
                model_name=self.models['primary'],
                max_tokens=300
            )
            
            efficient_time = time.time() - efficient_start
            total_time = time.time() - start_time
            
            self._update_model_stats(self.models['primary'], efficient_time)
            
            result = self._parse_threat_response(response)
            result.update({
                'mixnmatch_analysis': True,
                'screening_result': screening_result,
                'efficient_time': efficient_time,
                'total_analysis_time': total_time,
                'models_used': [self.models['primary']],
                'analysis_path': 'efficient_normal_scene'
            })
            
            print(f"📊 Efficient analysis complete: {efficient_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Efficient analysis error: {e}")
            return self._create_error_response(f"Efficient analysis error: {e}")
    
    def _single_model_analysis(self, image, audio_description: str, performance_level: PerformanceLevel, start_time: float) -> Dict[str, Any]:
        """Fallback single model analysis"""
        try:
            # Use primary model with specified performance settings
            visual_analysis = self._analyze_visual_content(image)
            audio_analysis = self._analyze_audio_content(audio_description)
            
            combined_prompt = self._create_multimodal_prompt(visual_analysis, audio_analysis, audio_description)
            
            response = self._query_model_with_image(
                prompt=combined_prompt,
                image=image,
                model_name=self.models['primary'],
                performance_level=performance_level
            )
            
            analysis_time = time.time() - start_time
            
            result = self._parse_threat_response(response)
            result.update({
                'mixnmatch_analysis': False,
                'analysis_time': analysis_time,
                'model_used': self.models['primary'],
                'performance_level': performance_level
            })
            
            return result
            
        except Exception as e:
            return self._create_error_response(f"Single model analysis error: {e}")
    
    def _query_specific_model(self, prompt: str, model_name: str, max_tokens: int = 200) -> Optional[str]:
        """Query a specific Gemma model"""
        try:
            query_data = {
                'model': model_name,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'num_predict': max_tokens,
                    'temperature': 0.1,
                    'top_p': 0.9
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=query_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
            return None
            
        except Exception as e:
            print(f"❌ Specific model query error for {model_name}: {e}")
            return None
    
    def _query_model_with_image(self, prompt: str, image, model_name: str, performance_level: str = None, max_tokens: int = 500) -> Optional[str]:
        """Query Gemma model with image"""
        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(image)
            
            # Get performance settings
            if performance_level:
                options = self._get_performance_settings(performance_level)
            else:
                options = {
                    'num_predict': max_tokens,
                    'temperature': 0.1,
                    'top_p': 0.95
                }
            
            query_data = {
                'model': model_name,
                'prompt': prompt,
                'images': [image_b64],
                'stream': False,
                'options': options
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=query_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
            return None
            
        except Exception as e:
            print(f"❌ Model with image query error for {model_name}: {e}")
            return None
    
    def _update_model_stats(self, model_name: str, execution_time: float):
        """Update performance statistics for each model"""
        if model_name in self.model_stats:
            stats = self.model_stats[model_name]
            stats['uses'] += 1
            stats['total_time'] += execution_time
            stats['avg_time'] = stats['total_time'] / stats['uses']
    
    def _parse_screening_response(self, response: str) -> Dict[str, Any]:
        """Parse quick audio screening response"""
        result = {
            'urgent': False,
            'confidence': 0.0,
            'keywords': [],
            'escalate': False
        }
        
        if not response:
            return result
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('URGENT:'):
                urgent_value = line.split(':', 1)[1].strip().upper()
                result['urgent'] = urgent_value == 'YES'
            
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence_str = line.split(':', 1)[1].strip()
                    result['confidence'] = float(confidence_str)
                except ValueError:
                    result['confidence'] = 0.5
            
            elif line.startswith('KEYWORDS:'):
                keywords_str = line.split(':', 1)[1].strip()
                result['keywords'] = [k.strip() for k in keywords_str.split(',') if k.strip()]
            
            elif line.startswith('ESCALATE:'):
                escalate_value = line.split(':', 1)[1].strip().upper()
                result['escalate'] = escalate_value == 'YES'
        
        return result
    
    def _parse_multimodal_response(self, response: str) -> Dict[str, Any]:
        """Parse full multimodal analysis response"""
        result = {
            'threat_detected': False,
            'confidence': 0.0,
            'threat_type': 'Unknown',
            'multimodal_correlation': 'No correlation analysis',
            'reasoning': 'No reasoning provided',
            'urgency': 'Low'
        }
        
        if not response:
            return result
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('THREAT:'):
                threat_value = line.split(':', 1)[1].strip().upper()
                result['threat_detected'] = threat_value == 'YES'
            
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence_str = line.split(':', 1)[1].strip()
                    result['confidence'] = float(confidence_str)
                except ValueError:
                    result['confidence'] = 0.5
            
            elif line.startswith('TYPE:'):
                result['threat_type'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('MULTIMODAL_CORRELATION:'):
                result['multimodal_correlation'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('URGENCY:'):
                result['urgency'] = line.split(':', 1)[1].strip()
        
        return result
    
    def get_mixnmatch_stats(self) -> Dict[str, Any]:
        """Get comprehensive Mix'n'Match statistics"""
        stats = {
            'models_available': list(self.models.keys()),
            'model_performance': self.model_stats.copy(),
            'mixnmatch_enabled': True,
            'total_analyses': sum(s['uses'] for s in self.model_stats.values())
        }
        
        # Calculate efficiency metrics
        if stats['total_analyses'] > 0:
            total_time = sum(s['total_time'] for s in self.model_stats.values())
            stats['average_analysis_time'] = total_time / stats['total_analyses']
            
            # Model usage distribution
            stats['model_usage_distribution'] = {
                model: (stats_data['uses'] / stats['total_analyses']) * 100
                for model, stats_data in self.model_stats.items()
            }
        
        return stats
    
    # Include all other methods from original ai_engine.py
    def _assess_scene_complexity(self, image) -> float:
        """Assess visual scene complexity for model selection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Edge density (more edges = more complex)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Variance in brightness (more variance = more complex)
            brightness_variance = np.var(gray) / (255 * 255)
            
            # Combine metrics
            complexity = (edge_density * 0.7) + (brightness_variance * 0.3)
            return min(complexity * 2, 1.0)  # Normalize to 0-1
            
        except:
            return 0.5  # Default medium complexity
    
    def _analyze_visual_content(self, image) -> str:
        """Quick visual content analysis for prompt preparation"""
        try:
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            brightness = np.mean(gray)
            
            description = f"Image: {width}x{height}, brightness: {brightness:.0f}/255"
            return description
            
        except Exception as e:
            return f"Visual analysis error: {e}"
    
    def _analyze_audio_content(self, audio_description: str) -> str:
        """Analyze audio content for prompt preparation"""
        if not audio_description:
            return "No audio data provided"
        
        return f"Audio description: {audio_description}"
    
    def _create_multimodal_prompt(self, visual_analysis: str, audio_analysis: str, audio_description: str) -> str:
        """Create multimodal analysis prompt"""
        return self.prompts['fallback_analysis'].format(
            visual_analysis=visual_analysis,
            audio_analysis=audio_analysis
        )
    
    def _image_to_base64(self, image) -> str:
        """Convert OpenCV image to base64"""
        try:
            height, width = image.shape[:2]
            if width > 1024:
                scale = 1024 / width
                new_width = 1024
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            _, buffer = cv2.imencode('.jpg', image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return image_b64
            
        except Exception as e:
            print(f"❌ Image encoding error: {e}")
            return ""
    
    def _parse_threat_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse standard threat response"""
        result = {
            'threat_detected': False,
            'confidence': 0.0,
            'threat_type': 'Unknown',
            'reasoning': 'No reasoning provided',
            'urgency': 'Low',
            'timestamp': datetime.now().isoformat(),
            'raw_response': ai_response
        }
        
        if not ai_response:
            return result
        
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('THREAT:'):
                threat_value = line.split(':', 1)[1].strip().upper()
                result['threat_detected'] = threat_value == 'YES'
            
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence_str = line.split(':', 1)[1].strip()
                    result['confidence'] = float(confidence_str)
                except ValueError:
                    result['confidence'] = 0.5
            
            elif line.startswith('TYPE:'):
                result['threat_type'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
            
            elif line.startswith('URGENCY:'):
                result['urgency'] = line.split(':', 1)[1].strip()
        
        return result
    
    def _get_performance_settings(self, performance_level: str) -> Dict[str, Any]:
        """Get performance settings"""
        settings_map = {
            "2b_efficient": {
                "temperature": 0.1,
                "top_p": 0.8,
                "num_predict": 200,
                "num_ctx": 2048
            },
            "3b_balanced": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 300,
                "num_ctx": 4096
            },
            "4b_full": {
                "temperature": 0.1,
                "top_p": 0.95,
                "num_predict": 500,
                "num_ctx": 8192
            }
        }
        
        return settings_map.get(performance_level, settings_map["3b_balanced"])
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            'threat_detected': False,
            'confidence': 0.0,
            'threat_type': 'Analysis Error',
            'reasoning': error_message,
            'urgency': 'Low',
            'timestamp': datetime.now().isoformat(),
            'error': True
        }


# Test the Mix'n'Match AI engine
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("🤖 Testing Gemma 3n Mix'n'Match AI Engine")
    print("=" * 50)
    
    # Mock config for testing
    class MockConfig:
        def get(self, key, default=None):
            return default
    
    # Initialize Mix'n'Match engine
    ai_engine = GemmaMixNMatchAI(MockConfig())
    
    if ai_engine.connection_status:
        print("✅ Mix'n'Match AI Engine ready for competition")
        
        # Test Mix'n'Match scenarios
        print("\n🧪 Testing Mix'n'Match scenarios...")
        
        test_scenarios = [
            ("Normal scene", "quiet background music"),
            ("Urgent audio", "help me please emergency"),
            ("Complex scene", "multiple people arguing loudly")
        ]
        
        for scenario, audio in test_scenarios:
            print(f"\n📋 Scenario: {scenario}")
            print(f"Audio: {audio}")
            
            # Mock image for testing
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Test Mix'n'Match analysis
            result = ai_engine.analyze_multimodal_threat(
                image=test_image, 
                audio_description=audio, 
                performance_level="mixnmatch"
            )
            
            print(f"Analysis Path: {result.get('analysis_path', 'unknown')}")
            print(f"Models Used: {result.get('models_used', [])}")
            print(f"Total Time: {result.get('total_analysis_time', 0):.2f}s")
        
        # Show Mix'n'Match stats
        stats = ai_engine.get_mixnmatch_stats()
        print(f"\n📊 Mix'n'Match Stats:")
        for key, value in stats.items():
            if key != 'model_performance':
                print(f"  {key}: {value}")
        
        print(f"\n📈 Model Performance:")
        for model, perf in stats['model_performance'].items():
            print(f"  {model}: {perf['uses']} uses, {perf['avg_time']:.2f}s avg")
        
    else:
        print("❌ Mix'n'Match AI Engine not ready - check model downloads")
        print("💡 Download missing models with:")
        print("   ollama pull gemma3n:2b")
        print("   ollama pull gemma3n:e4b")
    
    print("\n🛡️ Mix'n'Match AI Engine test complete")
