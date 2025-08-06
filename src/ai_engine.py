"""
WARYON Dynamic AI Engine
Gemma 3n Competition-Ready Multimodal Threat Detection
Uses Single Model with Dynamic Performance Scaling
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

PerformanceLevel = Literal["2b_efficient", "4b_full", "3b_balanced", "auto"]

class GemmaDynamicAI:
    def __init__(self, config_manager):
        self.config = config_manager
        self.ollama_url = "http://localhost:11434"
        
        # Single model with dynamic performance
        self.model = "gemma3n:e4b"
        
        # Performance monitoring
        self.performance_stats = {
            "total_analyses": 0,
            "2b_efficient_count": 0,
            "4b_full_count": 0,
            "3b_balanced_count": 0,
            "threats_detected": 0,
            "false_positives": 0,
            "avg_response_time": 0.0,
            "resource_usage": []
        }
        
        # Dynamic performance thresholds
        self.performance_thresholds = {
            "complex_scene_threshold": 0.6,
            "audio_urgency_threshold": 0.7,
            "system_load_threshold": 0.8,
            "battery_low_threshold": 0.3
        }
        
        # Threat detection prompts optimized for multimodal analysis
        self.prompts = self._initialize_prompts()
        
        # Test connection and capabilities
        self.connection_status = self.test_connection()
        
        print(f"🤖 Gemma 3n Dynamic AI Engine initialized")
        print(f"   Model: {self.model}")
        print(f"   Connection: {'✅ Ready' if self.connection_status else '❌ Failed'}")
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize multimodal threat detection prompts"""
        return {
            'multimodal_threat': """You are WARYON, an advanced AI safety system using Gemma 3n's multimodal capabilities.

Analyze the provided image AND audio context for safety threats. Look for:

VISUAL THREATS:
- Violence: hitting, restraining, threatening gestures, weapons
- Medical emergencies: falls, collapse, distress expressions
- Bullying: intimidation, aggressive confrontation, fear responses
- Expressions: fear, panic, distress, pain in faces

AUDIO THREATS:
- Distress calls: screaming, crying, calls for help
- Threatening language: verbal abuse, threats, aggressive shouting
- Emergency sounds: crashes, breaking, unusual silence

CONTEXT AWARENESS:
- Distinguish entertainment (TV, games) from real threats
- Consider normal activities vs emergency situations
- Analyze facial expressions and body language carefully
- Evaluate audio-visual correlation for accuracy

Respond EXACTLY in this format:
THREAT: [YES/NO]
CONFIDENCE: [0.0-1.0]
TYPE: [Violence/Fall/Bullying/Distress/Normal Activity]
VISUAL_ANALYSIS: [What you see in the image]
AUDIO_ANALYSIS: [What you detect in audio]
REASONING: [Why this is/isn't a threat]
URGENCY: [Low/Medium/High]""",

            'expression_analysis': """You are WARYON analyzing facial expressions for safety assessment.

Focus specifically on:
- Facial expressions indicating fear, panic, distress, pain
- Eye movements and gaze patterns
- Mouth position (screaming, crying, shock)
- Overall body language and posture
- Signs of physical or emotional distress

THREAT: [YES/NO]
CONFIDENCE: [0.0-1.0]
EXPRESSION: [Specific expression detected]
EMOTION: [Fear/Panic/Distress/Pain/Neutral]
REASONING: [Detailed expression analysis]""",

            'scene_context': """Analyze the overall scene context for threat assessment:

- Environment type (home, office, public, etc.)
- Number of people and their relationships
- Activities being performed
- Objects and potential weapons present
- Lighting and visibility conditions
- Any signs of struggle or disturbance

CONTEXT: [Scene description]
THREAT_INDICATORS: [List any concerning elements]
SAFETY_ASSESSMENT: [Overall safety evaluation]"""
        }
    
    def test_connection(self) -> bool:
        """Test connection to Gemma 3n model"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if self.model not in model_names:
                print(f"❌ Model {self.model} not found")
                return False
            
            # Test basic query
            test_response = self._query_model("Test connection. Respond with 'WARYON AI READY'", "2b_efficient")
            return test_response and "ready" in test_response.lower()
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def analyze_multimodal_threat(self, image, audio_description: str = "", performance_level: PerformanceLevel = "auto") -> Dict[str, Any]:
        """Analyze both visual and audio data for threats using dynamic performance"""
        start_time = time.time()
        
        try:
            # Auto-select performance level if needed
            if performance_level == "auto":
                performance_level = self._auto_select_performance(image, audio_description)
            
            # Update statistics
            self.performance_stats["total_analyses"] += 1
            self.performance_stats[f"{performance_level}_count"] += 1
            
            print(f"🔍 Multimodal analysis using {performance_level} performance")
            
            # Prepare multimodal prompt
            visual_analysis = self._analyze_visual_content(image, performance_level)
            audio_analysis = self._analyze_audio_content(audio_description, performance_level)
            
            # Combine analyses
            combined_prompt = self._create_multimodal_prompt(visual_analysis, audio_analysis, audio_description)
            
            # Send to Gemma 3n with appropriate performance settings
            ai_response = self._query_model_with_image(combined_prompt, image, performance_level)
            
            # Process response
            result = self._parse_threat_response(ai_response, performance_level)
            
            # Update performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time, result)
            
            return result
            
        except Exception as e:
            print(f"❌ Multimodal analysis error: {e}")
            return self._create_error_response(f"Analysis error: {e}")
    
    def _auto_select_performance(self, image, audio_description: str) -> PerformanceLevel:
        """Automatically select optimal performance level based on situation"""
        
        # Analyze scene complexity
        scene_complexity = self._assess_scene_complexity(image)
        audio_urgency = self._assess_audio_urgency(audio_description)
        system_resources = self._check_system_resources()
        
        print(f"📊 Auto-selection: Scene={scene_complexity:.2f}, Audio={audio_urgency:.2f}, Resources={system_resources:.2f}")
        
        # Decision logic for dynamic performance
        if audio_urgency > self.performance_thresholds["audio_urgency_threshold"] or \
           scene_complexity > self.performance_thresholds["complex_scene_threshold"]:
            return "4b_full"  # Maximum accuracy for critical situations
            
        elif system_resources > self.performance_thresholds["system_load_threshold"]:
            return "2b_efficient"  # Preserve resources when system is loaded
            
        else:
            return "3b_balanced"  # Balanced performance for normal situations
    
    def _assess_scene_complexity(self, image) -> float:
        """Assess visual scene complexity for performance selection"""
        try:
            # Simple complexity metrics
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
    
    def _assess_audio_urgency(self, audio_description: str) -> float:
        """Assess audio urgency for performance selection"""
        urgency_keywords = {
            "high": ["scream", "help", "emergency", "fire", "police", "stop", "hurt", "pain"],
            "medium": ["loud", "shouting", "crying", "arguing", "angry", "upset"],
            "low": ["talking", "quiet", "normal", "conversation", "music"]
        }
        
        audio_lower = audio_description.lower()
        
        high_count = sum(1 for word in urgency_keywords["high"] if word in audio_lower)
        medium_count = sum(1 for word in urgency_keywords["medium"] if word in audio_lower)
        low_count = sum(1 for word in urgency_keywords["low"] if word in audio_lower)
        
        if high_count > 0:
            return 0.9
        elif medium_count > 0:
            return 0.6
        elif low_count > 0:
            return 0.2
        else:
            return 0.4  # Unknown audio gets medium urgency
    
    def _check_system_resources(self) -> float:
        """Check current system resource usage"""
        try:
            import psutil # pyright: ignore[reportMissingModuleSource]
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Combine CPU and memory usage
            resource_usage = (cpu_percent + memory_percent) / 200.0
            return min(resource_usage, 1.0)
            
        except ImportError:
            # Fallback if psutil not available
            return 0.5
    
    def _analyze_visual_content(self, image, performance_level: PerformanceLevel) -> str:
        """Quick visual content analysis for prompt preparation"""
        try:
            height, width = image.shape[:2]
            
            # Basic image analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            brightness = np.mean(gray)
            
            # Simple content detection
            faces = self._detect_faces_basic(image)
            
            description = f"Image: {width}x{height}, brightness: {brightness:.0f}/255"
            
            if faces > 0:
                description += f", {faces} face(s) detected"
            
            if performance_level == "4b_full":
                description += ", requesting detailed analysis"
            
            return description
            
        except Exception as e:
            return f"Visual analysis error: {e}"
    
    def _detect_faces_basic(self, image) -> int:
        """Basic face detection for content analysis"""
        try:
            # Simple Haar cascade face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return len(faces)
        except:
            return 0
    
    def _analyze_audio_content(self, audio_description: str, performance_level: PerformanceLevel) -> str:
        """Analyze audio content for prompt preparation"""
        if not audio_description:
            return "No audio data provided"
        
        # Basic audio analysis
        volume_indicators = ["loud", "quiet", "silent", "normal"]
        emotion_indicators = ["angry", "scared", "calm", "excited", "distressed"]
        
        analysis = f"Audio description: {audio_description}"
        
        if performance_level == "4b_full":
            analysis += " (detailed audio analysis requested)"
        
        return analysis
    
    def _create_multimodal_prompt(self, visual_analysis: str, audio_analysis: str, audio_description: str) -> str:
        """Create comprehensive multimodal analysis prompt"""
        base_prompt = self.prompts['multimodal_threat']
        
        context = f"""
VISUAL_CONTEXT: {visual_analysis}
AUDIO_CONTEXT: {audio_analysis}
AUDIO_DESCRIPTION: {audio_description if audio_description else "No audio data"}

Now analyze the provided image for threats, considering the audio context above.
"""
        
        return base_prompt + "\n" + context
    
    def _query_model_with_image(self, prompt: str, image, performance_level: PerformanceLevel) -> Optional[str]:
        """Query Gemma 3n with image and performance optimization"""
        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(image)
            
            # Performance-based settings
            performance_settings = self._get_performance_settings(performance_level)
            
            query_data = {
                'model': self.model,
                'prompt': prompt,
                'images': [image_b64],
                'stream': False,
                'options': performance_settings
            }
            
            print(f"🔍 Sending multimodal query with {performance_level} settings...")
            
            timeout = 30 if performance_level == "2b_efficient" else 60
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=query_data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '').strip()
                print(f"✅ Response received: {len(ai_response)} characters")
                return ai_response
            else:
                print(f"❌ Query failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Model query error: {e}")
            return None
    
    def _query_model(self, prompt: str, performance_level: PerformanceLevel) -> Optional[str]:
        """Query model for text-only analysis"""
        try:
            performance_settings = self._get_performance_settings(performance_level)
            
            query_data = {
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': performance_settings
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
            print(f"❌ Text query error: {e}")
            return None
    
    def _get_performance_settings(self, performance_level: PerformanceLevel) -> Dict[str, Any]:
        """Get optimized settings for different performance levels"""
        settings_map = {
            "2b_efficient": {
                "temperature": 0.1,
                "top_p": 0.8,
                "num_predict": 200,
                # Simulated 2B performance settings
                "repeat_penalty": 1.1,
                "num_ctx": 2048
            },
            "3b_balanced": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 300,
                # Balanced performance
                "repeat_penalty": 1.05,
                "num_ctx": 4096
            },
            "4b_full": {
                "temperature": 0.1,
                "top_p": 0.95,
                "num_predict": 500,
                # Maximum accuracy settings
                "repeat_penalty": 1.0,
                "num_ctx": 8192
            }
        }
        
        return settings_map.get(performance_level, settings_map["3b_balanced"])
    
    def _image_to_base64(self, image) -> str:
        """Convert OpenCV image to base64 for AI analysis"""
        try:
            # Optimize image size based on performance level
            height, width = image.shape[:2]
            if width > 1024:  # Resize for efficiency
                scale = 1024 / width
                new_width = 1024
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Convert to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return image_b64
            
        except Exception as e:
            print(f"❌ Image encoding error: {e}")
            return ""
    
    def _parse_threat_response(self, ai_response: str, performance_level: PerformanceLevel) -> Dict[str, Any]:
        """Parse AI response into structured threat assessment"""
        try:
            result = {
                'threat_detected': False,
                'confidence': 0.0,
                'threat_type': 'Unknown',
                'visual_analysis': 'No analysis available',
                'audio_analysis': 'No analysis available',
                'reasoning': 'No reasoning provided',
                'urgency': 'Low',
                'performance_level': performance_level,
                'timestamp': datetime.now().isoformat(),
                'raw_response': ai_response
            }
            
            if not ai_response:
                return result
            
            # Parse structured response
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
                
                elif line.startswith('VISUAL_ANALYSIS:'):
                    result['visual_analysis'] = line.split(':', 1)[1].strip()
                
                elif line.startswith('AUDIO_ANALYSIS:'):
                    result['audio_analysis'] = line.split(':', 1)[1].strip()
                
                elif line.startswith('REASONING:'):
                    result['reasoning'] = line.split(':', 1)[1].strip()
                
                elif line.startswith('URGENCY:'):
                    result['urgency'] = line.split(':', 1)[1].strip()
            
            return result
            
        except Exception as e:
            print(f"❌ Response parsing error: {e}")
            return self._create_error_response(f"Parsing error: {e}")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response structure"""
        return {
            'threat_detected': False,
            'confidence': 0.0,
            'threat_type': 'Analysis Error',
            'visual_analysis': 'Error occurred',
            'audio_analysis': 'Error occurred',
            'reasoning': error_message,
            'urgency': 'Low',
            'performance_level': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'error': True
        }
    
    def _update_performance_metrics(self, response_time: float, result: Dict[str, Any]):
        """Update performance tracking metrics"""
        try:
            # Update response time average
            total_analyses = self.performance_stats["total_analyses"]
            current_avg = self.performance_stats["avg_response_time"]
            
            new_avg = ((current_avg * (total_analyses - 1)) + response_time) / total_analyses
            self.performance_stats["avg_response_time"] = new_avg
            
            # Track threat detection
            if result.get('threat_detected', False):
                self.performance_stats["threats_detected"] += 1
            
            # Track resource usage
            self.performance_stats["resource_usage"].append({
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time,
                "performance_level": result.get('performance_level', 'unknown')
            })
            
            # Keep only last 100 resource measurements
            if len(self.performance_stats["resource_usage"]) > 100:
                self.performance_stats["resource_usage"].pop(0)
                
        except Exception as e:
            print(f"⚠️ Metrics update error: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = self.performance_stats.copy()
        
        # Add calculated metrics
        total = stats["total_analyses"]
        if total > 0:
            stats["performance_distribution"] = {
                "2b_efficient": (stats["2b_efficient_count"] / total) * 100,
                "4b_full": (stats["4b_full_count"] / total) * 100,
                "3b_balanced": (stats["3b_balanced_count"] / total) * 100
            }
            
            stats["accuracy"] = (stats["threats_detected"] - stats["false_positives"]) / total if total > 0 else 0.0
        
        return stats
    
    def force_performance_level(self, level: PerformanceLevel):
        """Force a specific performance level for testing"""
        self._forced_performance = level
        print(f"🔧 Performance level forced to: {level}")
    
    def reset_performance_forcing(self):
        """Reset to automatic performance selection"""
        self._forced_performance = None
        print("🔧 Performance selection reset to automatic")


# Test the dynamic AI engine
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("🤖 Testing Gemma 3n Dynamic AI Engine")
    print("=" * 50)
    
    # Mock config for testing
    class MockConfig:
        def get(self, key, default=None):
            return default
    
    # Initialize engine
    ai_engine = GemmaDynamicAI(MockConfig())
    
    if ai_engine.connection_status:
        print("✅ AI Engine ready for competition")
        
        # Test performance level selection
        print("\n🧪 Testing performance level selection...")
        
        # Test with different scenarios
        test_scenarios = [
            ("Normal scene", "quiet background music"),
            ("Complex scene", "loud shouting and arguing"),
            ("Emergency scene", "screaming for help emergency")
        ]
        
        for scenario, audio in test_scenarios:
            print(f"\n📋 Scenario: {scenario}")
            print(f"Audio: {audio}")
            
            # Mock image for testing
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Test auto performance selection
            performance = ai_engine._auto_select_performance(test_image, audio)
            print(f"Selected performance: {performance}")
        
        # Show performance stats
        stats = ai_engine.get_performance_stats()
        print(f"\n📊 Performance Stats: {stats}")
        
    else:
        print("❌ AI Engine not ready - check Gemma 3n setup")
    
    print("\n🛡️ Dynamic AI Engine test complete")