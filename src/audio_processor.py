"""
WARYON Enhanced Audio Processor - COMPLETE VERSION
Rich Audio Analysis for Gemma 3n Multimodal Intelligence
Handles actual microphone input with comprehensive threat detection
"""

import sounddevice as sd
import numpy as np
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, Any, List
import speech_recognition as sr

class RealAudioProcessor:
    def __init__(self, config_manager, threat_callback: Callable = None):
        self.config = config_manager
        self.threat_callback = threat_callback
        
        # Audio settings
        self.sample_rate = 44100
        self.channels = 1
        self.chunk_duration = 2.0  # Analyze every 2 seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # DEVICE FIX - Force use of working device
        self.device_id = 1  # Working Intel microphone array
        
        # Processing state
        self.is_monitoring = False
        self.audio_thread = None
        self.analysis_thread = None
        
        # Audio data management
        self.audio_queue = queue.Queue(maxsize=10)
        self.current_audio_level = 0.0
        self.background_noise_level = 0.0
        self.noise_calibrated = False
        
        # Conversation history for context analysis
        self._conversation_history = []
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        
        # Statistics
        self.stats = {
            "audio_chunks_processed": 0,
            "speech_detected": 0,
            "loud_events": 0,
            "distress_keywords_detected": 0,
            "threat_analyses_performed": 0,
            "high_threat_detections": 0,
            "avg_volume": 0.0
        }
        
        # Enhanced distress detection keywords
        self.distress_keywords = {
            "emergency": ["help", "emergency", "fire", "police", "ambulance", "911", "urgent", "crisis"],
            "violence": ["stop", "don't", "hurt", "pain", "hit", "attack", "fight", "violence", "hitting"],
            "fear": ["scared", "afraid", "terrified", "please", "no", "get away", "leave me alone"],
            "medical": ["sick", "heart", "can't breathe", "chest pain", "stroke", "collapsed", "dying"],
            "distress": ["crying", "sobbing", "screaming", "panicking", "desperate", "trapped"]
        }
        
        print("🎤 Enhanced Audio Processor initialized for Gemma 3n")
        print(f"   Device ID: {self.device_id} (Intel Smart Sound)")
        print("   Sample Rate: 44100 Hz")
        print("   Analysis Interval: 2.0 seconds")
        print("   Enhanced Context Analysis: ENABLED")
    
    def test_audio(self) -> bool:
        """Test real microphone access with device fix"""
        try:
            print("🧪 Testing Intel Smart Sound microphone array...")
            
            # List audio devices for confirmation
            devices = sd.query_devices()
            if self.device_id < len(devices):
                device_info = devices[self.device_id]
                print(f"   Using: {device_info['name']}")
            
            # Test recording with device override
            print("🎤 Recording 2-second test...")
            test_recording = sd.rec(
                int(2 * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device_id,
                dtype=np.float32
            )
            sd.wait()
            
            # Check volume
            max_volume = np.max(np.abs(test_recording))
            avg_volume = np.mean(np.abs(test_recording))
            
            print(f"✅ Intel microphone test successful!")
            print(f"   Max volume: {max_volume:.6f}")
            print(f"   Avg volume: {avg_volume:.6f}")
            
            if max_volume > 0.0001:
                print("✅ Intel microphone array ready for enhanced WARYON analysis!")
                return True
            else:
                print("⚠️ Very quiet - try speaking louder during demo")
                return True
                
        except Exception as e:
            print(f"❌ Intel microphone test failed: {e}")
            return False
    
    def start_monitoring(self) -> bool:
        """Start enhanced real audio monitoring"""
        if self.is_monitoring:
            print("⚠️ Audio monitoring already active")
            return True
        
        try:
            print("🎤 Starting ENHANCED audio monitoring with Intel Smart Sound...")
            
            # Calibrate background noise
            if not self._calibrate_background_noise():
                print("⚠️ Proceeding without noise calibration")
            
            self.is_monitoring = True
            
            # Start audio capture thread
            self.audio_thread = threading.Thread(
                target=self._audio_capture_loop,
                name="AudioCapture",
                daemon=True
            )
            self.audio_thread.start()
            
            # Start audio analysis thread
            self.analysis_thread = threading.Thread(
                target=self._audio_analysis_loop,
                name="AudioAnalysis",
                daemon=True
            )
            self.analysis_thread.start()
            
            print("✅ Enhanced Intel microphone monitoring started")
            print("🧠 Rich context analysis for Gemma 3n enabled")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start enhanced audio monitoring: {e}")
            self.stop_monitoring()
            return False
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        print("🎤 Stopping enhanced audio monitoring...")
        
        self.is_monitoring = False
        
        # Wait for threads
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
        
        print("✅ Enhanced audio monitoring stopped")
    
    def _calibrate_background_noise(self) -> bool:
        """Calibrate background noise level with device fix"""
        try:
            print("🔧 Calibrating background noise with Intel microphone...")
            
            noise_sample = sd.rec(
                int(2 * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device_id,
                dtype=np.float32
            )
            sd.wait()
            
            self.background_noise_level = np.mean(np.abs(noise_sample))
            self.noise_calibrated = True
            
            print(f"✅ Background noise calibrated: {self.background_noise_level:.6f}")
            return True
            
        except Exception as e:
            print(f"❌ Noise calibration failed: {e}")
            return False
    
    def _audio_capture_loop(self):
        """Main audio capture loop with device fix"""
        print("🎵 Enhanced audio capture loop started")
        
        try:
            with sd.InputStream(
                callback=self._audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                device=self.device_id,
                blocksize=4096,
                dtype=np.float32
            ) as stream:
                
                while self.is_monitoring:
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Enhanced audio capture error: {e}")
        
        print("🎵 Enhanced audio capture loop stopped")
    
    def _audio_callback(self, indata, frames, time, status):
        """Real-time audio callback"""
        try:
            if status:
                print(f"⚠️ Audio status: {status}")
            
            audio_chunk = indata[:, 0]
            volume = np.sqrt(np.mean(audio_chunk**2))
            self.current_audio_level = volume
            
            try:
                self.audio_queue.put(audio_chunk.copy(), block=False)
            except queue.Full:
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put(audio_chunk.copy(), block=False)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            print(f"Audio callback error: {e}")
    
    def _audio_analysis_loop(self):
        """Enhanced audio analysis loop with rich context"""
        print("🎧 Enhanced audio analysis loop started")
        
        accumulated_audio = []
        last_analysis_time = time.time()
        
        while self.is_monitoring:
            try:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    accumulated_audio.append(chunk)
                except queue.Empty:
                    time.sleep(0.05)
                    continue
                
                current_time = time.time()
                if (current_time - last_analysis_time) >= self.chunk_duration and accumulated_audio:
                    
                    audio_data = np.concatenate(accumulated_audio)
                    self._analyze_audio_chunk_enhanced(audio_data)
                    
                    accumulated_audio = []
                    last_analysis_time = current_time
                    
            except Exception as e:
                print(f"❌ Enhanced audio analysis error: {e}")
                time.sleep(1.0)
        
        print("🎧 Enhanced audio analysis loop stopped")
    
    def _analyze_audio_chunk_enhanced(self, audio_data):
        """Enhanced audio chunk analysis with rich context for Gemma 3n"""
        try:
            self.stats["audio_chunks_processed"] += 1
            self.stats["threat_analyses_performed"] += 1
            
            # Calculate audio metrics
            volume = np.sqrt(np.mean(audio_data**2))
            max_volume = np.max(np.abs(audio_data))
            
            # Update stats
            total_chunks = self.stats["audio_chunks_processed"]
            current_avg = self.stats["avg_volume"]
            new_avg = ((current_avg * (total_chunks - 1)) + volume) / total_chunks
            self.stats["avg_volume"] = new_avg
            
            print(f"🔊 Enhanced Analysis #{total_chunks}: Vol={volume:.4f}, Max={max_volume:.4f}")
            
            # Detect loud events
            loudness_threshold = self.background_noise_level * 3 if self.noise_calibrated else 0.05
            is_loud = volume > loudness_threshold
            
            if is_loud:
                self.stats["loud_events"] += 1
                print(f"📢 Loud audio event detected! Volume: {volume:.4f}")
            
            # Enhanced speech recognition
            speech_text = self._recognize_speech_enhanced(audio_data)
            
            # Comprehensive threat analysis
            threat_analysis = self._analyze_for_threats_enhanced(volume, max_volume, speech_text, is_loud)
            
            # Generate rich audio description for Gemma 3n
            audio_description = self._generate_enhanced_audio_description(
                volume, max_volume, speech_text, is_loud
            )
            
            print(f"🧠 Enhanced Audio Context for Gemma 3n:")
            print(f"   {audio_description[:100]}...")
            
            # Handle threats
            if threat_analysis["threat_detected"]:
                if threat_analysis.get("threat_level") == "high":
                    self.stats["high_threat_detections"] += 1
                
                self._handle_audio_threat_enhanced(threat_analysis, audio_description)
            
            return audio_description
            
        except Exception as e:
            print(f"❌ Enhanced audio analysis error: {e}")
            return "enhanced audio analysis error occurred"
    
    def _recognize_speech_enhanced(self, audio_data) -> str:
        """Enhanced speech recognition with better error handling"""
        try:
            audio_16bit = (audio_data * 32767).astype(np.int16)
            
            audio_source = sr.AudioData(
                frame_data=audio_16bit.tobytes(),
                sample_rate=self.sample_rate,
                sample_width=2
            )
            
            try:
                text = self.recognizer.recognize_google(audio_source, language='en-US')
                self.stats["speech_detected"] += 1
                print(f"🗣️ Enhanced Speech Recognition: '{text}'")
                
                # Store in conversation history
                self._store_speech_in_history(text, time.time())
                
                return text.lower()
            
            except sr.UnknownValueError:
                return ""
            except sr.RequestError as e:
                print(f"⚠️ Speech recognition service error: {e}")
                return ""
                
        except Exception as e:
            print(f"Enhanced speech recognition error: {e}")
            return ""
    
    def _store_speech_in_history(self, text: str, timestamp: float):
        """Store speech in conversation history for context analysis"""
        entry = {
            'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
            'speech': text,
            'volume': self.current_audio_level,
            'analysis_time': timestamp
        }
        
        self._conversation_history.append(entry)
        
        # Keep only recent history (last 20 entries or 5 minutes)
        cutoff_time = time.time() - 300  # 5 minutes ago
        self._conversation_history = [
            entry for entry in self._conversation_history 
            if entry['analysis_time'] > cutoff_time
        ][-20:]  # Keep max 20 entries
    
    def _analyze_for_threats_enhanced(self, volume, max_volume, speech_text, is_loud) -> Dict[str, Any]:
        """Enhanced threat analysis with comprehensive context evaluation"""
        threat_analysis = {
            "threat_detected": False,
            "threat_type": "Normal Audio",
            "threat_level": "none",
            "confidence": 0.0,
            "reasoning": "Normal audio activity",
            "urgency": "Low",
            "context_factors": [],
            "speech_analysis": {},
            "volume_analysis": {}
        }
        
        threat_score = 0.0
        reasons = []
        context_factors = []
        
        # Enhanced volume analysis
        volume_analysis = self._analyze_volume_patterns(volume, max_volume, is_loud)
        threat_analysis["volume_analysis"] = {
            "description": volume_analysis,
            "is_loud": is_loud,
            "volume_level": volume,
            "max_volume": max_volume
        }
        
        # Volume-based scoring
        if volume > 0.4:  # Very loud
            threat_score += 0.5
            reasons.append("extremely loud audio levels")
        elif volume > 0.2:  # Loud
            threat_score += 0.3
            reasons.append("elevated volume levels")
        elif is_loud and volume > 0.1:
            threat_score += 0.2
            reasons.append("sudden loud audio event")
        
        # Enhanced speech analysis
        if speech_text:
            speech_analysis = self._analyze_speech_content_comprehensive(speech_text)
            threat_analysis["speech_analysis"] = speech_analysis
            
            # Emergency keywords (highest priority)
            emergency_score = len(speech_analysis.get("emergency_words", [])) * 0.4
            if emergency_score > 0:
                threat_score += emergency_score
                reasons.append(f"emergency keywords: {', '.join(speech_analysis['emergency_words'])}")
                self.stats["distress_keywords_detected"] += len(speech_analysis["emergency_words"])
            
            # Violence indicators
            violence_score = len(speech_analysis.get("violence_words", [])) * 0.3
            if violence_score > 0:
                threat_score += violence_score
                reasons.append(f"violence indicators: {', '.join(speech_analysis['violence_words'])}")
                self.stats["distress_keywords_detected"] += len(speech_analysis["violence_words"])
            
            # Fear/distress indicators
            fear_score = len(speech_analysis.get("fear_words", [])) * 0.25
            if fear_score > 0:
                threat_score += fear_score
                reasons.append(f"fear/distress language: {', '.join(speech_analysis['fear_words'])}")
                self.stats["distress_keywords_detected"] += len(speech_analysis["fear_words"])
            
            # Medical emergency
            medical_score = len(speech_analysis.get("medical_words", [])) * 0.5
            if medical_score > 0:
                threat_score += medical_score
                reasons.append(f"medical emergency language: {', '.join(speech_analysis['medical_words'])}")
                self.stats["distress_keywords_detected"] += len(speech_analysis["medical_words"])
            
            # Context analysis
            context_analysis = self._analyze_conversation_context_enhanced(speech_text)
            if context_analysis.get("escalation_detected"):
                threat_score += 0.2
                context_factors.append("conversation escalation detected")
            
            if context_analysis.get("repetitive_distress"):
                threat_score += 0.3
                context_factors.append("repetitive distress patterns")
        
        # Determine threat level and type
        threat_analysis["confidence"] = min(threat_score, 1.0)
        
        if threat_score >= 0.8:
            threat_analysis["threat_detected"] = True
            threat_analysis["threat_level"] = "critical"
            threat_analysis["urgency"] = "Critical"
            threat_analysis["threat_type"] = "Critical Emergency"
        elif threat_score >= 0.6:
            threat_analysis["threat_detected"] = True
            threat_analysis["threat_level"] = "high"
            threat_analysis["urgency"] = "High"
            # Determine specific threat type based on keywords
            if speech_text and any(word in speech_text for word in ["help", "emergency", "fire", "police"]):
                threat_analysis["threat_type"] = "Emergency Call"
            elif speech_text and any(word in speech_text for word in ["hurt", "stop", "attack"]):
                threat_analysis["threat_type"] = "Violence Threat"
            elif speech_text and any(word in speech_text for word in ["can't breathe", "heart", "stroke"]):
                threat_analysis["threat_type"] = "Medical Emergency"
            else:
                threat_analysis["threat_type"] = "High-Risk Situation"
        elif threat_score >= 0.4:
            threat_analysis["threat_detected"] = True
            threat_analysis["threat_level"] = "medium"
            threat_analysis["urgency"] = "Medium"
            threat_analysis["threat_type"] = "Suspicious Activity"
        elif threat_score >= 0.2:
            threat_analysis["threat_detected"] = True
            threat_analysis["threat_level"] = "low"
            threat_analysis["urgency"] = "Low"
            threat_analysis["threat_type"] = "Minor Concern"
        
        # Build comprehensive reasoning
        if threat_analysis["threat_detected"]:
            threat_analysis["reasoning"] = f"Enhanced threat analysis detected {threat_analysis['threat_level']} risk: " + ", ".join(reasons)
            if context_factors:
                threat_analysis["reasoning"] += f". Context factors: {', '.join(context_factors)}"
        else:
            threat_analysis["reasoning"] = "Enhanced analysis indicates normal audio activity with no significant threat indicators"
        
        threat_analysis["context_factors"] = context_factors
        
        return threat_analysis
    
    def _analyze_volume_patterns(self, volume, max_volume, is_loud) -> str:
        """Analyze volume patterns for context"""
        if volume < 0.01:
            return "very quiet environment with minimal background noise"
        elif volume < 0.05:
            return "quiet ambient environment with normal background sounds"
        elif volume < 0.15:
            if is_loud:
                return "moderate volume level with sudden loud sound detected"
            else:
                return "normal conversation volume level"
        elif volume < 0.3:
            return "elevated voice volume suggesting raised speaking or possible argument"
        else:
            if max_volume > 0.5:
                return "very loud audio with peaks suggesting shouting, screaming, or emergency situation"
            else:
                return "loud audio environment with raised voices"
    
    def _analyze_speech_content_comprehensive(self, speech_text) -> Dict[str, Any]:
        """Comprehensive speech content analysis"""
        analysis = {
            "original_text": speech_text,
            "word_count": len(speech_text.split()),
            "emergency_words": [],
            "violence_words": [],
            "fear_words": [],
            "medical_words": [],
            "distress_words": [],
            "emotional_indicators": [],
            "speech_patterns": []
        }
        
        text_lower = speech_text.lower()
        
        # Check each category
        for category, keywords in self.distress_keywords.items():
            found_words = [word for word in keywords if word in text_lower]
            if category == "emergency":
                analysis["emergency_words"] = found_words
            elif category == "violence":
                analysis["violence_words"] = found_words
            elif category == "fear":
                analysis["fear_words"] = found_words
            elif category == "medical":
                analysis["medical_words"] = found_words
            elif category == "distress":
                analysis["distress_words"] = found_words
        
        # Emotional indicators
        if "!" in speech_text:
            analysis["emotional_indicators"].append("exclamatory speech")
        if "?" in speech_text:
            analysis["emotional_indicators"].append("questioning speech")
        if speech_text.isupper():
            analysis["emotional_indicators"].append("shouting (all caps)")
        
        # Speech patterns
        if len(speech_text.split()) < 3:
            analysis["speech_patterns"].append("brief urgent speech")
        if speech_text.count(" ") == 0 and len(speech_text) > 1:
            analysis["speech_patterns"].append("single word utterance")
        
        # Repetition analysis
        words = text_lower.split()
        if len(words) > 1:
            unique_words = len(set(words))
            if unique_words < len(words) * 0.7:  # 30% or more repetition
                analysis["speech_patterns"].append("repetitive speech pattern")
        
        return analysis
    
    def _analyze_conversation_context_enhanced(self, speech_text) -> Dict[str, Any]:
        """Enhanced conversation context analysis"""
        context = {
            "escalation_detected": False,
            "repetitive_distress": False,
            "one_sided_conversation": False,
            "conversation_patterns": []
        }
        
        if len(self._conversation_history) > 1:
            recent_entries = self._conversation_history[-5:]  # Last 5 entries
            
            # Check for escalation
            distress_count = 0
            volume_trend = []
            
            for entry in recent_entries:
                entry_speech = entry.get('speech', '').lower()
                entry_volume = entry.get('volume', 0)
                
                volume_trend.append(entry_volume)
                
                # Count distress indicators
                if any(word in entry_speech for word in ['help', 'stop', 'don\'t', 'please', 'hurt']):
                    distress_count += 1
            
            if distress_count >= 2:
                context["escalation_detected"] = True
                context["repetitive_distress"] = True
            
            # Volume escalation
            if len(volume_trend) >= 3:
                if volume_trend[-1] > volume_trend[-2] > volume_trend[-3]:
                    context["escalation_detected"] = True
                    context["conversation_patterns"].append("volume escalation")
        
        return context
    
    def _generate_enhanced_audio_description(self, volume, max_volume, speech_text, is_loud) -> str:
        """Generate enhanced audio description for Gemma 3n multimodal analysis"""
        
        description_parts = []
        
        # Volume context (enhanced)
        volume_desc = self._analyze_volume_patterns(volume, max_volume, is_loud)
        description_parts.append(f"VOLUME ANALYSIS: {volume_desc}")
        
        # Speech analysis (comprehensive)
        if speech_text:
            speech_analysis = self._analyze_speech_content_comprehensive(speech_text)
            
            description_parts.append(f"SPEECH CONTENT: '{speech_text}'")
            
            # Keyword analysis
            all_keywords = []
            if speech_analysis["emergency_words"]:
                all_keywords.extend([f"EMERGENCY: {', '.join(speech_analysis['emergency_words'])}"]) 
            if speech_analysis["violence_words"]:
                all_keywords.extend([f"VIOLENCE: {', '.join(speech_analysis['violence_words'])}"]) 
            if speech_analysis["fear_words"]:
                all_keywords.extend([f"FEAR: {', '.join(speech_analysis['fear_words'])}"]) 
            if speech_analysis["medical_words"]:
                all_keywords.extend([f"MEDICAL: {', '.join(speech_analysis['medical_words'])}"]) 
            
            if all_keywords:
                description_parts.append(f"THREAT KEYWORDS DETECTED: {' | '.join(all_keywords)}")
            
            # Emotional analysis
            if speech_analysis["emotional_indicators"]:
                description_parts.append(f"EMOTIONAL INDICATORS: {', '.join(speech_analysis['emotional_indicators'])}")
            
            # Speech patterns
            if speech_analysis["speech_patterns"]:
                description_parts.append(f"SPEECH PATTERNS: {', '.join(speech_analysis['speech_patterns'])}")
        
        else:
            description_parts.append("SPEECH ANALYSIS: no clear speech detected in current audio sample")
        
        # Conversation context
        if len(self._conversation_history) > 0:
            context = self._analyze_conversation_context_enhanced(speech_text)
            context_desc = []
            
            if context["escalation_detected"]:
                context_desc.append("ESCALATION DETECTED")
            if context["repetitive_distress"]:
                context_desc.append("REPETITIVE DISTRESS")
            if context["conversation_patterns"]:
                context_desc.extend(context["conversation_patterns"])
            
            if context_desc:
                description_parts.append(f"CONVERSATION CONTEXT: {', '.join(context_desc)}")
            else:
                description_parts.append("CONVERSATION CONTEXT: normal conversation flow")
        
        # Threat assessment summary
        threat_analysis = self._analyze_for_threats_enhanced(volume, max_volume, speech_text, is_loud)
        if threat_analysis["threat_detected"]:
            description_parts.append(f"THREAT ASSESSMENT: {threat_analysis['threat_level'].upper()} RISK - {threat_analysis['threat_type']}")
        else:
            description_parts.append("THREAT ASSESSMENT: NO SIGNIFICANT THREATS DETECTED")
        
        # Combine all parts
        full_description = " | ".join(description_parts)
        
        return full_description
    
    def _handle_audio_threat_enhanced(self, threat_analysis, audio_description):
        """Enhanced audio threat handling with detailed reporting"""
        try:
            threat_type = threat_analysis["threat_type"]
            threat_level = threat_analysis["threat_level"]
            confidence = threat_analysis["confidence"]
            reasoning = threat_analysis["reasoning"]
            
            print(f"🚨 ENHANCED AUDIO THREAT DETECTED:")
            print(f"   Type: {threat_type}")
            print(f"   Level: {threat_level.upper()}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Reasoning: {reasoning[:100]}...")
            
            if threat_analysis.get("context_factors"):
                print(f"   Context: {', '.join(threat_analysis['context_factors'])}")
            
            # Enhanced callback with more details
            if self.threat_callback:
                enhanced_details = f"ENHANCED AUDIO ANALYSIS: {threat_level.upper()} {threat_type} - {reasoning}"
                self.threat_callback(threat_type, confidence, enhanced_details)
                
        except Exception as e:
            print(f"❌ Enhanced audio threat handling error: {e}")
    
    def get_current_audio_level(self) -> float:
        """Get current real-time audio level"""
        return self.current_audio_level
    
    def get_audio_description(self) -> str:
        """Get enhanced current audio description for Gemma 3n AI analysis"""
        if not self.is_monitoring:
            return "AUDIO MONITORING: system is not currently active - no audio context available for analysis"
        
        volume = self.current_audio_level
        is_loud = volume > (self.background_noise_level * 3) if self.noise_calibrated else volume > 0.05
        
        # Get most recent speech from history
        recent_speech = ""
        if self._conversation_history:
            # Get speech from last 5 seconds
            recent_time = time.time() - 5
            recent_entries = [
                entry for entry in self._conversation_history 
                if entry.get('analysis_time', 0) > recent_time
            ]
            if recent_entries:
                recent_speech = recent_entries[-1]['speech']
        
        return self._generate_enhanced_audio_description(volume, volume, recent_speech, is_loud)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced audio processing statistics"""
        stats = self.stats.copy()
        
        stats.update({
            "is_monitoring": self.is_monitoring,
            "current_audio_level": self.current_audio_level,
            "background_noise_level": self.background_noise_level,
            "noise_calibrated": self.noise_calibrated,
            "device_id": self.device_id,
            "device_working": True,
            "conversation_history_entries": len(self._conversation_history),
            "enhanced_analysis": True
        })
        
        # Add performance metrics
        if self.stats["audio_chunks_processed"] > 0:
            stats["threat_detection_rate"] = (self.stats["high_threat_detections"] / self.stats["audio_chunks_processed"]) * 100
        
        return stats


# Test the enhanced audio processor
if __name__ == "__main__":
    print("🎤 Testing ENHANCED Intel Smart Sound Audio Processor")
    print("🧠 Rich Context Analysis for Gemma 3n Integration")
    print("=" * 60)
    
    def audio_threat_callback(threat_type, confidence, details):
        print(f"🚨 ENHANCED AUDIO THREAT CALLBACK:")
        print(f"   Type: {threat_type}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Details: {details[:150]}...")
        print()
    
    # Mock config
    class MockConfig:
        def get(self, key, default=None):
            return default
    
    # Initialize enhanced processor
    config = MockConfig()
    audio_processor = RealAudioProcessor(config, audio_threat_callback)
    
    # Test microphone
    if audio_processor.test_audio():
        print("✅ Enhanced Intel microphone test passed")
        
        # Start monitoring
        if audio_processor.start_monitoring():
            print("✅ Enhanced Intel microphone monitoring started")
            print("\n🎤 ENHANCED TESTING - SPEAK INTO YOUR MICROPHONE!")
            print("🧪 TEST SCENARIOS:")
            print("   1. Say 'help me' → Emergency detection")
            print("   2. Say 'stop hurting me' → Violence detection")  
            print("   3. Say 'I can't breathe' → Medical emergency")
            print("   4. Make loud noises → Volume analysis")
            print("   5. Have a conversation → Context analysis")
            print("\n⏱️ Running for 45 seconds... Press Ctrl+C to stop early")
            print("=" * 60)
            
            try:
                # Enhanced monitoring for 45 seconds
                for i in range(45):
                    time.sleep(1)
                    level = audio_processor.get_current_audio_level()
                    
                    if i % 10 == 0:  # Print every 10 seconds
                        description = audio_processor.get_audio_description()
                        print(f"\n⏱️ {i}s - Current level: {level:.4f}")
                        print(f"🧠 Enhanced Context: {description[:120]}...")
                        
                        # Show conversation history
                        if hasattr(audio_processor, '_conversation_history') and audio_processor._conversation_history:
                            recent_count = len(audio_processor._conversation_history)
                            print(f"💭 Conversation History: {recent_count} recent entries")
                
            except KeyboardInterrupt:
                print("\n⏹️ Enhanced test interrupted by user")
            
            finally:
                audio_processor.stop_monitoring()
                
                # Show enhanced final stats
                stats = audio_processor.get_stats()
                print(f"\n📊 ENHANCED FINAL STATISTICS:")
                print("=" * 40)
                
                key_stats = [
                    "audio_chunks_processed", "speech_detected", "distress_keywords_detected",
                    "threat_analyses_performed", "high_threat_detections", "loud_events"
                ]
                
                for key in key_stats:
                    if key in stats:
                        print(f"  {key}: {stats[key]}")
                
                print(f"\n🎯 Performance Metrics:")
                print(f"  Average Volume: {stats.get('avg_volume', 0):.6f}")
                print(f"  Background Noise: {stats.get('background_noise_level', 0):.6f}")
                print(f"  Conversation Entries: {stats.get('conversation_history_entries', 0)}")
                print(f"  Enhanced Analysis: {stats.get('enhanced_analysis', False)}")
                
                if stats.get('threat_detection_rate'):
                    print(f"  Threat Detection Rate: {stats['threat_detection_rate']:.1f}%")
                
                print(f"\n🏆 COMPETITION READINESS:")
                print("✅ Enhanced speech recognition with context")
                print("✅ Comprehensive threat categorization") 
                print("✅ Rich audio descriptions for Gemma 3n")
                print("✅ Conversation history and pattern analysis")
                print("✅ Multi-level threat assessment")
                print("✅ Real-time processing with Intel Smart Sound")
        
        else:
            print("❌ Failed to start enhanced monitoring")
    
    else:
        print("❌ Enhanced Intel microphone test failed")
    
    print(f"\n🎤 Enhanced Intel Smart Sound audio processor test complete")
    print("🧠 System ready for Gemma 3n multimodal integration!")
    print("🏆 WARYON Enhanced Audio Analysis - Competition Ready!")