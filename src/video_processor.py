"""
WARYON Video Processor
Competition-Ready Real-Time Threat Detection with Gemma 3n
"""

import cv2
import threading
import time
import queue
import numpy as np
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, Any
from pathlib import Path
import json

class CompetitionVideoProcessor:
    def __init__(self, config_manager, threat_callback: Callable = None):
        self.config = config_manager
        self.threat_callback = threat_callback
        
        # Camera settings
        self.camera_index = 0
        self.capture_fps = 30
        self.analysis_interval = 2.0  # Analyze every 2 seconds for competition demo
        
        # Processing state
        self.is_monitoring = False
        self.capture_thread = None
        self.analysis_thread = None
        self.camera = None
        
        # Frame management
        self.frame_queue = queue.Queue(maxsize=10)
        self.current_frame = None
        self.last_analysis_time = datetime.now()
        
        # Competition statistics
        self.stats = {
            "frames_captured": 0,
            "frames_analyzed": 0,
            "threats_detected": 0,
            "performance_switches": 0,
            "avg_fps": 0.0,
            "analysis_times": [],
            "threat_history": []
        }
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.current_fps = 0.0
        
        # AI Engine reference
        self.ai_engine = None
        
        print("📹 Competition Video Processor initialized")
    
    def set_ai_engine(self, ai_engine):
        """Set reference to dynamic AI engine"""
        self.ai_engine = ai_engine
        print("🤖 AI Engine connected to video processor")
    
    def test_camera(self) -> bool:
        """Test camera access for competition demo"""
        try:
            print("🧪 Testing camera for competition demo...")
            
            test_camera = cv2.VideoCapture(self.camera_index)
            
            if not test_camera.isOpened():
                print("❌ Camera: Cannot open device")
                return False
            
            ret, frame = test_camera.read()
            
            if not ret or frame is None:
                print("❌ Camera: Cannot read frames")
                test_camera.release()
                return False
            
            # Get properties
            width = int(test_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(test_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = test_camera.get(cv2.CAP_PROP_FPS)
            
            test_camera.release()
            
            print(f"✅ Camera ready: {width}x{height} @ {fps} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Camera test failed: {e}")
            return False
    
    def start_monitoring(self) -> bool:
        """Start video monitoring for competition demo"""
        if self.is_monitoring:
            print("⚠️ Monitoring already active")
            return True
        
        try:
            print("📹 Starting competition video monitoring...")
            
            if not self._initialize_camera():
                return False
            
            self.is_monitoring = True
            self.stats["start_time"] = datetime.now()
            
            # Start capture thread
            self.capture_thread = threading.Thread(
                target=self._capture_loop,
                name="VideoCapture",
                daemon=True
            )
            self.capture_thread.start()
            
            # Start analysis thread
            self.analysis_thread = threading.Thread(
                target=self._analysis_loop,
                name="VideoAnalysis",
                daemon=True
            )
            self.analysis_thread.start()
            
            print("✅ Competition monitoring started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start monitoring: {e}")
            self.stop_monitoring()
            return False
    
    def stop_monitoring(self):
        """Stop video monitoring"""
        print("📹 Stopping video monitoring...")
        
        self.is_monitoring = False
        
        # Wait for threads
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
        
        print("✅ Video monitoring stopped")
    
    def _initialize_camera(self) -> bool:
        """Initialize camera with competition-optimal settings"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                print("❌ Cannot open camera")
                return False
            
            # Optimal settings for competition demo
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test capture
            ret, test_frame = self.camera.read()
            if not ret:
                print("❌ Cannot capture test frame")
                return False
            
            print(f"✅ Camera initialized: {test_frame.shape[1]}x{test_frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def _capture_loop(self):
        """Main camera capture loop with competition metrics"""
        print("🎥 Competition capture loop started")
        
        frame_interval = 1.0 / self.capture_fps
        last_frame_time = time.time()
        
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Maintain target FPS
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                # Capture frame
                ret, frame = self.camera.read()
                
                if ret and frame is not None:
                    self.stats["frames_captured"] += 1
                    self.current_frame = frame.copy()
                    
                    # Add to analysis queue (non-blocking)
                    try:
                        self.frame_queue.put(frame, block=False)
                    except queue.Full:
                        pass  # Skip frame if queue full
                    
                    # Update FPS
                    self._update_fps_counter()
                    last_frame_time = current_time
                
                else:
                    print("⚠️ Failed to capture frame")
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"❌ Capture loop error: {e}")
                time.sleep(0.1)
        
        print("🎥 Capture loop stopped")
    
    def _analysis_loop(self):
        """Competition analysis loop with dynamic performance"""
        print("🤖 Competition analysis loop started")
        
        while self.is_monitoring:
            try:
                current_time = datetime.now()
                
                # Check analysis interval
                if (current_time - self.last_analysis_time).total_seconds() < self.analysis_interval:
                    time.sleep(0.1)
                    continue
                
                # Get frame for analysis
                frame_to_analyze = None
                
                try:
                    while not self.frame_queue.empty():
                        frame_to_analyze = self.frame_queue.get_nowait()
                except queue.Empty:
                    if self.current_frame is not None:
                        frame_to_analyze = self.current_frame.copy()
                
                # Analyze frame
                if frame_to_analyze is not None:
                    self._analyze_frame_competition(frame_to_analyze)
                    self.last_analysis_time = current_time
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Analysis loop error: {e}")
                time.sleep(1.0)
        
        print("🤖 Analysis loop stopped")
    
    def _analyze_frame_competition(self, frame):
        """Competition-ready frame analysis with performance metrics"""
        try:
            if not self.ai_engine:
                print("⚠️ AI engine not available")
                return
            
            analysis_start = time.time()
            self.stats["frames_analyzed"] += 1
            
            print(f"🔍 Competition Analysis #{self.stats['frames_analyzed']} (FPS: {self.current_fps:.1f})")
            
            # Generate mock audio description for demo
            audio_description = self._generate_mock_audio_description()
            
            # Run multimodal AI analysis with auto-performance selection
            result = self.ai_engine.analyze_multimodal_threat(
                image=frame,
                audio_description=audio_description,
                performance_level="auto"
            )
            
            analysis_time = time.time() - analysis_start
            self.stats["analysis_times"].append(analysis_time)
            
            # Process results
            if result and not result.get('error', False):
                self._process_analysis_result(frame, result, analysis_time)
            else:
                print("⚠️ Analysis failed or returned error")
                
        except Exception as e:
            print(f"❌ Frame analysis error: {e}")
    
    def _generate_mock_audio_description(self) -> str:
        """Generate real audio description from audio processor"""
        # If we have a real audio processor, use it
        if hasattr(self, 'audio_processor') and self.audio_processor:
            return self.audio_processor.get_audio_description()
        
        # Fallback to mock for testing
        import random
        audio_scenarios = [
            "normal conversation in background",
            "quiet ambient room noise",
            "television playing in distance",
            "typing on keyboard",
            "phone notification sound",
            "footsteps approaching",
            "door opening and closing",
            "music playing softly"
        ]
        
        return random.choice(audio_scenarios)
    
    def set_audio_processor(self, audio_processor):
        """Set reference to real audio processor"""
        self.audio_processor = audio_processor
        print("🎤 Real audio processor connected to video processor")
    
    def _process_analysis_result(self, frame, result: Dict[str, Any], analysis_time: float):
        """Process AI analysis result with competition metrics"""
        try:
            confidence = result.get('confidence', 0.0)
            threat_detected = result.get('threat_detected', False)
            threat_type = result.get('threat_type', 'Unknown')
            performance_level = result.get('performance_level', 'unknown')
            visual_analysis = result.get('visual_analysis', 'No analysis')
            reasoning = result.get('reasoning', 'No reasoning')
            
            # Competition logging
            print(f"📊 Analysis Result:")
            print(f"   Threat: {threat_detected}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Type: {threat_type}")
            print(f"   Performance: {performance_level}")
            print(f"   Time: {analysis_time:.2f}s")
            print(f"   Visual: {visual_analysis[:50]}...")
            
            # Track performance level usage
            if performance_level != "unknown":
                self.stats["performance_switches"] += 1
            
            # Check threat threshold
            sensitivity = self._get_threat_sensitivity(threat_type)
            
            if threat_detected and confidence >= sensitivity:
                self._handle_threat_detection(frame, result, analysis_time)
            else:
                print(f"✅ Normal activity: {reasoning[:50]}...")
                
        except Exception as e:
            print(f"❌ Result processing error: {e}")
    
    def _get_threat_sensitivity(self, threat_type: str) -> float:
        """Get sensitivity threshold for threat type"""
        default_sensitivity = 0.7
        
        sensitivity_map = {
            "Violence": 0.6,
            "Fall": 0.8,
            "Bullying": 0.7,
            "Distress": 0.6
        }
        
        return sensitivity_map.get(threat_type, default_sensitivity)
    
    def _handle_threat_detection(self, frame, result: Dict[str, Any], analysis_time: float):
        """Handle confirmed threat detection"""
        try:
            self.stats["threats_detected"] += 1
            
            threat_type = result.get('threat_type', 'Unknown')
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', 'No reasoning')
            
            print(f"🚨 THREAT DETECTED #{self.stats['threats_detected']}:")
            print(f"   Type: {threat_type}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Analysis Time: {analysis_time:.2f}s")
            print(f"   Reasoning: {reasoning}")
            
            # Save evidence
            evidence_path = self._save_threat_evidence(frame, result)
            
            # Add to threat history
            threat_record = {
                "timestamp": datetime.now().isoformat(),
                "type": threat_type,
                "confidence": confidence,
                "analysis_time": analysis_time,
                "evidence_path": evidence_path,
                "reasoning": reasoning,
                "performance_level": result.get('performance_level', 'unknown')
            }
            
            self.stats["threat_history"].append(threat_record)
            
            # Trigger callback
            if self.threat_callback:
                self.threat_callback(threat_type, confidence, reasoning)
                
        except Exception as e:
            print(f"❌ Threat handling error: {e}")
    
    def _save_threat_evidence(self, frame, result: Dict[str, Any]) -> str:
        """Save threat frame and analysis as evidence"""
        try:
            # Create evidence directory
            evidence_dir = Path("logs/evidence")
            evidence_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            threat_type = result.get('threat_type', 'Unknown').replace(' ', '_')
            confidence = result.get('confidence', 0.0)
            performance = result.get('performance_level', 'unknown')
            
            base_filename = f"threat_{timestamp}_{threat_type}_{confidence:.2f}_{performance}"
            
            # Save frame with overlay
            frame_with_overlay = self._add_threat_overlay(frame, result)
            image_path = evidence_dir / f"{base_filename}.jpg"
            cv2.imwrite(str(image_path), frame_with_overlay)
            
            # Save analysis JSON
            json_path = evidence_dir / f"{base_filename}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"💾 Evidence saved: {base_filename}")
            return str(image_path)
            
        except Exception as e:
            print(f"❌ Failed to save evidence: {e}")
            return ""
    
    def _add_threat_overlay(self, frame, result: Dict[str, Any]):
        """Add threat detection overlay to frame"""
        try:
            overlay_frame = frame.copy()
            
            # Threat info
            threat_type = result.get('threat_type', 'Unknown')
            confidence = result.get('confidence', 0.0)
            performance = result.get('performance_level', 'unknown')
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Add red border for threat
            cv2.rectangle(overlay_frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 5)
            
            # Add threat information
            cv2.putText(overlay_frame, f"THREAT DETECTED: {threat_type}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.putText(overlay_frame, f"Confidence: {confidence:.1%}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.putText(overlay_frame, f"Performance: {performance}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(overlay_frame, f"Time: {timestamp}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return overlay_frame
            
        except Exception as e:
            print(f"Overlay error: {e}")
            return frame
    
    def _update_fps_counter(self):
        """Update FPS calculation for competition metrics"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_last_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_last_time)
            self.stats["avg_fps"] = self.current_fps
            self.fps_counter = 0
            self.fps_last_time = current_time
    
    def get_competition_stats(self) -> Dict[str, Any]:
        """Get comprehensive competition statistics"""
        stats = self.stats.copy()
        
        # Add calculated metrics
        if self.stats["frames_analyzed"] > 0:
            avg_analysis_time = sum(self.stats["analysis_times"]) / len(self.stats["analysis_times"])
            stats["avg_analysis_time"] = avg_analysis_time
            
            # Detection rate
            stats["detection_rate"] = (self.stats["threats_detected"] / self.stats["frames_analyzed"]) * 100
        
        # Add AI engine performance stats if available
        if self.ai_engine:
            ai_stats = self.ai_engine.get_performance_stats()
            stats["ai_performance"] = ai_stats
        
        return stats
    
    def show_competition_demo(self, duration: int = 60):
        """Show live competition demo with real-time metrics"""
        if not self.is_monitoring:
            print("⚠️ Start monitoring first")
            return
        
        print(f"📺 Competition Demo Mode - {duration} seconds")
        print("Press 'q' to quit, 's' for snapshot, 'p' to force performance level")
        
        start_time = time.time()
        forced_performance = None
        
        while time.time() - start_time < duration:
            if self.current_frame is not None:
                # Create demo display frame
                demo_frame = self._create_demo_frame(self.current_frame)
                
                cv2.imshow('WARYON Competition Demo', demo_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._take_demo_snapshot()
                elif key == ord('p'):
                    forced_performance = self._cycle_performance_level(forced_performance)
            
            time.sleep(0.033)  # ~30 FPS display
        
        cv2.destroyAllWindows()
        print("📺 Competition demo ended")
    
    def _create_demo_frame(self, frame):
        """Create demo frame with competition metrics overlay"""
        try:
            demo_frame = frame.copy()
            
            # Add status overlay
            cv2.putText(demo_frame, f"WARYON Competition Demo", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(demo_frame, f"FPS: {self.current_fps:.1f}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(demo_frame, f"Frames Analyzed: {self.stats['frames_analyzed']}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(demo_frame, f"Threats Detected: {self.stats['threats_detected']}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Add AI performance info if available
            if self.ai_engine:
                ai_stats = self.ai_engine.get_performance_stats()
                cv2.putText(demo_frame, f"AI Analyses: {ai_stats.get('total_analyses', 0)}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show performance distribution
                dist = ai_stats.get('performance_distribution', {})
                y_pos = 130
                for level, percentage in dist.items():
                    cv2.putText(demo_frame, f"{level}: {percentage:.1f}%", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_pos += 20
            
            return demo_frame
            
        except Exception as e:
            print(f"Demo frame error: {e}")
            return frame
    
    def _take_demo_snapshot(self):
        """Take demo snapshot for competition presentation"""
        try:
            if self.current_frame is None:
                return
            
            snapshots_dir = Path("logs/snapshots")
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"demo_snapshot_{timestamp}.jpg"
            filepath = snapshots_dir / filename
            
            # Add demo overlay
            demo_frame = self._create_demo_frame(self.current_frame)
            cv2.imwrite(str(filepath), demo_frame)
            
            print(f"📸 Demo snapshot saved: {filename}")
            
        except Exception as e:
            print(f"❌ Snapshot failed: {e}")
    
    def _cycle_performance_level(self, current_forced):
        """Cycle through performance levels for demo"""
        levels = ["2b_efficient", "3b_balanced", "4b_full", None]
        
        if current_forced is None:
            next_level = levels[0]
        else:
            try:
                current_index = levels.index(current_forced)
                next_level = levels[(current_index + 1) % len(levels)]
            except ValueError:
                next_level = levels[0]
        
        if next_level is None:
            self.ai_engine.reset_performance_forcing()
            print("🔧 Performance forcing disabled (auto mode)")
        else:
            self.ai_engine.force_performance_level(next_level)
            print(f"🔧 Performance forced to: {next_level}")
        
        return next_level


# Test the competition video processor
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("📹 WARYON Competition Video Processor Test")
    print("=" * 50)
    
    # Mock config for testing
    class MockConfig:
        def get(self, key, default=None):
            return default
    
    def threat_callback(threat_type, confidence, details):
        print(f"🚨 COMPETITION THREAT DETECTED:")
        print(f"   Type: {threat_type}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Details: {details[:100]}...")
    
    # Initialize processor
    config = MockConfig()
    processor = CompetitionVideoProcessor(config, threat_callback)
    
    # Test camera
    if processor.test_camera():
        print("✅ Camera ready for competition")
        
        # Mock AI engine for testing
        class MockAI:
            def analyze_multimodal_threat(self, image, audio_description="", performance_level="auto"):
                import random
                return {
                    'threat_detected': random.choice([True, False]),
                    'confidence': random.uniform(0.3, 0.9),
                    'threat_type': random.choice(['Violence', 'Normal Activity', 'Fall']),
                    'performance_level': performance_level if performance_level != "auto" else random.choice(['2b_efficient', '3b_balanced', '4b_full']),
                    'visual_analysis': 'Mock visual analysis for competition demo',
                    'reasoning': 'Mock reasoning for demonstration purposes'
                }
            
            def get_performance_stats(self):
                return {
                    'total_analyses': 10,
                    'performance_distribution': {'2b_efficient': 60, '3b_balanced': 30, '4b_full': 10}
                }
            
            def force_performance_level(self, level):
                print(f"Mock: Forced to {level}")
            
            def reset_performance_forcing(self):
                print("Mock: Reset to auto")
        
        processor.set_ai_engine(MockAI())
        
        # Start monitoring
        if processor.start_monitoring():
            print("✅ Competition monitoring started")
            
            try:
                # Run demo
                processor.show_competition_demo(30)
                
                # Show final stats
                stats = processor.get_competition_stats()
                print(f"\n📊 Competition Statistics:")
                for key, value in stats.items():
                    if key not in ['analysis_times', 'threat_history']:
                        print(f"  {key}: {value}")
                
            except KeyboardInterrupt:
                print("\n⏹️ Demo interrupted")
            
            finally:
                processor.stop_monitoring()
        
        else:
            print("❌ Failed to start monitoring")
    
    else:
        print("❌ Camera test failed")
    
    print("\n🏆 Competition video processor test complete")