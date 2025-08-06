"""
WARYON: Competition-Ready AI Guardian
Main Application - Dynamic Gemma 3n Implementation
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import our competition-ready modules
try:
    from ai_engine import GemmaDynamicAI
    from video_processor import CompetitionVideoProcessor
    from audio_processor import RealAudioProcessor
    from config_manager import ConfigManager
except ImportError as e:
    print(f"❌ Missing module: {e}")
    print("Make sure all source files are in the src/ directory")

class WaryonCompetitionApp:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("WARYON - Competition AI Guardian")
        self.root.geometry("1000x800")
        self.root.configure(bg='#1a1a2e')
        
        # Competition theme colors
        self.colors = {
            'bg': '#1a1a2e',
            'primary': '#16213e',
            'accent': '#0f4c75',
            'success': '#00ff88',
            'warning': '#ffaa00',
            'danger': '#ff0066',
            'text': '#ffffff',
            'text_dim': '#cccccc'
        }
        
        # Initialize configuration
        self.config = ConfigManager()
        
        # Initialize competition state
        self.is_monitoring = False
        self.start_time = None
        self.competition_stats = {
            'total_runtime': 0,
            'threats_detected': 0,
            'performance_switches': 0,
            'ai_analyses': 0,
            'avg_response_time': 0.0
        }
        
        # Initialize core components
        self.ai_engine = None
        self.video_processor = None
        self.audio_processor = None
        
        # Setup UI
        self.setup_competition_ui()
        
        # Initialize components
        self.initialize_competition_components()
        
        # Start statistics updater
        self.start_stats_updater()
    
    def setup_competition_ui(self):
        """Setup competition-ready user interface"""
        # Title section
        self.create_title_section()
        
        # Main content area
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Controls and status
        left_panel = tk.Frame(main_frame, bg=self.colors['primary'], relief='ridge', bd=2)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.config(width=300)
        left_panel.pack_propagate(False)
        
        self.create_control_panel(left_panel)
        
        # Right panel - Statistics and metrics
        right_panel = tk.Frame(main_frame, bg=self.colors['bg'])
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.create_statistics_panel(right_panel)
    
    def create_title_section(self):
        """Create competition title section"""
        title_frame = tk.Frame(self.root, bg=self.colors['bg'])
        title_frame.pack(fill='x', padx=20, pady=20)
        
        # Main title
        title_label = tk.Label(
            title_frame,
            text="🛡️ WARYON",
            font=("Arial", 32, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['success']
        )
        title_label.pack()
        
        # Competition subtitle
        subtitle_label = tk.Label(
            title_frame,
            text="Competition AI Guardian • Powered by Gemma 3n Dynamic Intelligence",
            font=("Arial", 12),
            bg=self.colors['bg'],
            fg=self.colors['text_dim']
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Features highlight
        features_label = tk.Label(
            title_frame,
            text="Dynamic 2B→4B Scaling • Multimodal Threat Detection • 24/7 Privacy-First Monitoring",
            font=("Arial", 10, "italic"),
            bg=self.colors['bg'],
            fg=self.colors['accent']
        )
        features_label.pack(pady=(5, 0))
    
    def create_control_panel(self, parent):
        """Create competition control panel"""
        # Panel title
        tk.Label(
            parent,
            text="🎮 Competition Controls",
            font=("Arial", 14, "bold"),
            bg=self.colors['primary'],
            fg=self.colors['text']
        ).pack(pady=15)
        
        # System status
        status_frame = tk.LabelFrame(
            parent,
            text=" System Status ",
            font=("Arial", 11, "bold"),
            bg=self.colors['primary'],
            fg=self.colors['text']
        )
        status_frame.pack(fill='x', padx=15, pady=10)
        
        self.status_indicators = {}
        
        components = [
            ("AI Engine", "ai_status"),
            ("Camera", "camera_status"),
            ("Audio", "audio_status"),
            ("Monitoring", "monitor_status")
        ]
        
        for i, (name, key) in enumerate(components):
            frame = tk.Frame(status_frame, bg=self.colors['primary'])
            frame.pack(fill='x', padx=10, pady=5)
            
            tk.Label(
                frame,
                text=f"{name}:",
                bg=self.colors['primary'],
                fg=self.colors['text_dim'],
                width=10,
                anchor='w'
            ).pack(side='left')
            
            status_label = tk.Label(
                frame,
                text="⚪ Checking...",
                bg=self.colors['primary'],
                fg=self.colors['warning'],
                anchor='w'
            )
            status_label.pack(side='left')
            
            self.status_indicators[key] = status_label
        
        # Main control button
        self.main_control_button = tk.Button(
            parent,
            text="🚀 Start Competition Demo",
            font=("Arial", 14, "bold"),
            bg=self.colors['success'],
            fg='black',
            padx=20,
            pady=15,
            command=self.toggle_monitoring,
            relief='raised',
            bd=3
        )
        self.main_control_button.pack(pady=20)
        
        # Performance controls
        perf_frame = tk.LabelFrame(
            parent,
            text=" Performance Controls ",
            font=("Arial", 11, "bold"),
            bg=self.colors['primary'],
            fg=self.colors['text']
        )
        perf_frame.pack(fill='x', padx=15, pady=10)
        
        # Performance level selector
        tk.Label(
            perf_frame,
            text="Force Performance:",
            bg=self.colors['primary'],
            fg=self.colors['text_dim']
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        self.performance_var = tk.StringVar(value="auto")
        perf_combo = ttk.Combobox(
            perf_frame,
            textvariable=self.performance_var,
            values=["auto", "2b_efficient", "3b_balanced", "4b_full"],
            state="readonly",
            width=15
        )
        perf_combo.pack(padx=10, pady=(0, 10))
        perf_combo.bind("<<ComboboxSelected>>", self.on_performance_change)
        
        # Competition actions
        actions_frame = tk.LabelFrame(
            parent,
            text=" Demo Actions ",
            font=("Arial", 11, "bold"),
            bg=self.colors['primary'],
            fg=self.colors['text']
        )
        actions_frame.pack(fill='x', padx=15, pady=10)
        
        actions = [
            ("📸 Take Screenshot", self.take_screenshot),
            ("🧪 Test Threat Detection", self.test_threat_detection),
            ("📊 Show Live Metrics", self.show_live_metrics),
            ("💾 Export Statistics", self.export_statistics)
        ]
        
        for text, command in actions:
            btn = tk.Button(
                actions_frame,
                text=text,
                command=command,
                bg=self.colors['accent'],
                fg=self.colors['text'],
                width=20,
                pady=5
            )
            btn.pack(pady=3, padx=10)
    
    def create_statistics_panel(self, parent):
        """Create competition statistics panel"""
        # Statistics title
        tk.Label(
            parent,
            text="📊 Competition Metrics",
            font=("Arial", 16, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text']
        ).pack(pady=(0, 15))
        
        # Performance metrics
        perf_frame = tk.LabelFrame(
            parent,
            text=" Performance Metrics ",
            font=("Arial", 12, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text'],
            relief='ridge',
            bd=2
        )
        perf_frame.pack(fill='x', pady=(0, 15))
        
        # Create metrics grid
        metrics_grid = tk.Frame(perf_frame, bg=self.colors['bg'])
        metrics_grid.pack(fill='x', padx=15, pady=15)
        
        # Configure grid
        for i in range(3):
            metrics_grid.columnconfigure(i, weight=1)
        
        # Metrics displays
        self.metric_displays = {}
        
        metrics_config = [
            ("Runtime", "runtime", "00:00:00", 0, 0),
            ("Threats Detected", "threats", "0", 0, 1),
            ("AI Analyses", "analyses", "0", 0, 2),
            ("Avg Response Time", "response_time", "0.0s", 1, 0),
            ("Performance Switches", "switches", "0", 1, 1),
            ("Current FPS", "fps", "0.0", 1, 2)
        ]
        
        for name, key, default, row, col in metrics_config:
            metric_frame = tk.Frame(
                metrics_grid, 
                bg=self.colors['primary'], 
                relief='ridge', 
                bd=1
            )
            metric_frame.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            
            tk.Label(
                metric_frame,
                text=name,
                font=("Arial", 9),
                bg=self.colors['primary'],
                fg=self.colors['text_dim']
            ).pack(pady=(8, 2))
            
            value_label = tk.Label(
                metric_frame,
                text=default,
                font=("Arial", 14, "bold"),
                bg=self.colors['primary'],
                fg=self.colors['success']
            )
            value_label.pack(pady=(0, 8))
            
            self.metric_displays[key] = value_label
        
        # Performance distribution
        dist_frame = tk.LabelFrame(
            parent,
            text=" Dynamic Performance Distribution ",
            font=("Arial", 12, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text'],
            relief='ridge',
            bd=2
        )
        dist_frame.pack(fill='x', pady=(0, 15))
        
        # Performance bars
        self.performance_bars = {}
        
        performance_levels = [
            ("2B Efficient", "2b_efficient", self.colors['success']),
            ("3B Balanced", "3b_balanced", self.colors['warning']),
            ("4B Full Power", "4b_full", self.colors['danger'])
        ]
        
        for name, key, color in performance_levels:
            bar_frame = tk.Frame(dist_frame, bg=self.colors['bg'])
            bar_frame.pack(fill='x', padx=15, pady=5)
            
            # Label
            tk.Label(
                bar_frame,
                text=name,
                font=("Arial", 10),
                bg=self.colors['bg'],
                fg=self.colors['text'],
                width=15,
                anchor='w'
            ).pack(side='left')
            
            # Progress bar frame
            progress_frame = tk.Frame(bar_frame, bg=self.colors['primary'], height=20)
            progress_frame.pack(side='left', fill='x', expand=True, padx=(10, 10))
            
            # Progress bar
            progress_bar = tk.Frame(progress_frame, bg=color, height=18)
            progress_bar.place(x=1, y=1, width=0, height=18)
            
            # Percentage label
            percent_label = tk.Label(
                bar_frame,
                text="0%",
                font=("Arial", 10),
                bg=self.colors['bg'],
                fg=self.colors['text'],
                width=5,
                anchor='e'
            )
            percent_label.pack(side='right')
            
            self.performance_bars[key] = {
                'bar': progress_bar,
                'label': percent_label,
                'frame': progress_frame
            }
        
        # Recent threats
        threats_frame = tk.LabelFrame(
            parent,
            text=" Recent Threat Detections ",
            font=("Arial", 12, "bold"),
            bg=self.colors['bg'],
            fg=self.colors['text'],
            relief='ridge',
            bd=2
        )
        threats_frame.pack(fill='both', expand=True)
        
        # Threats list with scrollbar
        threats_container = tk.Frame(threats_frame, bg=self.colors['bg'])
        threats_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(threats_container)
        scrollbar.pack(side='right', fill='y')
        
        # Threats listbox
        self.threats_listbox = tk.Listbox(
            threats_container,
            bg=self.colors['primary'],
            fg=self.colors['text'],
            font=("Consolas", 9),
            yscrollcommand=scrollbar.set,
            selectbackground=self.colors['accent']
        )
        self.threats_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.threats_listbox.yview)
        
        # Initial threat list entry
        self.threats_listbox.insert('end', "🛡️ No threats detected - System ready for demonstration")
    
    def initialize_competition_components(self):
        """Initialize competition-ready components"""
        try:
            # Initialize AI Engine
            self.status_indicators['ai_status'].config(text="🟡 Initializing...", fg=self.colors['warning'])
            self.root.update()
            
            self.ai_engine = GemmaDynamicAI(self.config)
            
            if self.ai_engine.connection_status:
                self.status_indicators['ai_status'].config(text="🟢 Ready", fg=self.colors['success'])
            else:
                self.status_indicators['ai_status'].config(text="🔴 Failed", fg=self.colors['danger'])
                messagebox.showerror("AI Engine Error", "Failed to initialize Gemma 3n AI engine")
                return
            
            # Initialize Video Processor
            self.status_indicators['camera_status'].config(text="🟡 Testing...", fg=self.colors['warning'])
            self.root.update()
            
            self.video_processor = CompetitionVideoProcessor(self.config, self.on_threat_detected)
            self.video_processor.set_ai_engine(self.ai_engine)
            
            if self.video_processor.test_camera():
                self.status_indicators['camera_status'].config(text="🟢 Ready", fg=self.colors['success'])
            else:
                self.status_indicators['camera_status'].config(text="🔴 Failed", fg=self.colors['danger'])
                messagebox.showerror("Camera Error", "Failed to initialize camera system")
                return
            
            # Initialize Audio Processor
            self.status_indicators['audio_status'].config(text="🟡 Testing...", fg=self.colors['warning'])
            self.root.update()
            
            self.audio_processor = RealAudioProcessor(self.config, self.on_threat_detected)
            
            if self.audio_processor.test_audio():
                self.status_indicators['audio_status'].config(text="🟢 Ready", fg=self.colors['success'])
                # Connect audio processor to video processor
                self.video_processor.set_audio_processor(self.audio_processor)
            else:
                self.status_indicators['audio_status'].config(text="🟡 Limited", fg=self.colors['warning'])
                print("⚠️ Audio processor has limited functionality")
            
            self.status_indicators['monitor_status'].config(text="🔴 Stopped", fg=self.colors['danger'])
            
            print("✅ Competition components initialized successfully")
            
        except Exception as e:
            print(f"❌ Component initialization error: {e}")
            messagebox.showerror("Initialization Error", f"Failed to initialize components: {e}")
    
    def toggle_monitoring(self):
        """Toggle competition monitoring"""
        if not self.is_monitoring:
            self.start_competition_monitoring()
        else:
            self.stop_competition_monitoring()
    
    def start_competition_monitoring(self):
        """Start competition monitoring demo"""
        try:
            if not self.ai_engine or not self.video_processor:
                messagebox.showerror("Error", "Components not initialized")
                return
            
            # Start audio monitoring if available
            if self.audio_processor:
                if not self.audio_processor.start_monitoring():
                    print("⚠️ Audio monitoring failed to start")
            
            if self.video_processor.start_monitoring():
                self.is_monitoring = True
                self.start_time = datetime.now()
                
                # Update UI
                self.status_indicators['monitor_status'].config(text="🟢 Active", fg=self.colors['success'])
                self.main_control_button.config(
                    text="⏹️ Stop Demo",
                    bg=self.colors['danger']
                )
                
                # Add to threats list
                audio_status = "with REAL audio" if self.audio_processor else "with mock audio"
                self.add_threat_entry(f"🚀 Competition monitoring started - Dynamic Gemma 3n active {audio_status}")
                
                print("✅ Competition monitoring started")
                
            else:
                messagebox.showerror("Error", "Failed to start monitoring")
                
        except Exception as e:
            print(f"❌ Failed to start monitoring: {e}")
            messagebox.showerror("Error", f"Failed to start monitoring: {e}")
    
    def stop_competition_monitoring(self):
        """Stop competition monitoring"""
        try:
            if self.video_processor:
                self.video_processor.stop_monitoring()
            
            if self.audio_processor:
                self.audio_processor.stop_monitoring()
            
            self.is_monitoring = False
            
            # Update UI
            self.status_indicators['monitor_status'].config(text="🔴 Stopped", fg=self.colors['danger'])
            self.main_control_button.config(
                text="🚀 Start Competition Demo",
                bg=self.colors['success']
            )
            
            # Add to threats list
            self.add_threat_entry("⏹️ Competition monitoring stopped")
            
            print("✅ Competition monitoring stopped")
            
        except Exception as e:
            print(f"❌ Error stopping monitoring: {e}")
            self.colors['success']

            # Add to threats list
            self.add_threat_entry("⏹️ Competition monitoring stopped")
            
            print("✅ Competition monitoring stopped")
            
        except Exception as e:
            print(f"❌ Error stopping monitoring: {e}")
            self.colors['success']

            # Add to threats list
            self.add_threat_entry("⏹️ Competition monitoring stopped")
            
            print("✅ Competition monitoring stopped")
            
        except Exception as e:
            print(f"❌ Error stopping monitoring: {e}")
    
    def on_threat_detected(self, threat_type, confidence, reasoning):
        """Handle threat detection for competition demo"""
        try:
            self.competition_stats['threats_detected'] += 1
            
            # Update threats display
            timestamp = datetime.now().strftime("%H:%M:%S")
            threat_entry = f"🚨 {timestamp} - {threat_type} ({confidence:.1%}) - {reasoning[:50]}..."
            
            self.add_threat_entry(threat_entry)
            
            # Flash the threats counter
            self.flash_metric('threats')
            
            print(f"🚨 Competition threat logged: {threat_type}")
            
        except Exception as e:
            print(f"❌ Error handling threat: {e}")
    
    def add_threat_entry(self, entry):
        """Add entry to threats list"""
        try:
            self.threats_listbox.insert('end', entry)
            self.threats_listbox.see('end')
            
            # Keep only last 50 entries
            if self.threats_listbox.size() > 50:
                self.threats_listbox.delete(0)
                
        except Exception as e:
            print(f"Error adding threat entry: {e}")
    
    def flash_metric(self, metric_key):
        """Flash a metric display for visual feedback"""
        try:
            if metric_key in self.metric_displays:
                original_bg = self.metric_displays[metric_key]['bg']
                
                # Flash red
                self.metric_displays[metric_key].config(bg=self.colors['danger'])
                self.root.after(200, lambda: self.metric_displays[metric_key].config(bg=original_bg))
                
        except Exception as e:
            print(f"Flash error: {e}")
    
    def on_performance_change(self, event=None):
        """Handle performance level change"""
        try:
            level = self.performance_var.get()
            
            if self.ai_engine:
                if level == "auto":
                    self.ai_engine.reset_performance_forcing()
                    self.add_threat_entry(f"⚙️ Performance set to AUTO (dynamic scaling)")
                else:
                    self.ai_engine.force_performance_level(level)
                    self.add_threat_entry(f"⚙️ Performance forced to {level.upper()}")
                
        except Exception as e:
            print(f"Performance change error: {e}")
    
    def take_screenshot(self):
        """Take competition screenshot"""
        try:
            if self.video_processor and self.video_processor.current_frame is not None:
                # Take demo snapshot
                self.video_processor._take_demo_snapshot()
                self.add_threat_entry("📸 Competition screenshot captured")
            else:
                messagebox.showwarning("Screenshot", "No video feed available")
                
        except Exception as e:
            print(f"Screenshot error: {e}")
    
    def test_threat_detection(self):
        """Test threat detection for competition demo"""
        try:
            if not self.ai_engine or not self.video_processor:
                messagebox.showwarning("Test", "Monitoring not active")
                return
            
            # Simulate threat detection test
            self.add_threat_entry("🧪 Testing threat detection capabilities...")
            
            # Show test dialog
            test_window = tk.Toplevel(self.root)
            test_window.title("Threat Detection Test")
            test_window.geometry("400x300")
            test_window.configure(bg=self.colors['bg'])
            
            tk.Label(
                test_window,
                text="🧪 Competition Threat Detection Test",
                font=("Arial", 14, "bold"),
                bg=self.colors['bg'],
                fg=self.colors['text']
            ).pack(pady=20)
            
            tk.Label(
                test_window,
                text="Show different expressions to the camera:\n\n• Normal expression\n• Fearful expression\n• Surprised expression\n• Distressed expression",
                font=("Arial", 11),
                bg=self.colors['bg'],
                fg=self.colors['text_dim'],
                justify='left'
            ).pack(pady=20)
            
            tk.Button(
                test_window,
                text="Close Test",
                command=test_window.destroy,
                bg=self.colors['accent'],
                fg=self.colors['text'],
                padx=20,
                pady=10
            ).pack(pady=20)
            
        except Exception as e:
            print(f"Test error: {e}")
    
    def show_live_metrics(self):
        """Show live competition metrics"""
        try:
            if self.video_processor:
                # Start live demo window
                self.video_processor.show_competition_demo(60)
            else:
                messagebox.showwarning("Live Metrics", "Video processor not available")
                
        except Exception as e:
            print(f"Live metrics error: {e}")
    
    def export_statistics(self):
        """Export competition statistics"""
        try:
            stats = self.get_comprehensive_stats()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"waryon_competition_stats_{timestamp}.json"
            filepath = Path("logs") / filename
            
            filepath.parent.mkdir(exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            self.add_threat_entry(f"📊 Statistics exported: {filename}")
            messagebox.showinfo("Export", f"Statistics exported to {filename}")
            
        except Exception as e:
            print(f"Export error: {e}")
            messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def start_stats_updater(self):
        """Start competition statistics updater"""
        self.update_competition_stats()
    
    def update_competition_stats(self):
        """Update competition statistics display"""
        try:
            # Update runtime
            if self.is_monitoring and self.start_time:
                runtime = datetime.now() - self.start_time
                hours, remainder = divmod(runtime.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                runtime_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                self.metric_displays['runtime'].config(text=runtime_str)
            
            # Update other metrics
            if self.video_processor:
                video_stats = self.video_processor.get_competition_stats()
                
                self.metric_displays['threats'].config(text=str(video_stats.get('threats_detected', 0)))
                self.metric_displays['fps'].config(text=f"{video_stats.get('avg_fps', 0.0):.1f}")
                
                if 'avg_analysis_time' in video_stats:
                    self.metric_displays['response_time'].config(text=f"{video_stats['avg_analysis_time']:.1f}s")
            
            if self.ai_engine:
                ai_stats = self.ai_engine.get_performance_stats()
                
                self.metric_displays['analyses'].config(text=str(ai_stats.get('total_analyses', 0)))
                
                # Update performance distribution
                dist = ai_stats.get('performance_distribution', {})
                
                for level, data in self.performance_bars.items():
                    percentage = dist.get(level, 0)
                    
                    # Update bar width
                    frame_width = data['frame'].winfo_width()
                    if frame_width > 0:
                        bar_width = int((percentage / 100) * (frame_width - 2))
                        data['bar'].place(width=bar_width)
                    
                    # Update percentage label
                    data['label'].config(text=f"{percentage:.1f}%")
            
            # Schedule next update
            self.root.after(1000, self.update_competition_stats)
            
        except Exception as e:
            print(f"Stats update error: {e}")
            self.root.after(5000, self.update_competition_stats)  # Retry in 5 seconds
    
    def get_comprehensive_stats(self) -> dict:
        """Get comprehensive competition statistics"""
        stats = {
            "competition_info": {
                "app_name": "WARYON AI Guardian",
                "version": "Competition 1.0",
                "technology": "Gemma 3n Dynamic Scaling",
                "timestamp": datetime.now().isoformat()
            },
            "runtime_stats": self.competition_stats.copy(),
            "video_stats": {},
            "ai_stats": {}
        }
        
        if self.video_processor:
            stats["video_stats"] = self.video_processor.get_competition_stats()
        
        if self.ai_engine:
            stats["ai_stats"] = self.ai_engine.get_performance_stats()
        
        return stats
    
    def run(self):
        """Start the competition application"""
        try:
            print("🏆 Starting WARYON Competition Application")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("⏹️ Application interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup competition resources"""
        try:
            if self.is_monitoring:
                self.stop_competition_monitoring()
            
            print("🧹 Competition cleanup completed")
            
        except Exception as e:
            print(f"Cleanup error: {e}")


def main():
    """Main competition entry point"""
    try:
        print("🏆 WARYON Competition Application Starting...")
        print("=" * 50)
        
        # Create and run application
        app = WaryonCompetitionApp()
        app.run()
        
    except Exception as e:
        print(f"❌ Failed to start competition application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()