"""
Personalized WARYON Extensions
Custom Words, Visual Signs, and Personal Threat Patterns
Add these to your existing audio_processor.py and create new personalization_manager.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

class PersonalizationManager:
    """Manages personalized settings for individual users"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.user_profile_path = Path("config/user_profile.json")
        self.user_profile_path.parent.mkdir(exist_ok=True)
        
        # Load personalized settings
        self.personal_settings = self._load_personal_settings()
        
        print("👤 Personalization Manager initialized")
        print(f"   Custom words: {len(self.personal_settings.get('custom_distress_words', []))}")
        print(f"   Custom gestures: {len(self.personal_settings.get('custom_gestures', []))}")
    
    def _load_personal_settings(self) -> Dict[str, Any]:
        """Load user's personal settings"""
        if self.user_profile_path.exists():
            try:
                with open(self.user_profile_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading personal settings: {e}")
        
        return self._default_personal_settings()
    
    def _default_personal_settings(self) -> Dict[str, Any]:
        """Default personalized settings"""
        return {
            "user_info": {
                "name": "",
                "nickname": "",
                "family_names": [],
                "emergency_phrase": "",
                "safe_word": "",
                "created": datetime.now().isoformat()
            },
            "custom_distress_words": {
                "emergency_phrases": [],
                "family_calls": [],
                "personal_help_words": [],
                "danger_signals": [],
                "medical_terms": []
            },
            "custom_gestures": {
                "hand_signals": [],
                "facial_expressions": [],
                "body_movements": [],
                "emergency_gestures": []
            },
            "personal_threat_patterns": {
                "specific_fears": [],
                "trigger_situations": [],
                "escalation_patterns": [],
                "safe_contexts": []
            },
            "personalization_level": "basic"  # basic, intermediate, advanced
        }
    
    def save_personal_settings(self) -> bool:
        """Save personal settings to file"""
        try:
            with open(self.user_profile_path, 'w') as f:
                json.dump(self.personal_settings, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving personal settings: {e}")
            return False
    
    def setup_personalization_wizard(self, parent_window=None):
        """Launch personalization setup wizard"""
        wizard = PersonalizationWizard(self, parent_window)
        wizard.run()
    
    def get_custom_distress_words(self) -> Dict[str, List[str]]:
        """Get user's custom distress words"""
        return self.personal_settings.get("custom_distress_words", {})
    
    def get_custom_gestures(self) -> Dict[str, List[Dict]]:
        """Get user's custom gestures"""
        return self.personal_settings.get("custom_gestures", {})
    
    def add_custom_word(self, category: str, word: str, context: str = "") -> bool:
        """Add a custom distress word"""
        try:
            if category not in self.personal_settings["custom_distress_words"]:
                self.personal_settings["custom_distress_words"][category] = []
            
            word_entry = {
                "word": word.lower(),
                "context": context,
                "added": datetime.now().isoformat(),
                "usage_count": 0
            }
            
            self.personal_settings["custom_distress_words"][category].append(word_entry)
            return self.save_personal_settings()
            
        except Exception as e:
            print(f"❌ Error adding custom word: {e}")
            return False
    
    def add_custom_gesture(self, category: str, gesture_name: str, description: str, 
                          visual_cues: List[str], confidence_threshold: float = 0.7) -> bool:
        """Add a custom visual gesture"""
        try:
            if category not in self.personal_settings["custom_gestures"]:
                self.personal_settings["custom_gestures"][category] = []
            
            gesture_entry = {
                "name": gesture_name,
                "description": description,
                "visual_cues": visual_cues,
                "confidence_threshold": confidence_threshold,
                "added": datetime.now().isoformat(),
                "detection_count": 0,
                "enabled": True
            }
            
            self.personal_settings["custom_gestures"][category].append(gesture_entry)
            return self.save_personal_settings()
            
        except Exception as e:
            print(f"❌ Error adding custom gesture: {e}")
            return False
    
    def get_all_distress_words_combined(self) -> Dict[str, List[str]]:
        """Get combined default + custom distress words"""
        # Default words
        default_words = {
            "emergency": ["help", "emergency", "fire", "police", "ambulance", "911", "urgent", "crisis"],
            "violence": ["stop", "don't", "hurt", "pain", "hit", "attack", "fight", "violence", "hitting"],
            "fear": ["scared", "afraid", "terrified", "please", "no", "get away", "leave me alone"],
            "medical": ["sick", "heart", "can't breathe", "chest pain", "stroke", "collapsed", "dying"],
            "distress": ["crying", "sobbing", "screaming", "panicking", "desperate", "trapped"]
        }
        
        # Add custom words
        custom_words = self.get_custom_distress_words()
        
        combined = default_words.copy()
        
        for category, word_entries in custom_words.items():
            if category not in combined:
                combined[category] = []
            
            for entry in word_entries:
                if isinstance(entry, dict):
                    combined[category].append(entry["word"])
                else:
                    combined[category].append(entry)
        
        return combined
    
    def create_personal_gemma_prompt_additions(self) -> str:
        """Create personalized additions to Gemma 3n prompts"""
        user_info = self.personal_settings.get("user_info", {})
        custom_words = self.get_custom_distress_words()
        custom_gestures = self.get_custom_gestures()
        
        prompt_additions = []
        
        # User context
        if user_info.get("name"):
            prompt_additions.append(f"USER CONTEXT: This person is {user_info['name']}")
            
            if user_info.get("family_names"):
                family_names = ", ".join(user_info["family_names"])
                prompt_additions.append(f"FAMILY CONTEXT: Family members include {family_names}")
        
        # Emergency phrase
        if user_info.get("emergency_phrase"):
            prompt_additions.append(f"PERSONAL EMERGENCY PHRASE: '{user_info['emergency_phrase']}' is this person's specific emergency call for help")
        
        # Safe word
        if user_info.get("safe_word"):
            prompt_additions.append(f"PERSONAL SAFE WORD: '{user_info['safe_word']}' means this person is safe and cancels any alerts")
        
        # Custom distress words
        if custom_words:
            custom_word_desc = []
            for category, word_entries in custom_words.items():
                if word_entries:
                    words = [entry["word"] if isinstance(entry, dict) else entry for entry in word_entries]
                    custom_word_desc.append(f"{category}: {', '.join(words)}")
            
            if custom_word_desc:
                prompt_additions.append(f"PERSONAL DISTRESS WORDS: {' | '.join(custom_word_desc)}")
        
        # Custom gestures
        if custom_gestures:
            gesture_desc = []
            for category, gestures in custom_gestures.items():
                for gesture in gestures:
                    if gesture.get("enabled", True):
                        gesture_desc.append(f"{gesture['name']}: {gesture['description']}")
            
            if gesture_desc:
                prompt_additions.append(f"PERSONAL VISUAL SIGNALS: {' | '.join(gesture_desc)}")
        
        return "\n".join(prompt_additions) if prompt_additions else ""


class PersonalizationWizard:
    """Interactive wizard for setting up personalization"""
    
    def __init__(self, personalization_manager: PersonalizationManager, parent=None):
        self.pm = personalization_manager
        self.parent = parent
        self.window = None
    
    def run(self):
        """Run the personalization wizard"""
        self.window = tk.Toplevel(self.parent) if self.parent else tk.Tk()
        self.window.title("WARYON Personalization Wizard")
        self.window.geometry("600x700")
        self.window.configure(bg='#1a1a2e')
        
        # Colors
        colors = {
            'bg': '#1a1a2e',
            'primary': '#16213e',
            'accent': '#0f4c75',
            'success': '#00ff88',
            'text': '#ffffff'
        }
        
        # Title
        title_label = tk.Label(
            self.window,
            text="🛡️ Personalize Your WARYON Guardian",
            font=("Arial", 18, "bold"),
            bg=colors['bg'],
            fg=colors['success']
        )
        title_label.pack(pady=20)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Basic Info Tab
        self.create_basic_info_tab(notebook, colors)
        
        # Custom Words Tab
        self.create_custom_words_tab(notebook, colors)
        
        # Custom Gestures Tab
        self.create_custom_gestures_tab(notebook, colors)
        
        # Advanced Settings Tab
        self.create_advanced_settings_tab(notebook, colors)
        
        # Buttons
        button_frame = tk.Frame(self.window, bg=colors['bg'])
        button_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Button(
            button_frame,
            text="💾 Save Personalization",
            command=self.save_and_close,
            bg=colors['success'],
            fg='black',
            font=("Arial", 12, "bold"),
            padx=20,
            pady=10
        ).pack(side='right', padx=10)
        
        tk.Button(
            button_frame,
            text="❌ Cancel",
            command=self.window.destroy,
            bg=colors['primary'],
            fg=colors['text'],
            padx=20,
            pady=10
        ).pack(side='right')
        
        # Load existing data
        self.load_existing_data()
        
        if not self.parent:
            self.window.mainloop()
    
    def create_basic_info_tab(self, notebook, colors):
        """Create basic information tab"""
        frame = tk.Frame(notebook, bg=colors['bg'])
        notebook.add(frame, text="👤 Basic Info")
        
        # User info
        tk.Label(frame, text="Personal Information", font=("Arial", 14, "bold"), 
                bg=colors['bg'], fg=colors['text']).pack(pady=(20, 10))
        
        # Name
        tk.Label(frame, text="Your Name:", bg=colors['bg'], fg=colors['text']).pack(anchor='w', padx=20)
        self.name_entry = tk.Entry(frame, font=("Arial", 11), width=40)
        self.name_entry.pack(pady=(5, 15), padx=20)
        
        # Nickname
        tk.Label(frame, text="Nickname (optional):", bg=colors['bg'], fg=colors['text']).pack(anchor='w', padx=20)
        self.nickname_entry = tk.Entry(frame, font=("Arial", 11), width=40)
        self.nickname_entry.pack(pady=(5, 15), padx=20)
        
        # Family names
        tk.Label(frame, text="Family Member Names (comma-separated):", bg=colors['bg'], fg=colors['text']).pack(anchor='w', padx=20)
        self.family_entry = tk.Entry(frame, font=("Arial", 11), width=40)
        self.family_entry.pack(pady=(5, 15), padx=20)
        
        # Emergency phrase
        tk.Label(frame, text="Personal Emergency Phrase:", bg=colors['bg'], fg=colors['text']).pack(anchor='w', padx=20)
        tk.Label(frame, text="(A unique phrase only you would say in danger)", 
                font=("Arial", 9, "italic"), bg=colors['bg'], fg=colors['accent']).pack(anchor='w', padx=20)
        self.emergency_phrase_entry = tk.Entry(frame, font=("Arial", 11), width=40)
        self.emergency_phrase_entry.pack(pady=(5, 15), padx=20)
        
        # Safe word
        tk.Label(frame, text="Safe Word:", bg=colors['bg'], fg=colors['text']).pack(anchor='w', padx=20)
        tk.Label(frame, text="(A word to cancel false alarms)", 
                font=("Arial", 9, "italic"), bg=colors['bg'], fg=colors['accent']).pack(anchor='w', padx=20)
        self.safe_word_entry = tk.Entry(frame, font=("Arial", 11), width=40)
        self.safe_word_entry.pack(pady=(5, 15), padx=20)
    
    def create_custom_words_tab(self, notebook, colors):
        """Create custom words tab"""
        frame = tk.Frame(notebook, bg=colors['bg'])
        notebook.add(frame, text="🗣️ Custom Words")
        
        tk.Label(frame, text="Add Your Personal Distress Words", 
                font=("Arial", 14, "bold"), bg=colors['bg'], fg=colors['text']).pack(pady=(20, 10))
        
        # Custom words section
        words_frame = tk.Frame(frame, bg=colors['bg'])
        words_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Category selection
        tk.Label(words_frame, text="Category:", bg=colors['bg'], fg=colors['text']).pack(anchor='w')
        self.word_category_var = tk.StringVar(value="emergency_phrases")
        word_category_combo = ttk.Combobox(
            words_frame,
            textvariable=self.word_category_var,
            values=["emergency_phrases", "family_calls", "personal_help_words", "danger_signals", "medical_terms"],
            state="readonly",
            width=30
        )
        word_category_combo.pack(pady=(5, 15))
        
        # Word input
        tk.Label(words_frame, text="Custom Word/Phrase:", bg=colors['bg'], fg=colors['text']).pack(anchor='w')
        self.custom_word_entry = tk.Entry(words_frame, font=("Arial", 11), width=40)
        self.custom_word_entry.pack(pady=(5, 10))
        
        # Context input
        tk.Label(words_frame, text="Context (optional):", bg=colors['bg'], fg=colors['text']).pack(anchor='w')
        self.word_context_entry = tk.Entry(words_frame, font=("Arial", 11), width=40)
        self.word_context_entry.pack(pady=(5, 15))
        
        # Add button
        tk.Button(
            words_frame,
            text="➕ Add Custom Word",
            command=self.add_custom_word,
            bg=colors['accent'],
            fg=colors['text'],
            padx=15,
            pady=5
        ).pack(pady=10)
        
        # Words list
        tk.Label(words_frame, text="Your Custom Words:", bg=colors['bg'], fg=colors['text']).pack(anchor='w', pady=(20, 5))
        
        self.words_listbox = tk.Listbox(
            words_frame,
            bg=colors['primary'],
            fg=colors['text'],
            font=("Consolas", 10),
            height=8
        )
        self.words_listbox.pack(fill='both', expand=True, pady=(0, 10))
    
    def create_custom_gestures_tab(self, notebook, colors):
        """Create custom gestures tab"""
        frame = tk.Frame(notebook, bg=colors['bg'])
        notebook.add(frame, text="👋 Custom Gestures")
        
        tk.Label(frame, text="Add Your Personal Visual Signals", 
                font=("Arial", 14, "bold"), bg=colors['bg'], fg=colors['text']).pack(pady=(20, 10))
        
        gestures_frame = tk.Frame(frame, bg=colors['bg'])
        gestures_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Gesture category
        tk.Label(gestures_frame, text="Gesture Type:", bg=colors['bg'], fg=colors['text']).pack(anchor='w')
        self.gesture_category_var = tk.StringVar(value="hand_signals")
        gesture_category_combo = ttk.Combobox(
            gestures_frame,
            textvariable=self.gesture_category_var,
            values=["hand_signals", "facial_expressions", "body_movements", "emergency_gestures"],
            state="readonly",
            width=30
        )
        gesture_category_combo.pack(pady=(5, 15))
        
        # Gesture name
        tk.Label(gestures_frame, text="Gesture Name:", bg=colors['bg'], fg=colors['text']).pack(anchor='w')
        self.gesture_name_entry = tk.Entry(gestures_frame, font=("Arial", 11), width=40)
        self.gesture_name_entry.pack(pady=(5, 15))
        
        # Description
        tk.Label(gestures_frame, text="Description:", bg=colors['bg'], fg=colors['text']).pack(anchor='w')
        self.gesture_desc_entry = tk.Entry(gestures_frame, font=("Arial", 11), width=40)
        self.gesture_desc_entry.pack(pady=(5, 15))
        
        # Visual cues
        tk.Label(gestures_frame, text="Visual Cues (comma-separated):", bg=colors['bg'], fg=colors['text']).pack(anchor='w')
        tk.Label(gestures_frame, text="(e.g., 'raised hand', 'pointing up', 'covering face')", 
                font=("Arial", 9, "italic"), bg=colors['bg'], fg=colors['accent']).pack(anchor='w')
        self.visual_cues_entry = tk.Entry(gestures_frame, font=("Arial", 11), width=40)
        self.visual_cues_entry.pack(pady=(5, 15))
        
        # Add button
        tk.Button(
            gestures_frame,
            text="➕ Add Custom Gesture",
            command=self.add_custom_gesture,
            bg=colors['accent'],
            fg=colors['text'],
            padx=15,
            pady=5
        ).pack(pady=10)
        
        # Gestures list
        tk.Label(gestures_frame, text="Your Custom Gestures:", bg=colors['bg'], fg=colors['text']).pack(anchor='w', pady=(20, 5))
        
        self.gestures_listbox = tk.Listbox(
            gestures_frame,
            bg=colors['primary'],
            fg=colors['text'],
            font=("Consolas", 10),
            height=8
        )
        self.gestures_listbox.pack(fill='both', expand=True, pady=(0, 10))
    
    def create_advanced_settings_tab(self, notebook, colors):
        """Create advanced settings tab"""
        frame = tk.Frame(notebook, bg=colors['bg'])
        notebook.add(frame, text="⚙️ Advanced")
        
        tk.Label(frame, text="Advanced Personalization Settings", 
                font=("Arial", 14, "bold"), bg=colors['bg'], fg=colors['text']).pack(pady=(20, 10))
        
        # Personalization level
        level_frame = tk.LabelFrame(frame, text="Personalization Level", 
                                   bg=colors['bg'], fg=colors['text'])
        level_frame.pack(fill='x', padx=20, pady=15)
        
        self.personalization_level_var = tk.StringVar(value="intermediate")
        
        levels = [
            ("Basic", "basic", "Standard keywords and gestures"),
            ("Intermediate", "intermediate", "Custom words + basic gestures"),
            ("Advanced", "advanced", "Full customization + pattern learning")
        ]
        
        for name, value, desc in levels:
            rb = tk.Radiobutton(
                level_frame,
                text=f"{name} - {desc}",
                variable=self.personalization_level_var,
                value=value,
                bg=colors['bg'],
                fg=colors['text'],
                selectcolor=colors['primary']
            )
            rb.pack(anchor='w', padx=10, pady=5)
        
        # Preview frame
        preview_frame = tk.LabelFrame(frame, text="Personalization Preview", 
                                     bg=colors['bg'], fg=colors['text'])
        preview_frame.pack(fill='both', expand=True, padx=20, pady=15)
        
        self.preview_text = tk.Text(
            preview_frame,
            bg=colors['primary'],
            fg=colors['text'],
            font=("Consolas", 9),
            height=12,
            wrap='word'
        )
        self.preview_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Update preview button
        tk.Button(
            preview_frame,
            text="🔄 Update Preview",
            command=self.update_preview,
            bg=colors['accent'],
            fg=colors['text'],
            padx=15,
            pady=5
        ).pack(pady=10)
    
    def add_custom_word(self):
        """Add custom word to the list"""
        word = self.custom_word_entry.get().strip()
        category = self.word_category_var.get()
        context = self.word_context_entry.get().strip()
        
        if not word:
            messagebox.showwarning("Warning", "Please enter a word or phrase")
            return
        
        if self.pm.add_custom_word(category, word, context):
            self.words_listbox.insert('end', f"{category}: '{word}' - {context}")
            self.custom_word_entry.delete(0, 'end')
            self.word_context_entry.delete(0, 'end')
            print(f"✅ Added custom word: {word} ({category})")
        else:
            messagebox.showerror("Error", "Failed to add custom word")
    
    def add_custom_gesture(self):
        """Add custom gesture to the list"""
        name = self.gesture_name_entry.get().strip()
        category = self.gesture_category_var.get()
        description = self.gesture_desc_entry.get().strip()
        visual_cues_text = self.visual_cues_entry.get().strip()
        
        if not all([name, description, visual_cues_text]):
            messagebox.showwarning("Warning", "Please fill in all gesture fields")
            return
        
        visual_cues = [cue.strip() for cue in visual_cues_text.split(',') if cue.strip()]
        
        if self.pm.add_custom_gesture(category, name, description, visual_cues):
            self.gestures_listbox.insert('end', f"{category}: {name} - {description}")
            self.gesture_name_entry.delete(0, 'end')
            self.gesture_desc_entry.delete(0, 'end')
            self.visual_cues_entry.delete(0, 'end')
            print(f"✅ Added custom gesture: {name} ({category})")
        else:
            messagebox.showerror("Error", "Failed to add custom gesture")
    
    def update_preview(self):
        """Update the personalization preview"""
        # Collect current form data
        user_info = {
            "name": self.name_entry.get(),
            "nickname": self.nickname_entry.get(),
            "family_names": [name.strip() for name in self.family_entry.get().split(',') if name.strip()],
            "emergency_phrase": self.emergency_phrase_entry.get(),
            "safe_word": self.safe_word_entry.get()
        }
        
        # Update personal settings temporarily
        old_user_info = self.pm.personal_settings.get("user_info", {})
        self.pm.personal_settings["user_info"].update(user_info)
        self.pm.personal_settings["personalization_level"] = self.personalization_level_var.get()
        
        # Generate preview
        preview_text = self.pm.create_personal_gemma_prompt_additions()
        
        if not preview_text:
            preview_text = "No personalization configured yet. Fill in the forms and add custom words/gestures to see how WARYON will be personalized for you."
        
        # Show preview
        self.preview_text.delete(1.0, 'end')
        self.preview_text.insert(1.0, f"🛡️ PERSONALIZED WARYON CONTEXT FOR GEMMA 3N:\n\n{preview_text}")
        
        # Restore original settings
        self.pm.personal_settings["user_info"] = old_user_info
    
    def load_existing_data(self):
        """Load existing personalization data into the form"""
        settings = self.pm.personal_settings
        
        # Basic info
        user_info = settings.get("user_info", {})
        if user_info.get("name"):
            self.name_entry.insert(0, user_info["name"])
        if user_info.get("nickname"):
            self.nickname_entry.insert(0, user_info["nickname"])
        if user_info.get("family_names"):
            self.family_entry.insert(0, ", ".join(user_info["family_names"]))
        if user_info.get("emergency_phrase"):
            self.emergency_phrase_entry.insert(0, user_info["emergency_phrase"])
        if user_info.get("safe_word"):
            self.safe_word_entry.insert(0, user_info["safe_word"])
        
        # Load custom words
        custom_words = settings.get("custom_distress_words", {})
        for category, words in custom_words.items():
            for word_entry in words:
                if isinstance(word_entry, dict):
                    word = word_entry["word"]
                    context = word_entry.get("context", "")
                    self.words_listbox.insert('end', f"{category}: '{word}' - {context}")
        
        # Load custom gestures
        custom_gestures = settings.get("custom_gestures", {})
        for category, gestures in custom_gestures.items():
            for gesture in gestures:
                name = gesture["name"]
                description = gesture["description"]
                self.gestures_listbox.insert('end', f"{category}: {name} - {description}")
        
        # Personalization level
        level = settings.get("personalization_level", "intermediate")
        self.personalization_level_var.set(level)
    
    def save_and_close(self):
        """Save all personalization data and close wizard"""
        try:
            # Save user info
            user_info = {
                "name": self.name_entry.get(),
                "nickname": self.nickname_entry.get(),
                "family_names": [name.strip() for name in self.family_entry.get().split(',') if name.strip()],
                "emergency_phrase": self.emergency_phrase_entry.get(),
                "safe_word": self.safe_word_entry.get(),
                "updated": datetime.now().isoformat()
            }
            
            self.pm.personal_settings["user_info"].update(user_info)
            self.pm.personal_settings["personalization_level"] = self.personalization_level_var.get()
            
            if self.pm.save_personal_settings():
                messagebox.showinfo("Success", "✅ Personalization saved successfully!\n\nWARYON is now customized for you.")
                print("✅ Personalization wizard completed successfully")
                self.window.destroy()
            else:
                messagebox.showerror("Error", "Failed to save personalization settings")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving personalization: {e}")


# Enhanced Audio Processor Integration
class EnhancedAudioProcessorWithPersonalization:
    """Add these methods to your existing RealAudioProcessor class"""
    
    def __init__(self, config_manager, threat_callback: Callable = None): # pyright: ignore[reportUndefinedVariable]
        # ... existing init code ...
        
        # Add personalization manager
        self.personalization = PersonalizationManager(config_manager)
        
        # Get personalized distress words
        self.distress_keywords = self.personalization.get_all_distress_words_combined()
        
        print("👤 Personalized audio processor initialized")
    
    def _analyze_for_personalized_threats(self, volume, max_volume, speech_text, is_loud) -> Dict[str, Any]:
        """Enhanced threat analysis with personalization"""
        threat_analysis = self._analyze_for_threats_enhanced(volume, max_volume, speech_text, is_loud)
        
        if speech_text:
            user_info = self.personalization.personal_settings.get("user_info", {})
            
            # Check for emergency phrase
            emergency_phrase = user_info.get("emergency_phrase", "").lower()
            if emergency_phrase and emergency_phrase in speech_text.lower():
                threat_analysis["threat_detected"] = True
                threat_analysis["threat_level"] = "critical"
                threat_analysis["threat_type"] = "Personal Emergency Phrase"
                threat_analysis["confidence"] = 0.95
                threat_analysis["reasoning"] = f"User's personal emergency phrase detected: '{emergency_phrase}'"
            
            # Check for safe word
            safe_word = user_info.get("safe_word", "").lower()
            if safe_word and safe_word in speech_text.lower():
                threat_analysis["threat_detected"] = False
                threat_analysis["threat_level"] = "safe"
                threat_analysis["threat_type"] = "Safe Word Detected"
                threat_analysis["confidence"] = 0.0
                threat_analysis["reasoning"] = f"User's safe word detected - canceling all alerts: '{safe_word}'"
                threat_analysis["safe_word_used"] = True
            
            # Check for family names in distress context
            family_names = user_info.get("family_names", [])
            for family_name in family_names:
                if family_name.lower() in speech_text.lower():
                    if any(distress_word in speech_text.lower() for distress_word in ["help", "hurt", "stop", "emergency"]):
                        threat_analysis["threat_detected"] = True
                        threat_analysis["confidence"] = min(threat_analysis["confidence"] + 0.3, 1.0)
                        threat_analysis["reasoning"] += f" | Family member '{family_name}' mentioned in distress context"
        
        return threat_analysis
    
    def get_personalized_audio_description(self) -> str:
        """Get audio description with personal context for Gemma 3n"""
        base_description = self.get_audio_description()
        
        # Add personal context
        personal_context = self.personalization.create_personal_gemma_prompt_additions()
        
        if personal_context:
            return f"{personal_context}\n\nCURRENT AUDIO ANALYSIS: {base_description}"
        else:
            return base_description


# Enhanced AI Engine Integration
def create_personalized_gemma_prompt(base_prompt: str, personalization_manager: PersonalizationManager) -> str:
    """Enhance Gemma 3n prompts with personalization"""
    personal_additions = personalization_manager.create_personal_gemma_prompt_additions()
    
    if personal_additions:
        enhanced_prompt = f"""You are WARYON, analyzing this specific person's safety with their personal context.

{personal_additions}

{base_prompt}

IMPORTANT: Use this personal context when analyzing threats. The user's emergency phrase and custom words should be treated as high-priority threats. If you detect the safe word, immediately mark the situation as safe regardless of other indicators."""
        
        return enhanced_prompt
    
    return base_prompt


# Enhanced Video Processor Integration  
def add_personalized_gesture_detection(self, image, personalization_manager: PersonalizationManager):
    """Add custom gesture detection to video analysis"""
    custom_gestures = personalization_manager.get_custom_gestures()
    
    gesture_detections = []
    
    for category, gestures in custom_gestures.items():
        for gesture in gestures:
            if not gesture.get("enabled", True):
                continue
            
            gesture_name = gesture["name"]
            visual_cues = gesture["visual_cues"]
            confidence_threshold = gesture.get("confidence_threshold", 0.7)
            
            # Create gesture detection prompt for Gemma 3n
            gesture_prompt = f"""
CUSTOM GESTURE DETECTION: Look for the user's personal gesture '{gesture_name}'.
Description: {gesture['description']}
Visual indicators to look for: {', '.join(visual_cues)}
Required confidence threshold: {confidence_threshold}

Analyze if this gesture is present in the image.
"""
            
            # This would be sent to Gemma 3n along with the image
            gesture_detections.append({
                "name": gesture_name,
                "category": category,
                "prompt": gesture_prompt,
                "threshold": confidence_threshold
            })
    
    return gesture_detections


# Main Application Integration
def add_personalization_to_main_app(self):
    """Add personalization button to main WARYON application"""
    
    # Add to your main.py control panel
    personalization_button = tk.Button(
        self.control_panel,  # Your existing control panel
        text="👤 Personalize WARYON",
        command=self.open_personalization_wizard,
        bg=self.colors['accent'],
        fg=self.colors['text'],
        width=20,
        pady=5
    )
    personalization_button.pack(pady=5, padx=10)

def open_personalization_wizard(self):
    """Open personalization wizard from main app"""
    try:
        # Initialize personalization manager if not exists
        if not hasattr(self, 'personalization_manager'):
            self.personalization_manager = PersonalizationManager(self.config)
        
        # Launch wizard
        self.personalization_manager.setup_personalization_wizard(self.root)
        
        # Refresh audio processor with new personalization
        if hasattr(self, 'audio_processor') and self.audio_processor:
            self.audio_processor.distress_keywords = self.personalization_manager.get_all_distress_words_combined()
            print("✅ Audio processor updated with new personalization")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open personalization: {e}")


# Usage Example and Test
if __name__ == "__main__":
    print("👤 Testing WARYON Personalization System")
    print("=" * 50)
    
    # Mock config for testing
    class MockConfig:
        def get(self, key, default=None):
            return default
    
    config = MockConfig()
    
    # Test personalization manager
    pm = PersonalizationManager(config)
    
    # Add some test personalization
    pm.personal_settings["user_info"] = {
        "name": "Sarah",
        "family_names": ["Mom", "Dad", "Jake"],
        "emergency_phrase": "red apple emergency",
        "safe_word": "sunshine"
    }
    
    # Add custom words
    pm.add_custom_word("emergency_phrases", "code red", "Personal emergency signal")
    pm.add_custom_word("family_calls", "mom help", "Calling for mother in emergency")
    
    # Add custom gesture
    pm.add_custom_gesture("emergency_gestures", "Thumb Down", 
                         "Thumbs down gesture for silent emergency", 
                         ["thumb pointing down", "closed fist with thumb down"])
    
    # Test Gemma prompt enhancement
    base_prompt = "Analyze this image for threats."
    enhanced_prompt = create_personalized_gemma_prompt(base_prompt, pm)
    
    print("🧠 Enhanced Gemma 3n Prompt:")
    print("=" * 30)
    print(enhanced_prompt[:300] + "...")
    
    # Test personalized audio descriptions
    personal_context = pm.create_personal_gemma_prompt_additions()
    print(f"\n👤 Personal Context Generated:")
    print("=" * 30)
    print(personal_context)
    
    # Save test settings
    if pm.save_personal_settings():
        print(f"\n✅ Test personalization saved to: {pm.user_profile_path}")
    
    print(f"\n🛡️ Personalized WARYON Features:")
    print("✅ Custom distress words and phrases")
    print("✅ Personal emergency phrase and safe word")
    print("✅ Family member name recognition")
    print("✅ Custom visual gesture detection")
    print("✅ Personalized Gemma 3n prompts")
    print("✅ User-specific threat patterns")
    
    print(f"\n🚀 To launch personalization wizard:")
    print("pm.setup_personalization_wizard()")
    
    print(f"\n👤 Personalization system test complete!")