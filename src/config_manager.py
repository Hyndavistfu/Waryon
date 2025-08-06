"""
WARYON Configuration Manager
Competition-Ready Settings Management
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class ConfigManager:
    def __init__(self, config_path: str = "config/waryon_competition.json"):
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.config_dir.mkdir(exist_ok=True)
        
        self._config = self._load_config()
        
        print(f"⚙️ Configuration loaded: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Ensure all required sections exist
                    return self._validate_config(config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Error loading config: {e}")
                config = self._default_config()
                # Set the config before saving
                self._config = config
                self.save_config()  # Create default config file
                return config
        else:
            config = self._default_config()
            # Set the config before saving
            self._config = config
            self.save_config()  # Create default config file
            return config
    
    def _default_config(self) -> Dict[str, Any]:
        """Return competition-ready default configuration"""
        return {
            "competition_info": {
                "app_name": "WARYON AI Guardian",
                "version": "1.0",
                "technology": "Gemma 3n Dynamic Scaling",
                "created": datetime.now().isoformat()
            },
            "user": {
                "name": "Competition Demo",
                "email": "demo@waryon.ai",
                "phone": "",
                "age_group": "Adult (18-65)"
            },
            "emergency_contacts": [
                {
                    "name": "Emergency Services",
                    "email": "emergency@example.com",
                    "phone": "911",
                    "relationship": "Emergency Service",
                    "priority": "High"
                }
            ],
            "detection_settings": {
                "violence_sensitivity": 0.7,
                "fall_sensitivity": 0.8,
                "audio_sensitivity": 0.6,
                "expression_sensitivity": 0.75,
                "custom_gestures": []
            },
            "alert_settings": {
                "email_enabled": True,
                "sound_enabled": True,
                "notification_enabled": True,
                "desktop_alerts": True,
                "emergency_timeout": 30
            },
            "ai_settings": {
                "model": "gemma3n:e4b",
                "default_performance": "auto",
                "auto_scaling": True,
                "max_tokens": 500,
                "temperature": 0.1,
                "confidence_threshold": 0.6
            },
            "system_settings": {
                "video_fps": 30,
                "analysis_interval": 2.0,
                "audio_sample_rate": 44100,
                "enable_system_tray": True,
                "save_evidence": True,
                "max_evidence_files": 100
            },
            "competition_settings": {
                "demo_mode": True,
                "performance_tracking": True,
                "detailed_logging": True,
                "export_statistics": True,
                "live_metrics": True
            },
            "setup_completed": True
        }
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and update config with any missing sections"""
        default = self._default_config()
        
        # Ensure all top-level sections exist
        for section in default:
            if section not in config:
                config[section] = default[section]
            elif isinstance(default[section], dict):
                # Ensure all subsection keys exist
                for key in default[section]:
                    if key not in config[section]:
                        config[section][key] = default[section][key]
        
        return config
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            self.config_dir.mkdir(exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            
            return True
            
        except IOError as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def is_setup_completed(self) -> bool:
        """Check if initial setup is completed"""
        return self.get('setup_completed', False)
    
    def get_emergency_contacts(self) -> list:
        """Get list of emergency contacts"""
        return self.get('emergency_contacts', [])
    
    def get_detection_sensitivity(self, threat_type: str) -> float:
        """Get detection sensitivity for specific threat type"""
        return self.get(f'detection_settings.{threat_type}_sensitivity', 0.7)
    
    def get_ai_model(self) -> str:
        """Get AI model name"""
        return self.get('ai_settings.model', 'gemma3n:e4b')
    
    def get_performance_level(self) -> str:
        """Get default performance level"""
        return self.get('ai_settings.default_performance', 'auto')
    
    def is_competition_mode(self) -> bool:
        """Check if running in competition mode"""
        return self.get('competition_settings.demo_mode', True)
    
    def get_competition_settings(self) -> Dict[str, Any]:
        """Get all competition settings"""
        return self.get('competition_settings', {})
    
    def update_user_info(self, name: str = None, email: str = None, phone: str = None, age_group: str = None):
        """Update user information"""
        if name is not None:
            self.set('user.name', name)
        if email is not None:
            self.set('user.email', email)
        if phone is not None:
            self.set('user.phone', phone)
        if age_group is not None:
            self.set('user.age_group', age_group)
    
    def add_emergency_contact(self, name: str, email: str, phone: str = "", relationship: str = "Family", priority: str = "High") -> None:
        """Add emergency contact"""
        contacts = self.get_emergency_contacts()
        
        contact = {
            "name": name,
            "email": email,
            "phone": phone,
            "relationship": relationship,
            "priority": priority,
            "added": datetime.now().isoformat()
        }
        
        contacts.append(contact)
        self.set('emergency_contacts', contacts)
    
    def remove_emergency_contact(self, index: int) -> bool:
        """Remove emergency contact by index"""
        contacts = self.get_emergency_contacts()
        if 0 <= index < len(contacts):
            contacts.pop(index)
            self.set('emergency_contacts', contacts)
            return True
        return False
    
    def update_detection_settings(self, **kwargs):
        """Update detection sensitivity settings"""
        for key, value in kwargs.items():
            if key.endswith('_sensitivity'):
                self.set(f'detection_settings.{key}', value)
    
    def update_ai_settings(self, **kwargs):
        """Update AI engine settings"""
        for key, value in kwargs.items():
            self.set(f'ai_settings.{key}', value)
    
    def update_system_settings(self, **kwargs):
        """Update system settings"""
        for key, value in kwargs.items():
            self.set(f'system_settings.{key}', value)
    
    def get_user_info(self) -> Dict[str, str]:
        """Get user information"""
        return self.get('user', {})
    
    def mark_setup_completed(self):
        """Mark setup as completed"""
        self.set('setup_completed', True)
        self.save_config()
    
    def export_config(self, filepath: str) -> bool:
        """Export configuration to specific file"""
        try:
            export_path = Path(filepath)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            return False
    
    def import_config(self, filepath: str) -> bool:
        """Import configuration from file"""
        try:
            import_path = Path(filepath)
            
            if not import_path.exists():
                print(f"❌ Import file not found: {filepath}")
                return False
            
            with open(import_path, 'r') as f:
                imported_config = json.load(f)
            
            # Validate imported config
            self._config = self._validate_config(imported_config)
            
            # Save the imported config
            return self.save_config()
            
        except Exception as e:
            print(f"❌ Import error: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values"""
        try:
            self._config = self._default_config()
            return self.save_config()
            
        except Exception as e:
            print(f"❌ Reset error: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display"""
        return {
            "user_name": self.get('user.name', 'Unknown'),
            "emergency_contacts": len(self.get_emergency_contacts()),
            "ai_model": self.get_ai_model(),
            "performance_level": self.get_performance_level(),
            "competition_mode": self.is_competition_mode(),
            "setup_completed": self.is_setup_completed(),
            "last_modified": datetime.now().isoformat()
        }


# Test the configuration manager
if __name__ == "__main__":
    print("⚙️ Testing WARYON Configuration Manager")
    print("=" * 45)
    
    # Initialize config manager
    config = ConfigManager()
    
    # Display current configuration
    print("📋 Current Configuration:")
    summary = config.get_config_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test getting values
    print(f"\n🤖 AI Model: {config.get_ai_model()}")
    print(f"🎯 Performance Level: {config.get_performance_level()}")
    print(f"🏆 Competition Mode: {config.is_competition_mode()}")
    
    # Test emergency contacts
    contacts = config.get_emergency_contacts()
    print(f"\n🚨 Emergency Contacts: {len(contacts)}")
    for i, contact in enumerate(contacts):
        print(f"  {i+1}. {contact['name']} ({contact['relationship']})")
    
    # Test detection settings
    print(f"\n🎯 Detection Sensitivities:")
    for threat_type in ['violence', 'fall', 'audio', 'expression']:
        sensitivity = config.get_detection_sensitivity(threat_type)
        print(f"  {threat_type}: {sensitivity}")
    
    # Test saving
    if config.save_config():
        print(f"\n✅ Configuration saved to: {config.config_path}")
    else:
        print(f"\n❌ Failed to save configuration")
    
    print("\n🛡️ Configuration manager test complete")