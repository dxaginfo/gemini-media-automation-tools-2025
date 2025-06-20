#!/usr/bin/env python3
"""
SoundScaffold - Generates audio templates from scene descriptions

This tool converts textual scene descriptions into audio templates that match the described
atmosphere and requirements.

Usage:
    python sound_scaffold.py --input <description_file> --output <output_dir> [options]
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Any, Union

# For text processing
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import yaml
except ImportError:
    logging.error("Required NLP libraries not installed. Run: pip install nltk pyyaml")
    sys.exit(1)

# For audio generation and processing
try:
    import numpy as np
    import soundfile as sf
    from scipy import signal
    import librosa
    import librosa.display
except ImportError:
    logging.error("Required audio libraries not installed. Run: pip install numpy soundfile scipy librosa")
    sys.exit(1)

# For Google Cloud and Gemini API integration
try:
    import google.generativeai as genai
    from google.cloud import texttospeech
    from google.cloud import storage
except ImportError:
    logging.warning("Google Cloud libraries not installed. Some features will be disabled.")

# Download necessary NLTK data packages
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class SoundScaffold:
    """Generates audio templates from scene descriptions."""

    def __init__(self, api_key: Optional[str] = None, sound_library_path: Optional[str] = None):
        """Initialize the SoundScaffold tool.
        
        Args:
            api_key: Gemini API key. If not provided, attempts to use GEMINI_API_KEY env variable.
            sound_library_path: Path to the sound library directory
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('SoundScaffold')
        
        # Initialize API key
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            self.logger.warning("No Gemini API key provided. Some features will be limited.")
        else:
            genai.configure(api_key=self.api_key)
        
        # Initialize Gemini model if available
        self.gemini_model = None
        if self.api_key:
            try:
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                self.logger.info("Initialized Gemini model")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini model: {e}")
        
        # Initialize sound library
        self.sound_library_path = sound_library_path or './sound_library'
        self.sound_catalog = self._load_sound_catalog()
        
        # Text processing components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Store results
        self.results = {}
        self.scene_analysis = {}
        self.sound_elements = []
    
    def _load_sound_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load the sound catalog from the sound library.
        
        Returns:
            Dictionary mapping sound categories to available sounds
        """
        catalog_path = os.path.join(self.sound_library_path, 'catalog.json')
        
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, 'r') as f:
                    catalog = json.load(f)
                self.logger.info(f"Loaded sound catalog with {len(catalog)} categories")
                return catalog
            except Exception as e:
                self.logger.error(f"Error loading sound catalog: {e}")
        
        # If catalog doesn't exist or failed to load, create a minimal structure
        self.logger.warning("No sound catalog found. Creating minimal structure.")
        return {
            "ambient": {},
            "effects": {},
            "music": {},
            "voices": {}
        }
    
    def load_description(self, description_file: str) -> Dict[str, Any]:
        """Load a scene description from a file.
        
        Args:
            description_file: Path to the description file (JSON, YAML, or TXT)
            
        Returns:
            Dictionary containing scene description data
        """
        try:
            with open(description_file, 'r') as f:
                file_ext = os.path.splitext(description_file)[1].lower()
                
                if file_ext == '.json':
                    data = json.load(f)
                elif file_ext in ('.yaml', '.yml'):
                    data = yaml.safe_load(f)
                else:  # Treat as plain text
                    text = f.read()
                    data = {"description": text, "title": os.path.basename(description_file)}
            
            self.logger.info(f"Loaded description from {description_file}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading description from {description_file}: {e}")
            return {"error": str(e)}
    
    def analyze_description(self, description: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a scene description to identify sound elements.
        
        Args:
            description: Dictionary containing scene description data
            
        Returns:
            Dictionary containing analysis results
        """
        # Extract the main description text
        description_text = description.get("description", "")
        if not description_text:
            self.logger.error("No description text found in input")
            return {"error": "No description text found in input"}
        
        # Basic analysis results structure
        analysis = {
            "title": description.get("title", "Untitled Scene"),
            "mood": description.get("mood", []),
            "tempo": description.get("tempo", "medium"),
            "setting": description.get("setting", ""),
            "time_period": description.get("time_period", "present"),
            "key_elements": [],
            "sound_categories": {
                "ambient": [],
                "effects": [],
                "music": [],
                "voices": []
            },
            "suggested_layers": []
        }
        
        # Use Gemini for advanced analysis if available
        if self.gemini_model:
            gemini_analysis = self._analyze_with_gemini(description_text)
            if gemini_analysis:
                # Merge the Gemini analysis with our basic structure
                # but keep our structure as the base
                for key, value in gemini_analysis.items():
                    if key in analysis and isinstance(analysis[key], dict) and isinstance(value, dict):
                        analysis[key].update(value)
                    elif key in analysis and isinstance(analysis[key], list) and isinstance(value, list):
                        analysis[key].extend(value)
                    else:
                        analysis[key] = value
        
        # Fallback to basic NLP if Gemini is unavailable or failed
        if not analysis.get("key_elements") and not self.gemini_model:
            analysis["key_elements"] = self._extract_key_elements(description_text)
            analysis["sound_categories"] = self._categorize_sound_elements(analysis["key_elements"])
        
        # Store the analysis for later use
        self.scene_analysis = analysis
        self.logger.info(f"Completed analysis for scene: {analysis['title']}")
        
        return analysis
    
    def _analyze_with_gemini(self, description_text: str) -> Dict[str, Any]:
        """Use Gemini API to analyze the scene description.
        
        Args:
            description_text: Text description of the scene
            
        Returns:
            Dictionary containing Gemini's analysis or empty dict if analysis failed
        """
        if not self.gemini_model:
            return {}
        
        try:
            prompt = f"""
Analyze this scene description and extract audio-relevant information. 
Format your response as a JSON object with the following structure:
{{
  "mood": [list of mood descriptors],
  "tempo": "slow", "medium", or "fast",
  "setting": "description of the physical setting",
  "time_period": "historical period if relevant",
  "key_elements": [list of important sound elements],
  "sound_categories": {{
    "ambient": [list of ambient sounds needed],
    "effects": [list of sound effects needed],
    "music": [list of music style or instrument suggestions],
    "voices": [list of voice types or dialogue elements]
  }},
  "suggested_layers": [ordered list of sound layers from background to foreground]
}}

Scene description:
{description_text}
"""
            
            response = self.gemini_model.generate_content(prompt)
            
            # Extract the JSON from the response
            response_text = response.text
            
            # Look for JSON-like content between curly braces
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
                self.logger.info("Successfully analyzed scene with Gemini API")
                return analysis
            else:
                self.logger.warning("Couldn't extract valid JSON from Gemini response")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error during Gemini analysis: {e}")
            return {}
    
    def _extract_key_elements(self, text: str) -> List[str]:
        """Extract key sound-related elements from text using basic NLP.
        
        Args:
            text: Description text
            
        Returns:
            List of key sound elements
        """
        # Tokenize the text
        words = word_tokenize(text.lower())
        
        # Remove stop words and lemmatize
        filtered_words = [self.lemmatizer.lemmatize(word) for word in words 
                          if word.isalpha() and word not in self.stop_words]
        
        # Sound-related keywords to look for
        sound_keywords = [
            'sound', 'noise', 'music', 'voice', 'speak', 'talk', 'shout', 'whisper',
            'loud', 'quiet', 'silent', 'echo', 'bang', 'crash', 'thud', 'ring', 'buzz',
            'hum', 'crackle', 'rustle', 'chirp', 'screech', 'howl', 'cry', 'laugh',
            'sing', 'play', 'instrument', 'melody', 'rhythm', 'beat', 'ambient',
            'atmosphere', 'mood', 'tone', 'background', 'foreground'
        ]
        
        # Find words that are sound-related or occur near sound-related words
        key_elements = []
        for i, word in enumerate(filtered_words):
            if word in sound_keywords:
                key_elements.append(word)
                
                # Include surrounding words for context
                start = max(0, i - 2)
                end = min(len(filtered_words), i + 3)
                for j in range(start, end):
                    if j != i and filtered_words[j] not in key_elements:
                        key_elements.append(filtered_words[j])
        
        return key_elements
    
    def _categorize_sound_elements(self, elements: List[str]) -> Dict[str, List[str]]:
        """Categorize sound elements into different categories.
        
        Args:
            elements: List of sound elements
            
        Returns:
            Dictionary mapping categories to lists of elements
        """
        # Simple keyword-based categorization
        categories = {
            "ambient": ['ambient', 'atmosphere', 'background', 'environment', 'room', 'space', 'location'],
            "effects": ['effect', 'sound', 'noise', 'crash', 'bang', 'footstep', 'door', 'impact'],
            "music": ['music', 'melody', 'rhythm', 'beat', 'song', 'tune', 'instrument', 'play', 'piano', 'guitar'],
            "voices": ['voice', 'speak', 'talk', 'shout', 'whisper', 'conversation', 'dialogue', 'monologue']
        }
        
        # Categorize elements
        categorized = {
            "ambient": [],
            "effects": [],
            "music": [],
            "voices": []
        }
        
        for element in elements:
            for category, keywords in categories.items():
                if element in keywords:
                    if element not in categorized[category]:
                        categorized[category].append(element)
                    break
        
        return categorized
    
    def generate_audio_template(self, output_dir: str) -> Dict[str, Any]:
        """Generate an audio template based on the scene analysis.
        
        Args:
            output_dir: Directory to save the generated audio files
            
        Returns:
            Dictionary containing information about the generated template
        """
        if not self.scene_analysis:
            self.logger.error("No scene analysis available. Run analyze_description() first.")
            return {"error": "No scene analysis available"}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Base result structure
        result = {
            "title": self.scene_analysis.get("title", "Untitled"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "layers": [],
            "files": [],
            "metadata": {}
        }
        
        # In a real implementation, we would generate actual audio templates here
        # For this example, we'll create placeholder files for each sound category
        
        # Get sound categories from analysis
        sound_categories = self.scene_analysis.get("sound_categories", {})
        
        # Process each sound category
        for category, elements in sound_categories.items():
            if elements:
                # Create a placeholder audio file for this category
                output_file = os.path.join(output_dir, f"{category}_template.wav")
                
                # In a real implementation, we would generate actual audio content
                # based on the elements in this category
                self._create_placeholder_audio(output_file, category, elements)
                
                # Add to results
                result["layers"].append({
                    "name": category,
                    "elements": elements,
                    "file": output_file
                })
                
                result["files"].append(output_file)
        
        # Create a mixed version combining all layers
        if result["layers"]:
            mixed_file = os.path.join(output_dir, f"mixed_template.wav")
            self._create_mixed_audio(result["files"], mixed_file)
            result["mixed_file"] = mixed_file
        
        # Save metadata
        metadata = {
            "scene_analysis": self.scene_analysis,
            "layers": result["layers"],
            "generated_timestamp": result["timestamp"]
        }
        
        metadata_file = os.path.join(output_dir, "template_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        result["metadata_file"] = metadata_file
        
        self.logger.info(f"Generated audio template with {len(result['layers'])} layers")
        self.results = result
        
        return result
    
    def _create_placeholder_audio(self, output_file: str, category: str, elements: List[str]) -> None:
        """Create a placeholder audio file for a specific category.
        
        In a real implementation, this would generate actual audio content based on the elements.
        
        Args:
            output_file: Path to save the audio file
            category: Sound category (ambient, effects, music, voices)
            elements: List of sound elements in this category
        """
        # Sample rate (44.1 kHz)
        sr = 44100
        
        # Duration (5 seconds)
        duration = 5
        
        # Generate different audio for each category
        t = np.linspace(0, duration, sr * duration, endpoint=False)
        
        if category == "ambient":
            # White noise for ambient
            audio = np.random.normal(0, 0.1, sr * duration)
            # Apply a lowpass filter
            b, a = signal.butter(4, 0.1, 'lowpass')
            audio = signal.filtfilt(b, a, audio)
        
        elif category == "effects":
            # Some random tones for effects
            audio = np.zeros(sr * duration)
            for i in range(min(len(elements), 5)):
                # Random timing for the effect
                start = int(sr * (i * 0.8))
                end = min(start + int(sr * 0.5), len(audio))
                # Generate a tone with frequency based on the element
                freq = 220 * (i + 1)
                effect = 0.3 * np.sin(2 * np.pi * freq * t[0:(end-start)])
                # Apply envelope
                envelope = np.hanning(end - start)
                audio[start:end] += effect * envelope
        
        elif category == "music":
            # Simple chord progression for music
            freqs = [262, 330, 392, 523]  # C4, E4, G4, C5
            audio = np.zeros(sr * duration)
            for freq in freqs:
                audio += 0.1 * np.sin(2 * np.pi * freq * t)
            # Apply a gentle fade in/out
            fade_len = int(sr * 0.5)
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            audio[:fade_len] *= fade_in
            audio[-fade_len:] *= fade_out
        
        elif category == "voices":
            # Vocal-like formants for voices
            carrier = np.sin(2 * np.pi * 150 * t)  # Base frequency (like a voice)
            # Add some modulation
            modulator = 1.0 + 0.3 * np.sin(2 * np.pi * 2 * t)
            audio = 0.3 * carrier * modulator
            # Add some random amplitude variations
            amplitude = np.ones_like(audio)
            for i in range(10):  # 10 "syllables"
                start = int(sr * (i * 0.4))
                if start >= len(amplitude):
                    break
                end = min(start + int(sr * 0.2), len(amplitude))
                amplitude[start:end] *= 0.7 + 0.3 * np.random.random()
            audio *= amplitude
        
        else:
            # Default: white noise
            audio = 0.1 * np.random.normal(0, 1, sr * duration)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Save to file
        sf.write(output_file, audio, sr)
        
        self.logger.info(f"Created placeholder audio for {category} at {output_file}")
    
    def _create_mixed_audio(self, input_files: List[str], output_file: str) -> None:
        """Create a mixed audio file combining all layers.
        
        Args:
            input_files: List of input audio files to mix
            output_file: Path to save the mixed audio file
        """
        if not input_files:
            self.logger.warning("No input files to mix")
            return
        
        try:
            # Load the first file to get sample rate and initialize the mix
            audio, sr = librosa.load(input_files[0], sr=None)
            mixed = audio
            
            # Mix in the remaining files
            for file in input_files[1:]:
                # Load and ensure same length as mixed
                next_audio, _ = librosa.load(file, sr=sr)
                
                # Match lengths
                if len(next_audio) < len(mixed):
                    # Pad with zeros
                    next_audio = np.pad(next_audio, (0, len(mixed) - len(next_audio)))
                elif len(next_audio) > len(mixed):
                    # Trim
                    next_audio = next_audio[:len(mixed)]
                
                # Add to mix (with reduced amplitude to avoid clipping)
                mixed = mixed + 0.7 * next_audio
            
            # Normalize to avoid clipping
            mixed = mixed / np.max(np.abs(mixed)) * 0.9
            
            # Save mixed file
            sf.write(output_file, mixed, sr)
            
            self.logger.info(f"Created mixed audio template at {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating mixed audio: {e}")

def main():
    """Main function to run the tool from command line."""
    parser = argparse.ArgumentParser(description='Generate audio templates from scene descriptions')
    parser.add_argument('--input', required=True, help='Path to the scene description file (JSON, YAML, or TXT)')
    parser.add_argument('--output', required=True, help='Path to the output directory for generated templates')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY environment variable)')
    parser.add_argument('--sound-library', help='Path to the sound library directory')
    
    args = parser.parse_args()
    
    # Initialize the tool
    scaffold = SoundScaffold(api_key=args.api_key, sound_library_path=args.sound_library)
    
    # Load the description
    description = scaffold.load_description(args.input)
    if "error" in description:
        print(f"Error loading description: {description['error']}")
        sys.exit(1)
    
    # Analyze the description
    analysis = scaffold.analyze_description(description)
    if "error" in analysis:
        print(f"Error analyzing description: {analysis['error']}")
        sys.exit(1)
    
    # Generate the audio template
    result = scaffold.generate_audio_template(args.output)
    if "error" in result:
        print(f"Error generating audio template: {result['error']}")
        sys.exit(1)
    
    # Print summary
    print(f"\nGenerated audio template for: {result['title']}")
    print(f"Output directory: {args.output}")
    print(f"\nLayers:")
    for layer in result.get("layers", []):
        print(f"  - {layer['name']}: {', '.join(layer['elements'][:3])}{'...' if len(layer['elements']) > 3 else ''}")
    print(f"\nFiles:")
    for file in result.get("files", []):
        print(f"  - {os.path.basename(file)}")
    if "mixed_file" in result:
        print(f"  - {os.path.basename(result['mixed_file'])} (mixed)")
    print(f"  - {os.path.basename(result.get('metadata_file', ''))} (metadata)")

if __name__ == "__main__":
    main()
