#!/usr/bin/env python3
"""
StoryboardGen - Generates storyboards from script segments

This tool converts script segments into visual storyboard sequences for pre-production planning.

Usage:
    python storyboard_gen.py --input <script_file> --output <output_dir> [options]
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
    from nltk.tokenize import sent_tokenize
    import yaml
except ImportError:
    logging.error("Required NLP libraries not installed. Run: pip install nltk pyyaml")
    sys.exit(1)

# For image generation and processing
try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import base64
    import io
except ImportError:
    logging.error("Required image libraries not installed. Run: pip install pillow numpy")
    sys.exit(1)

# For Google Cloud and Gemini API integration
try:
    import google.generativeai as genai
    from google.cloud import storage
except ImportError:
    logging.warning("Google Cloud libraries not installed. Some features will be disabled.")

# Download necessary NLTK data packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class StoryboardGen:
    """Generates storyboards from script segments."""

    def __init__(self, api_key: Optional[str] = None, style_guide_path: Optional[str] = None):
        """Initialize the StoryboardGen tool.
        
        Args:
            api_key: Gemini API key. If not provided, attempts to use GEMINI_API_KEY env variable.
            style_guide_path: Path to the character and style guide file (JSON or YAML)
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('StoryboardGen')
        
        # Initialize API key
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            self.logger.warning("No Gemini API key provided. Image generation will be limited to placeholders.")
        else:
            genai.configure(api_key=self.api_key)
        
        # Initialize Gemini model if available
        self.gemini_model = None
        if self.api_key:
            try:
                self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
                self.logger.info("Initialized Gemini model")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini model: {e}")
        
        # Load style guide if provided
        self.style_guide = self._load_style_guide(style_guide_path) if style_guide_path else {}
        
        # Store results
        self.script_data = {}
        self.storyboard_frames = []
        self.scene_segments = []
    
    def _load_style_guide(self, style_guide_path: str) -> Dict[str, Any]:
        """Load character and style guide from file.
        
        Args:
            style_guide_path: Path to the style guide file (JSON or YAML)
            
        Returns:
            Dictionary containing style guide data
        """
        try:
            with open(style_guide_path, 'r') as f:
                file_ext = os.path.splitext(style_guide_path)[1].lower()
                
                if file_ext == '.json':
                    guide = json.load(f)
                elif file_ext in ('.yaml', '.yml'):
                    guide = yaml.safe_load(f)
                else:
                    self.logger.error(f"Unsupported style guide format: {file_ext}")
                    return {}
            
            self.logger.info(f"Loaded style guide from {style_guide_path}")
            return guide
        except Exception as e:
            self.logger.error(f"Failed to load style guide: {e}")
            return {}
    
    def load_script(self, script_file: str) -> Dict[str, Any]:
        """Load script from file.
        
        Args:
            script_file: Path to the script file (JSON, YAML, or TXT)
            
        Returns:
            Dictionary containing script data
        """
        try:
            with open(script_file, 'r') as f:
                file_ext = os.path.splitext(script_file)[1].lower()
                
                if file_ext == '.json':
                    data = json.load(f)
                elif file_ext in ('.yaml', '.yml'):
                    data = yaml.safe_load(f)
                else:  # Treat as plain text
                    text = f.read()
                    data = {"script": text, "title": os.path.basename(script_file)}
            
            self.logger.info(f"Loaded script from {script_file}")
            self.script_data = data
            return data
        except Exception as e:
            self.logger.error(f"Error loading script from {script_file}: {e}")
            return {"error": str(e)}
    
    def segment_script(self, max_segments: int = 10) -> List[Dict[str, Any]]:
        """Segment the script into meaningful chunks for storyboard generation.
        
        Args:
            max_segments: Maximum number of segments to create
            
        Returns:
            List of dictionaries containing segmented script data
        """
        if not self.script_data:
            self.logger.error("No script data loaded. Call load_script() first.")
            return []
        
        # Extract script text
        if isinstance(self.script_data, dict) and "script" in self.script_data:
            script_text = self.script_data["script"]
        elif isinstance(self.script_data, str):
            script_text = self.script_data
        else:
            self.logger.error("Invalid script data format")
            return []
        
        # Simple segmentation by sentences first
        try:
            sentences = sent_tokenize(script_text)
            
            # Group sentences into segments
            segment_size = max(1, len(sentences) // max_segments)
            segments = []
            
            for i in range(0, len(sentences), segment_size):
                segment_text = " ".join(sentences[i:i + segment_size])
                
                # Skip empty segments
                if not segment_text.strip():
                    continue
                
                segment = {
                    "id": len(segments) + 1,
                    "text": segment_text,
                    "start_index": i,
                    "end_index": min(i + segment_size, len(sentences))
                }
                segments.append(segment)
                
                if len(segments) >= max_segments:
                    break
            
            self.logger.info(f"Segmented script into {len(segments)} parts")
            self.scene_segments = segments
            return segments
            
        except Exception as e:
            self.logger.error(f"Error segmenting script: {e}")
            return []
    
    def generate_storyboard_frames(self, output_dir: str) -> List[Dict[str, Any]]:
        """Generate storyboard frames from segmented script.
        
        Args:
            output_dir: Directory to save the generated storyboard frames
            
        Returns:
            List of dictionaries containing storyboard frame data
        """
        if not self.scene_segments:
            self.logger.error("No scene segments available. Call segment_script() first.")
            return []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        frames = []
        
        for i, segment in enumerate(self.scene_segments):
            self.logger.info(f"Generating storyboard frame {i+1}/{len(self.scene_segments)}")
            
            # Generate image for this segment
            frame_data = self._generate_frame(segment, i+1, output_dir)
            frames.append(frame_data)
        
        # Save metadata
        metadata = {
            "title": self.script_data.get("title", "Untitled"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": frames,
            "style_guide": self.style_guide
        }
        
        metadata_file = os.path.join(output_dir, "storyboard_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Generated {len(frames)} storyboard frames")
        self.storyboard_frames = frames
        return frames
    
    def _generate_frame(self, segment: Dict[str, Any], frame_number: int, output_dir: str) -> Dict[str, Any]:
        """Generate a single storyboard frame for a script segment.
        
        Args:
            segment: Dictionary containing segment data
            frame_number: Frame number (for filename)
            output_dir: Directory to save the frame image
            
        Returns:
            Dictionary containing frame data
        """
        # Base frame data
        frame_data = {
            "id": frame_number,
            "segment_id": segment["id"],
            "text": segment["text"],
            "filename": f"frame_{frame_number:03d}.png",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # File path for the frame image
        image_path = os.path.join(output_dir, frame_data["filename"])
        
        # Use Gemini to generate frame if available
        if self.gemini_model:
            try:
                # Extract style guide information if available
                style_info = ""
                if self.style_guide:
                    if "characters" in self.style_guide:
                        character_desc = ", ".join([f"{name}: {desc}" for name, desc in 
                                             self.style_guide["characters"].items()])
                        style_info += f"Characters: {character_desc}. "
                    
                    if "style" in self.style_guide:
                        style_info += f"Visual style: {self.style_guide['style']}. "
                
                # Prepare the prompt
                prompt = f"""
Create a storyboard frame based on this script segment:

"{segment['text']}"

{style_info}

Style: Detailed storyboard sketch, clean lines, suitable for production planning.
"""
                
                # Call Gemini API to get image
                response = self.gemini_model.generate_content(prompt)
                
                # Parse response to get image
                # This is a simplified approach - in a real implementation,
                # we'd properly handle the image data from Gemini's response
                # For now, we'll create a placeholder
                
                # Since we can't actually get an image from the current Gemini API in this form,
                # we'll create a placeholder with text from the response
                placeholder = self._create_placeholder_image(segment, frame_number, response.text[:100])
                placeholder.save(image_path)
                
                self.logger.info(f"Generated storyboard frame {frame_number} with Gemini API")
                
            except Exception as e:
                self.logger.error(f"Error generating frame with Gemini: {e}")
                placeholder = self._create_placeholder_image(segment, frame_number)
                placeholder.save(image_path)
        else:
            # Create placeholder image
            placeholder = self._create_placeholder_image(segment, frame_number)
            placeholder.save(image_path)
        
        # Add image path to frame data
        frame_data["image_path"] = image_path
        
        return frame_data
    
    def _create_placeholder_image(self, segment: Dict[str, Any], frame_number: int, 
                                 description: str = None) -> Image.Image:
        """Create a placeholder storyboard frame image.
        
        Args:
            segment: Dictionary containing segment data
            frame_number: Frame number to display
            description: Optional text description to include
            
        Returns:
            PIL Image object with the placeholder frame
        """
        # Create a blank image (16:9 aspect ratio, 1280x720 pixels)
        width, height = 1280, 720
        image = Image.new('RGB', (width, height), (240, 240, 240))
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            # Use a built-in font if available
            title_font = ImageFont.truetype("Arial", 36)
            text_font = ImageFont.truetype("Arial", 24)
        except IOError:
            # Fall back to default font
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
        
        # Draw border
        draw.rectangle([10, 10, width - 10, height - 10], outline="black", width=2)
        
        # Draw frame number
        draw.text((30, 30), f"Frame {frame_number}", fill="black", font=title_font)
        
        # Draw segment text (wrapped to fit)
        text = segment["text"]
        wrapped_text = self._wrap_text(text, text_font, width - 100)
        draw.text((30, 100), wrapped_text, fill="black", font=text_font)
        
        # Draw description if provided
        if description:
            description_y = height - 120
            draw.text((30, description_y), "Description:", fill="black", font=title_font)
            wrapped_desc = self._wrap_text(description, text_font, width - 100)
            draw.text((30, description_y + 50), wrapped_desc, fill="black", font=text_font)
        
        return image
    
    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
        """Wrap text to fit within a given width.
        
        Args:
            text: Text to wrap
            font: Font to use for measuring text width
            max_width: Maximum width in pixels
            
        Returns:
            String with newlines inserted for wrapping
        """
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Add word to current line
            current_line.append(word)
            
            # Check if current line is too long
            line_text = " ".join(current_line)
            try:
                # For newer PIL versions with getbbox
                if hasattr(font, "getbbox"):
                    line_width = font.getbbox(line_text)[2]
                # For older PIL versions
                else:
                    line_width, _ = font.getsize(line_text)
            except AttributeError:
                # If all else fails, estimate based on character count
                line_width = len(line_text) * 12  # Rough estimate
            
            # If too long, move to next line
            if line_width > max_width and len(current_line) > 1:
                current_line.pop()  # Remove last word
                lines.append(" ".join(current_line))
                current_line = [word]  # Start new line with the word that didn't fit
        
        # Add the last line
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)
    
    def create_storyboard_pdf(self, output_dir: str, output_filename: str = "storyboard.pdf") -> str:
        """Create a PDF with all storyboard frames.
        
        Args:
            output_dir: Directory containing the storyboard frames
            output_filename: Name of the output PDF file
            
        Returns:
            Path to the created PDF file, or empty string if failed
        """
        if not self.storyboard_frames:
            self.logger.error("No storyboard frames available. Call generate_storyboard_frames() first.")
            return ""
        
        try:
            from reportlab.lib.pagesizes import letter, landscape
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import ImageReader
        except ImportError:
            self.logger.error("ReportLab not installed. Run: pip install reportlab")
            return ""
        
        pdf_path = os.path.join(output_dir, output_filename)
        
        try:
            # Create a PDF with landscape orientation
            c = canvas.Canvas(pdf_path, pagesize=landscape(letter))
            width, height = landscape(letter)
            
            for frame in self.storyboard_frames:
                image_path = frame["image_path"]
                
                if os.path.exists(image_path):
                    # Add image to the PDF
                    img = ImageReader(image_path)
                    
                    # Calculate image size to fit the page
                    img_width, img_height = 720, 405  # 16:9 ratio
                    x = (width - img_width) / 2
                    y = height - 100 - img_height  # Leave space for text
                    
                    c.drawImage(img, x, y, width=img_width, height=img_height)
                    
                    # Add frame number and caption
                    c.setFont("Helvetica", 14)
                    c.drawString(x, y - 20, f"Frame {frame['id']}")
                    
                    # Add script text (truncated if necessary)
                    c.setFont("Helvetica", 10)
                    text = frame["text"]
                    if len(text) > 200:
                        text = text[:197] + "..."
                    text_lines = self._split_text_for_pdf(text, 100)  # ~100 chars per line
                    
                    for i, line in enumerate(text_lines[:3]):  # Max 3 lines
                        c.drawString(x, y - 40 - (i * 15), line)
                    
                    # Add a new page for the next frame
                    c.showPage()
            
            c.save()
            self.logger.info(f"Created storyboard PDF at {pdf_path}")
            return pdf_path
            
        except Exception as e:
            self.logger.error(f"Error creating storyboard PDF: {e}")
            return ""
    
    def _split_text_for_pdf(self, text: str, chars_per_line: int) -> List[str]:
        """Split text into lines for PDF rendering.
        
        Args:
            text: Text to split
            chars_per_line: Approximate characters per line
            
        Returns:
            List of text lines
        """
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for the space
            
            if current_length + word_length > chars_per_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(" ".join(current_line))
            
        return lines

def main():
    """Main function to run the tool from command line."""
    parser = argparse.ArgumentParser(description='Generate storyboards from script segments')
    parser.add_argument('--input', required=True, help='Path to the script file (JSON, YAML, or TXT)')
    parser.add_argument('--output', required=True, help='Path to the output directory for generated storyboards')
    parser.add_argument('--style-guide', help='Path to the character and style guide file (JSON or YAML)')
    parser.add_argument('--max-segments', type=int, default=10, help='Maximum number of segments to create')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY environment variable)')
    parser.add_argument('--create-pdf', action='store_true', help='Create a PDF with all storyboard frames')
    
    args = parser.parse_args()
    
    # Initialize the tool
    generator = StoryboardGen(api_key=args.api_key, style_guide_path=args.style_guide)
    
    # Load script
    script_data = generator.load_script(args.input)
    if "error" in script_data:
        print(f"Error loading script: {script_data['error']}")
        sys.exit(1)
    
    # Segment script
    segments = generator.segment_script(max_segments=args.max_segments)
    if not segments:
        print("Error segmenting script")
        sys.exit(1)
    
    # Generate storyboard frames
    frames = generator.generate_storyboard_frames(args.output)
    if not frames:
        print("Error generating storyboard frames")
        sys.exit(1)
    
    # Create PDF if requested
    if args.create_pdf:
        pdf_path = generator.create_storyboard_pdf(args.output)
        if pdf_path:
            print(f"Created storyboard PDF at: {pdf_path}")
    
    # Print summary
    print(f"\nGenerated storyboard for: {script_data.get('title', 'Untitled')}")
    print(f"Output directory: {args.output}")
    print(f"Number of frames: {len(frames)}")
    print("\nFrames:")
    for frame in frames:
        print(f"  - Frame {frame['id']}: {frame['filename']}")

if __name__ == "__main__":
    main()
