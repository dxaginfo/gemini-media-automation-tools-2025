#!/usr/bin/env python3
"""
LoopOptimizer - Optimizes loop points in audio/video content

This tool analyzes audio and video content to identify optimal loop points for seamless playback.

Usage:
    python loop_optimizer.py --input <input_file> --output <output_file> [options]
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# For media processing
try:
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import correlate
    import librosa
    import librosa.display
    import soundfile as sf
    # FFmpeg Python wrapper for video processing
    import ffmpeg
except ImportError:
    logging.error("Required media processing libraries not installed. Run: pip install numpy scipy librosa soundfile ffmpeg-python")
    sys.exit(1)

# For Google Cloud integration
try:
    from google.cloud import storage
except ImportError:
    logging.warning("Google Cloud Storage not installed. Cloud storage features will be disabled.")

class LoopOptimizer:
    """Optimizes loop points in audio/video content."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the LoopOptimizer.
        
        Args:
            config_file: Optional path to configuration file
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('LoopOptimizer')
        
        # Default configuration
        self.config = {
            "min_loop_length_seconds": 1.0,
            "max_loop_length_seconds": 30.0,
            "correlation_threshold": 0.75,
            "fade_length_seconds": 0.1,
            "analyze_video": False,
            "temp_directory": None,
            "output_formats": ["wav", "mp3"],
            "include_metadata": True
        }
        
        # Load custom configuration if provided
        if config_file:
            self._load_config(config_file)
        
        # Results storage
        self.results = {}
        self.loop_points = []
        self.input_file = None
        self.output_file = None
    
    def _load_config(self, config_file: str) -> None:
        """Load configuration from file.
        
        Args:
            config_file: Path to configuration JSON file
        """
        try:
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                self.config.update(custom_config)
            self.logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    def analyze_audio(self, audio_file: str) -> List[Dict[str, Any]]:
        """Analyze audio file to find optimal loop points.
        
        Args:
            audio_file: Path to the audio file to analyze
            
        Returns:
            List of dictionaries containing potential loop points and their quality scores
        """
        self.logger.info(f"Analyzing audio file: {audio_file}")
        
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            self.logger.info(f"Loaded audio: {duration:.2f} seconds, {sr} Hz sample rate")
            
            # Minimum and maximum loop lengths in samples
            min_loop_samples = int(self.config["min_loop_length_seconds"] * sr)
            max_loop_samples = int(min(self.config["max_loop_length_seconds"], duration * 0.8) * sr)
            
            # List to store potential loop points
            potential_loops = []
            
            # Analyze segments of different lengths
            for segment_length in np.linspace(min_loop_samples, max_loop_samples, 10, dtype=int):
                # Convert to seconds for logging
                segment_seconds = segment_length / sr
                self.logger.info(f"Analyzing segments of length {segment_seconds:.2f} seconds")
                
                # For very long files, we may need to chunk our analysis
                max_offset = len(y) - segment_length
                step = max(1, int(segment_length / 10))  # Analyze at ~10 positions
                
                for offset in range(0, max_offset, step):
                    segment = y[offset:offset + segment_length]
                    
                    # Look for segments that loop well with themselves
                    correlation = self._calculate_loop_correlation(segment)
                    
                    # If correlation is above threshold, this is a potential loop point
                    if correlation > self.config["correlation_threshold"]:
                        loop_info = {
                            "start_sample": offset,
                            "end_sample": offset + segment_length,
                            "start_time": offset / sr,
                            "end_time": (offset + segment_length) / sr,
                            "duration": segment_length / sr,
                            "correlation": correlation,
                            "quality_score": self._calculate_quality_score(segment, correlation)
                        }
                        potential_loops.append(loop_info)
            
            # Sort loops by quality score, highest first
            potential_loops.sort(key=lambda x: x["quality_score"], reverse=True)
            
            # Take the top 5 loops
            best_loops = potential_loops[:5]
            
            self.logger.info(f"Found {len(potential_loops)} potential loops, selected top {len(best_loops)}")
            self.loop_points = best_loops
            
            return best_loops
            
        except Exception as e:
            self.logger.error(f"Error analyzing audio file: {e}")
            return []
    
    def _calculate_loop_correlation(self, segment: np.ndarray) -> float:
        """Calculate how well a segment loops with itself.
        
        Args:
            segment: Audio segment as numpy array
            
        Returns:
            Correlation value between 0 and 1
        """
        # Take the first and last 10% of the segment for comparison
        segment_length = len(segment)
        comparison_length = int(segment_length * 0.1)
        
        start_segment = segment[:comparison_length]
        end_segment = segment[-comparison_length:]
        
        # Calculate cross-correlation
        correlation = np.corrcoef(start_segment, end_segment)[0, 1]
        
        # Handle NaN values (occurs if one segment is constant)
        if np.isnan(correlation):
            correlation = 0.0
        
        return correlation
    
    def _calculate_quality_score(self, segment: np.ndarray, correlation: float) -> float:
        """Calculate overall quality score for a potential loop.
        
        Args:
            segment: Audio segment as numpy array
            correlation: Base correlation score
            
        Returns:
            Quality score between 0 and 1
        """
        # Factors to consider in the quality score:
        # 1. Base correlation between start and end
        # 2. Lack of silence or very quiet sections (could indicate bad loop points)
        # 3. Consistency of audio energy throughout segment
        
        # Check for silence or very quiet sections
        rms_energy = librosa.feature.rms(y=segment)[0]
        if np.mean(rms_energy) < 0.01:  # Very quiet segment
            silence_penalty = 0.5
        else:
            silence_penalty = 0.0
        
        # Check energy consistency
        energy_variance = np.var(rms_energy) / np.mean(rms_energy) if np.mean(rms_energy) > 0 else 1.0
        consistency_score = 1.0 / (1.0 + energy_variance)  # Higher variance = lower score
        
        # Combine factors
        quality_score = (correlation * 0.6) + (consistency_score * 0.4) - silence_penalty
        
        return max(0.0, min(1.0, quality_score))  # Ensure score is between 0 and 1
    
    def create_looped_audio(self, audio_file: str, output_file: str, loop_info: Dict[str, Any]) -> bool:
        """Create a looped version of the audio based on identified loop points.
        
        Args:
            audio_file: Path to the original audio file
            output_file: Path to save the looped audio file
            loop_info: Dictionary containing loop point information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the audio file
            y, sr = librosa.load(audio_file, sr=None)
            
            # Extract the loop segment
            start_sample = loop_info["start_sample"]
            end_sample = loop_info["end_sample"]
            loop_segment = y[start_sample:end_sample]
            
            # Create a looped version (3 repetitions for example)
            # In a real implementation, we'd add crossfading between loops
            num_loops = 3
            looped_audio = np.tile(loop_segment, num_loops)
            
            # Apply fade in/out for smoothness
            fade_samples = int(self.config["fade_length_seconds"] * sr)
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            # Apply fade in
            looped_audio[:fade_samples] *= fade_in
            
            # Apply fade out
            looped_audio[-fade_samples:] *= fade_out
            
            # Save the result
            sf.write(output_file, looped_audio, sr)
            
            self.logger.info(f"Created looped audio file: {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error creating looped audio: {e}")
            return False
    
    def process_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Process an audio/video file to find and apply optimal loop points.
        
        Args:
            input_file: Path to the input media file
            output_file: Path to save the output files
            
        Returns:
            Dictionary containing processing results and metadata
        """
        self.input_file = input_file
        self.output_file = output_file
        
        # Check input file exists
        if not os.path.exists(input_file):
            self.logger.error(f"Input file not found: {input_file}")
            return {"error": "Input file not found"}
        
        # Determine if input is audio or video
        is_video = self._is_video_file(input_file)
        
        # For video files, extract audio for analysis
        temp_audio_file = None
        audio_file_for_analysis = input_file
        
        if is_video:
            self.logger.info(f"Input is video. Extracting audio for analysis.")
            temp_audio_file = self._extract_audio_from_video(input_file)
            if not temp_audio_file:
                return {"error": "Failed to extract audio from video"}
            audio_file_for_analysis = temp_audio_file
        
        # Analyze audio to find loop points
        loop_points = self.analyze_audio(audio_file_for_analysis)
        
        if not loop_points:
            self.logger.warning("No suitable loop points found")
            result = {
                "status": "warning",
                "message": "No suitable loop points found",
                "input_file": input_file,
                "is_video": is_video
            }
        else:
            # Get the best loop point
            best_loop = loop_points[0]
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create looped audio
            output_base = os.path.splitext(output_file)[0]
            looped_audio_file = f"{output_base}_looped.wav"
            self.create_looped_audio(audio_file_for_analysis, looped_audio_file, best_loop)
            
            # For video, we would reattach the looped audio to the video
            if is_video and self.config["analyze_video"]:
                video_with_loop = self._create_looped_video(input_file, looped_audio_file, best_loop)
                if video_with_loop:
                    self.logger.info(f"Created video with looped audio: {video_with_loop}")
            
            # Save metadata
            metadata_file = f"{output_base}_metadata.json"
            metadata = {
                "input_file": input_file,
                "is_video": is_video,
                "loop_points": loop_points,
                "best_loop": best_loop,
                "config": self.config
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            result = {
                "status": "success",
                "message": "Successfully identified loop points and created looped media",
                "input_file": input_file,
                "is_video": is_video,
                "loop_points": loop_points,
                "best_loop": best_loop,
                "output_files": {
                    "looped_audio": looped_audio_file,
                    "metadata": metadata_file
                }
            }
            
            if is_video and self.config["analyze_video"]:
                result["output_files"]["looped_video"] = f"{output_base}_looped.mp4"
        
        # Clean up temporary files
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
        
        self.results = result
        return result
    
    def _is_video_file(self, file_path: str) -> bool:
        """Determine if a file is a video file based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if the file is a video, False otherwise
        """
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]
        _, ext = os.path.splitext(file_path.lower())
        return ext in video_extensions
    
    def _extract_audio_from_video(self, video_file: str) -> Optional[str]:
        """Extract audio from a video file.
        
        Args:
            video_file: Path to the video file
            
        Returns:
            Path to the extracted audio file, or None if extraction failed
        """
        try:
            # Create temporary file for audio
            temp_dir = self.config["temp_directory"] or tempfile.gettempdir()
            temp_audio_file = os.path.join(temp_dir, f"temp_audio_{os.path.basename(video_file)}.wav")
            
            # Use ffmpeg to extract audio
            (ffmpeg
                .input(video_file)
                .output(temp_audio_file, acodec='pcm_s16le', ac=2, ar='44100')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            self.logger.info(f"Extracted audio to temporary file: {temp_audio_file}")
            return temp_audio_file
        
        except Exception as e:
            self.logger.error(f"Error extracting audio from video: {e}")
            return None
    
    def _create_looped_video(self, video_file: str, looped_audio_file: str, loop_info: Dict[str, Any]) -> Optional[str]:
        """Create a video with looped audio.
        
        This is a placeholder - in a real implementation, this would be more sophisticated.
        
        Args:
            video_file: Path to the original video file
            looped_audio_file: Path to the looped audio file
            loop_info: Dictionary containing loop point information
            
        Returns:
            Path to the created video file, or None if creation failed
        """
        try:
            output_video = os.path.splitext(self.output_file)[0] + "_looped.mp4"
            
            # Use ffmpeg to combine original video with looped audio
            # This is a simple implementation; a real one would be more complex
            (ffmpeg
                .input(video_file)
                .input(looped_audio_file)
                .output(output_video, map='0:v', map='1:a', shortest=None, codec='copy')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            return output_video
        
        except Exception as e:
            self.logger.error(f"Error creating looped video: {e}")
            return None

def main():
    """Main function to run the optimizer from command line."""
    parser = argparse.ArgumentParser(description='Optimize loop points in audio/video content')
    parser.add_argument('--input', required=True, help='Path to the input media file')
    parser.add_argument('--output', required=True, help='Path to save the output files')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = LoopOptimizer(config_file=args.config)
    
    # Process file
    result = optimizer.process_file(args.input, args.output)
    
    # Print result summary
    if result.get("status") == "success":
        print(f"Successfully processed {args.input}")
        best_loop = result.get("best_loop", {})
        print(f"Best loop: {best_loop.get('start_time', 0):.2f}s to {best_loop.get('end_time', 0):.2f}s (score: {best_loop.get('quality_score', 0):.2f})")
        print(f"Output files:")
        for file_type, file_path in result.get("output_files", {}).items():
            print(f"  - {file_type}: {file_path}")
    else:
        print(f"Error: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()
