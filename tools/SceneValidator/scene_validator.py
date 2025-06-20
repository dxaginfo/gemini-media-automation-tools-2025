#!/usr/bin/env python3
"""
SceneValidator - Validates scene elements for continuity and compliance

This tool analyzes media scenes to check for continuity, compliance with guidelines,
and technical specifications.

Usage:
    python scene_validator.py --input <input_file> --rules <rules_file> --output <output_file>
"""

import argparse
import json
import logging
import os
import sys
import yaml
from typing import Dict, List, Any, Optional, Union

# For Google Cloud and Gemini API integration
try:
    import google.generativeai as genai
    from google.cloud import storage
    from google.cloud import vision
except ImportError:
    logging.error("Required Google APIs not installed. Run: pip install google-cloud-storage google-cloud-vision google-generativeai")
    sys.exit(1)

class SceneValidator:
    """Validates scene elements for continuity and compliance."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the SceneValidator.
        
        Args:
            api_key: Gemini API key. If not provided, attempts to use GEMINI_API_KEY env variable.
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            logging.warning("No Gemini API key provided. Some features will be limited.")
        else:
            genai.configure(api_key=self.api_key)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('SceneValidator')
        
        # Initialize the models we'll use
        self.gemini_model = None
        if self.api_key:
            try:
                self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini model: {e}")
        
        # Store rules and validation results
        self.rules = {}
        self.validation_results = {}
    
    def load_rules(self, rules_file: str) -> bool:
        """Load validation rules from a file.
        
        Args:
            rules_file: Path to the rules file (JSON or YAML)
            
        Returns:
            bool: True if rules were successfully loaded, False otherwise
        """
        try:
            with open(rules_file, 'r') as f:
                if rules_file.endswith('.json'):
                    self.rules = json.load(f)
                elif rules_file.endswith(('.yaml', '.yml')):
                    self.rules = yaml.safe_load(f)
                else:
                    self.logger.error(f"Unsupported rules file format: {rules_file}")
                    return False
                
            self.logger.info(f"Loaded {len(self.rules)} rules from {rules_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load rules from {rules_file}: {e}")
            return False
    
    def load_scene_data(self, input_file: str) -> Dict[str, Any]:
        """Load scene data from a file.
        
        Args:
            input_file: Path to the input file (JSON or YAML)
            
        Returns:
            Dict containing scene data or empty dict if load failed
        """
        try:
            with open(input_file, 'r') as f:
                if input_file.endswith('.json'):
                    return json.load(f)
                elif input_file.endswith(('.yaml', '.yml')):
                    return yaml.safe_load(f)
                else:
                    self.logger.error(f"Unsupported input file format: {input_file}")
                    return {}
        except Exception as e:
            self.logger.error(f"Failed to load scene data from {input_file}: {e}")
            return {}
    
    def validate_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scene data against loaded rules.
        
        Args:
            scene_data: Dictionary containing scene data to validate
            
        Returns:
            Dict containing validation results
        """
        if not self.rules:
            self.logger.error("No rules loaded. Call load_rules() first.")
            return {"error": "No rules loaded"}
        
        if not scene_data:
            self.logger.error("No scene data provided.")
            return {"error": "No scene data provided"}
        
        # Initialize results
        results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "scene_id": scene_data.get("scene_id", "unknown"),
            "rule_compliance": {}
        }
        
        # Apply each rule
        for rule_name, rule_config in self.rules.items():
            rule_result = self._apply_rule(rule_name, rule_config, scene_data)
            results["rule_compliance"][rule_name] = rule_result
            
            # If any rule fails, the scene is not valid
            if rule_result.get("valid") is False:
                results["valid"] = False
                results["issues"].append({
                    "rule": rule_name,
                    "message": rule_result.get("message", "Rule validation failed")
                })
            
            # Collect warnings
            if rule_result.get("warnings"):
                for warning in rule_result["warnings"]:
                    results["warnings"].append({
                        "rule": rule_name,
                        "message": warning
                    })
        
        # Store results
        self.validation_results = results
        return results
    
    def _apply_rule(self, rule_name: str, rule_config: Dict[str, Any], scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific rule to scene data.
        
        Args:
            rule_name: Name of the rule
            rule_config: Rule configuration
            scene_data: Scene data to validate
            
        Returns:
            Dict containing rule validation results
        """
        rule_type = rule_config.get("type")
        result = {"valid": True, "warnings": []}
        
        if rule_type == "field_required":
            field = rule_config.get("field")
            if field not in scene_data or scene_data[field] is None:
                result["valid"] = False
                result["message"] = f"Required field '{field}' is missing"
        
        elif rule_type == "field_value_in_set":
            field = rule_config.get("field")
            allowed_values = rule_config.get("allowed_values", [])
            
            if field in scene_data and scene_data[field] not in allowed_values:
                result["valid"] = False
                result["message"] = f"Field '{field}' value '{scene_data[field]}' not in allowed values: {allowed_values}"
        
        elif rule_type == "numerical_range":
            field = rule_config.get("field")
            min_value = rule_config.get("min")
            max_value = rule_config.get("max")
            
            if field in scene_data:
                value = scene_data[field]
                if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                    result["valid"] = False
                    result["message"] = f"Field '{field}' value {value} outside allowed range ({min_value}, {max_value})"
        
        elif rule_type == "continuity_check":
            # For continuity checks, we often need to refer to previous scenes
            # This is just a placeholder for actual continuity logic
            field = rule_config.get("field")
            if field not in scene_data:
                result["warnings"].append(f"Continuity field '{field}' not found, cannot verify continuity")
        
        elif rule_type == "gemini_analysis" and self.gemini_model:
            # This would use Gemini to analyze the scene
            # Placeholder for actual Gemini API call
            prompt = rule_config.get("prompt", "")
            field = rule_config.get("field", "")
            
            if field in scene_data:
                # In a real implementation, we would call the Gemini API here
                # For now, we'll just add a placeholder warning
                result["warnings"].append(f"Gemini analysis on '{field}' would be performed here")
        
        else:
            result["warnings"].append(f"Unknown rule type: {rule_type}")
        
        return result
    
    def save_results(self, output_file: str) -> bool:
        """Save validation results to a file.
        
        Args:
            output_file: Path to save results to
            
        Returns:
            bool: True if results were successfully saved, False otherwise
        """
        if not self.validation_results:
            self.logger.error("No validation results to save. Run validate_scene() first.")
            return False
        
        try:
            with open(output_file, 'w') as f:
                if output_file.endswith('.json'):
                    json.dump(self.validation_results, f, indent=2)
                elif output_file.endswith(('.yaml', '.yml')):
                    yaml.dump(self.validation_results, f)
                else:
                    self.logger.error(f"Unsupported output file format: {output_file}")
                    return False
            
            self.logger.info(f"Saved validation results to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save results to {output_file}: {e}")
            return False

def main():
    """Main function to run the validator from command line."""
    parser = argparse.ArgumentParser(description='Validate scene elements for continuity and compliance')
    parser.add_argument('--input', required=True, help='Path to the input scene data file (JSON or YAML)')
    parser.add_argument('--rules', required=True, help='Path to the rules file (JSON or YAML)')
    parser.add_argument('--output', required=True, help='Path to save validation results (JSON or YAML)')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = SceneValidator(api_key=args.api_key)
    
    # Load rules
    if not validator.load_rules(args.rules):
        sys.exit(1)
    
    # Load scene data
    scene_data = validator.load_scene_data(args.input)
    if not scene_data:
        sys.exit(1)
    
    # Validate scene
    results = validator.validate_scene(scene_data)
    
    # Save results
    if not validator.save_results(args.output):
        sys.exit(1)
    
    # Print summary
    if results["valid"]:
        print(f"Scene {results['scene_id']} is valid.")
    else:
        print(f"Scene {results['scene_id']} is invalid. Found {len(results['issues'])} issues.")
    
    if results["warnings"]:
        print(f"Warnings: {len(results['warnings'])}")

if __name__ == "__main__":
    main()
