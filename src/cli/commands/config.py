"""
Configuration command module.

This module provides CLI commands for configuration management.
"""

import argparse
from pathlib import Path
from typing import Any

from ...evtol.core.config import ConfigManager


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add config parser to subparsers."""
    parser = subparsers.add_parser(
        "config",
        help="Configuration commands"
    )
    
    config_subparsers = parser.add_subparsers(
        dest="config_command",
        help="Available configuration commands"
    )
    
    # Show config command
    show_parser = config_subparsers.add_parser(
        "show",
        help="Show current configuration"
    )
    show_parser.add_argument(
        "--layer", "-l",
        help="Show configuration for specific layer"
    )
    show_parser.set_defaults(func=show_config)
    
    # Validate config command
    validate_parser = config_subparsers.add_parser(
        "validate",
        help="Validate configuration file"
    )
    validate_parser.add_argument(
        "--file", "-f",
        required=True,
        type=Path,
        help="Configuration file to validate"
    )
    validate_parser.set_defaults(func=validate_config)
    
    # Generate config command
    generate_parser = config_subparsers.add_parser(
        "generate",
        help="Generate default configuration"
    )
    generate_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output configuration file"
    )
    generate_parser.set_defaults(func=generate_config)


def show_config(args: argparse.Namespace) -> int:
    """Show current configuration."""
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    if args.layer:
        layer_config = config_manager.get_layer_config(args.layer)
        print(f"Configuration for layer '{args.layer}':")
        for key, value in layer_config.items():
            print(f"  {key}: {value}")
    else:
        print("System configuration:")
        for key, value in config.__dict__.items():
            print(f"  {key}: {value}")
    
    return 0


def validate_config(args: argparse.Namespace) -> int:
    """Validate configuration file."""
    print(f"Validating configuration file: {args.file}")
    
    try:
        config_manager = ConfigManager()
        config_manager.load_config(args.file)
        print("Configuration is valid!")
        return 0
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return 1


def generate_config(args: argparse.Namespace) -> int:
    """Generate default configuration."""
    output_file = args.output or Path("config/default.yaml")
    
    print(f"Generating default configuration: {output_file}")
    
    config_manager = ConfigManager()
    config_manager.save_config(output_file)
    
    print("Default configuration generated!")
    return 0


