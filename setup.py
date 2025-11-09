"""
Setup script for Stress Prediction Project
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✓ Requirements installed!")

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'saved_models', 'data/S2']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    print("=" * 60)
    print("Stress Prediction Project Setup")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Install required packages")
    print("2. Create necessary directories")
    print("\n" + "=" * 60 + "\n")
    
    try:
        install_requirements()
        create_directories()
        print("\n" + "=" * 60)
        print("✓ Setup complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Download dataset: python download_data.py")
        print("2. Run app: streamlit run app.py")
        print("\n" + "=" * 60)
    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()

