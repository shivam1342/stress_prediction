"""
Fixed WESAD extraction script (cross-platform)
Author: Shivam Singh (fixed by Aether)
"""

import os
import sys
import zipfile
import shutil


def extract_wesad_data(zip_path=None):
    print("=" * 60)
    print("WESAD Dataset Extraction Script")
    print("=" * 60)
    print("\nThis script will extract the S2 subject data from a zip file.")
    print("Make sure you have downloaded S2.zip manually from Kaggle.")
    print("\n" + "=" * 60 + "\n")

    # Always use os.path.join for portability
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    expected_s2_dir = os.path.join(data_dir, "S2")

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # --- Locate zip file ---
    if zip_path is None:
        possible_paths = [
            os.path.join(base_dir, "S2.zip"),
            os.path.join(data_dir, "S2.zip"),
            os.path.join(os.path.expanduser("~"), "Downloads", "S2.zip"),
            os.path.join(os.path.expanduser("~"), "Desktop", "S2.zip"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                zip_path = path
                print(f"✓ Found S2.zip at: {zip_path}")
                break

    if not zip_path or not os.path.exists(zip_path):
        print("✗ Could not find S2.zip.")
        return False

    # --- Extract ---
    print(f"\nExtracting: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"✓ Zip file contains {len(file_list)} files/folders")
            zip_ref.extractall(data_dir)
        print("✓ Extraction complete!")
    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return False

    # --- Locate extracted S2 folder or file ---
    s2_mat = None
    found_s2_dir = None
    for root, dirs, files in os.walk(data_dir):
        if "S2.mat" in files:
            s2_mat = os.path.join(root, "S2.mat")
        if "S2" in dirs:
            found_s2_dir = os.path.join(root, "S2")

    # --- Handle S2.mat ---
    if s2_mat:
        os.makedirs(expected_s2_dir, exist_ok=True)
        target = os.path.join(expected_s2_dir, "S2.mat")
        if s2_mat != target:
            shutil.move(s2_mat, target)
        print(f"✓ Found and moved S2.mat to: {target}")
        print("\n✓ Dataset ready to use (.mat format)")
        return True

    # --- Handle S2 directory ---
    if found_s2_dir:
        if os.path.abspath(found_s2_dir) != os.path.abspath(expected_s2_dir):
            os.makedirs(data_dir, exist_ok=True)
            if os.path.exists(expected_s2_dir):
                shutil.rmtree(expected_s2_dir)
            shutil.move(found_s2_dir, expected_s2_dir)
        print(f"✓ Found and moved S2 folder to: {expected_s2_dir}")
        print("\n✓ Dataset ready to use (CSV format)")
        return True

    print("✗ Could not find S2.mat or S2 directory after extraction.")
    return False


def check_data_exists():
    """Check if S2 data already exists (both .mat and CSV format)"""
    data_dir = os.path.join(os.getcwd(), "data")
    s2_mat = os.path.join(data_dir, "S2", "S2.mat")
    s2_dir = os.path.join(data_dir, "S2")
    if os.path.exists(s2_mat):
        return s2_mat
    elif os.path.exists(s2_dir):
        return s2_dir
    return None


if __name__ == "__main__":
    existing_path = check_data_exists()
    if existing_path:
        print(f"✓ S2 data already exists at: {existing_path}")
        ans = input("Do you want to re-extract? (y/n): ").strip().lower()
        if ans != "y":
            sys.exit(0)

    zip_path = sys.argv[1] if len(sys.argv) > 1 else None
    extract_wesad_data(zip_path)
