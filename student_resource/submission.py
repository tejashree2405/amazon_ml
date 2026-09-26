import os
import subprocess
import zipfile
import sys

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root_dir, "output")
    matching_path = os.path.join(out_dir, "matching_results.tsv")
    candidate_path = os.path.join(out_dir, "candidate_pairs.tsv")
    zip_path = os.path.join(root_dir, "submission.zip")
    
    if not os.path.exists(matching_path) or not os.path.exists(candidate_path):
        print("Output files not found! Please run inference.py first.")
        sys.exit(1)
    
    # Run validator
    print("Running submission validator...")
    cmd = [
        sys.executable, "utils/validate_submission.py",
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", "dataset/test"
    ]
    res = subprocess.run(cmd, cwd=root_dir)
    
    if res.returncode != 0:
        print("\nValidation failed! Please fix the errors listed above before submitting.")
        sys.exit(1)
        
    print("\nValidation passed. Creating submission.zip...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(matching_path, "output/matching_results.tsv")
        zf.write(candidate_path, "output/candidate_pairs.tsv")
        
    print(f"Submission ready: {zip_path}")

if __name__ == "__main__":
    main()
