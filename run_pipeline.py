<<<<<<< HEAD
#!/usr/bin/env python3
"""
ToxLens AI Pipeline Runner
Connects all components of the drug toxicity prediction system
"""

import os
import sys
import subprocess
import time
import webbrowser
import argparse

def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run ToxLens AI Pipeline')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning before training')
    args = parser.parse_args()

    print("🚀 ToxLens AI Pipeline Runner")
    print("=" * 40)

    if not os.path.exists("data/combined.csv"):
        print("📊 Step 1: Preparing dataset...")
        if not run_command("python prepare_dataset.py"):
            print("Failed to prepare dataset")
            return

    if args.tune or not os.path.exists("best_hyperparameters.json"):
        print("⚙️ Step 2: Tuning hyperparameters...")
        if not run_command("python tune_gnn.py"):
            print("Failed to tune hyperparameters")
            return

    if not os.path.exists("toxicity_gnn_model.pth"):
        print("🧠 Step 3: Training GNN model...")
        if not run_command("python train_gnn.py"):
            print("Failed to train model")
            return

    print("✅ Step 4: Validating model...")
    if not run_command("python validate_model.py"):
        print("Model validation failed, but continuing...")

    print("🌐 Step 5: Starting backend server...")
    # Start backend in background
    backend_process = subprocess.Popen([
        sys.executable, "backend/app.py"
    ], cwd=os.getcwd())

    # Wait for backend to start
    time.sleep(3)

    print("🖥️  Step 6: Opening frontend...")
    frontend_path = os.path.join(os.getcwd(), "TOXLENS FRONTEND", "index.html")
    webbrowser.open(f"file://{frontend_path}")

    print("\n🎉 ToxLens AI is now running!")
    print("📱 Frontend: Open TOXLENS FRONTEND/index.html in your browser")
    print("🔗 Backend API: http://127.0.0.1:5000")
    print("🛑 Press Ctrl+C to stop the server")

    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        backend_process.terminate()
        backend_process.wait()

if __name__ == "__main__":
=======
#!/usr/bin/env python3
"""
ToxLens AI Pipeline Runner
Connects all components of the drug toxicity prediction system
"""

import os
import sys
import subprocess
import time
import webbrowser
import argparse

def run_command(cmd, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run ToxLens AI Pipeline')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning before training')
    args = parser.parse_args()

    print("🚀 ToxLens AI Pipeline Runner")
    print("=" * 40)

    if not os.path.exists("data/combined.csv"):
        print("📊 Step 1: Preparing dataset...")
        if not run_command("python prepare_dataset.py"):
            print("Failed to prepare dataset")
            return

    if args.tune or not os.path.exists("best_hyperparameters.json"):
        print("⚙️ Step 2: Tuning hyperparameters...")
        if not run_command("python tune_gnn.py"):
            print("Failed to tune hyperparameters")
            return

    if not os.path.exists("toxicity_gnn_model.pth"):
        print("🧠 Step 3: Training GNN model...")
        if not run_command("python train_gnn.py"):
            print("Failed to train model")
            return

    print("✅ Step 4: Validating model...")
    if not run_command("python validate_model.py"):
        print("Model validation failed, but continuing...")

    print("🌐 Step 5: Starting backend server...")
    # Start backend in background
    backend_process = subprocess.Popen([
        sys.executable, "backend/app.py"
    ], cwd=os.getcwd())

    # Wait for backend to start
    time.sleep(3)

    print("🖥️  Step 6: Opening frontend...")
    frontend_path = os.path.join(os.getcwd(), "TOXLENS FRONTEND", "index.html")
    webbrowser.open(f"file://{frontend_path}")

    print("\n🎉 ToxLens AI is now running!")
    print("📱 Frontend: Open TOXLENS FRONTEND/index.html in your browser")
    print("🔗 Backend API: http://127.0.0.1:5000")
    print("🛑 Press Ctrl+C to stop the server")

    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        backend_process.terminate()
        backend_process.wait()

if __name__ == "__main__":
>>>>>>> 925dc03d (Add missing files)
    main()