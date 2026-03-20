"""
run.py  –  MediPredict AI Entry Point
──────────────────────────────────────
Usage:
    python run.py                  # start server (default port 5000)
    python run.py --port 8080      # custom port
    python run.py --setup          # generate data + train models first
"""

import os, sys, argparse

# Resolve paths
ROOT    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, 'backend')
sys.path.insert(0, BACKEND)

def setup():
    """Generate dataset and train all models."""
    print("\n" + "="*60)
    print("  STEP 1/2 — Generating dataset")
    print("="*60)
    os.chdir(BACKEND)
    sys.path.insert(0, BACKEND)
    from scripts.generate_dataset import save_datasets
    save_datasets(os.path.join(BACKEND, 'data'))

    print("\n" + "="*60)
    print("  STEP 2/2 — Training models  (this may take 2–5 min)")
    print("="*60)
    # Run training inline so paths resolve correctly
    os.system(f'cd {BACKEND} && python scripts/train_models.py')

    # Copy charts to frontend static
    import shutil
    src = os.path.join(BACKEND, 'models')
    dst = os.path.join(ROOT, 'frontend', 'static', 'models')
    os.makedirs(dst, exist_ok=True)
    for f in ['confusion_matrix.png', 'feature_importance.png', 'model_comparison.png']:
        fp = os.path.join(src, f)
        if os.path.exists(fp):
            shutil.copy(fp, dst)
    print("\n✅ Setup complete!")


def serve(port=5000):
    """Start the Flask development server."""
    os.chdir(BACKEND)
    from app import app
    print(f"\n🚀  MediPredict AI  →  http://localhost:{port}")
    print(f"   Dashboard       →  http://localhost:{port}/dashboard")
    print(f"   API docs        →  http://localhost:{port}/api/health\n")
    app.run(host='0.0.0.0', port=port, debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MediPredict AI')
    parser.add_argument('--setup', action='store_true', help='Generate data & train models')
    parser.add_argument('--port',  type=int, default=5000, help='Server port (default 5000)')
    args = parser.parse_args()

    if args.setup:
        setup()

    serve(args.port)
