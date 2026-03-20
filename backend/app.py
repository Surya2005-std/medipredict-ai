"""
app.py  –  Disease Symptom Predictor  (Flask Backend)
────────────────────────────────────────────────────────────────
Endpoints:
  GET  /                 → serve index.html
  GET  /api/symptoms     → list of all valid symptom names
  POST /api/predict      → predict diseases from symptoms
  GET  /api/metrics      → model training metrics
  GET  /api/health       → health check
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template, send_from_directory


from utils.predictor import predict, get_all_symptoms, get_metrics

# ── App setup ──────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, 'templates'),
    static_folder  =os.path.join(FRONTEND_DIR, 'static'),
)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response



# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'service': 'Disease Symptom Predictor'})


@app.route('/api/symptoms', methods=['GET'])
def symptoms_list():
    """Return all valid symptom names (for autocomplete / dropdown)."""
    try:
        symptoms = get_all_symptoms()
        # Human-readable labels: replace underscores with spaces, title-case
        formatted = [
            {'value': s, 'label': s.replace('_', ' ').title()}
            for s in symptoms
        ]
        return jsonify({'symptoms': formatted, 'count': len(formatted)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict_route():
    """
    Body (JSON):
      { "symptoms": ["fever", "headache", "nausea"], "top_k": 3 }
    """
    try:
        data     = request.get_json(force=True)
        symptoms = data.get('symptoms', [])
        top_k    = int(data.get('top_k', 3))

        if not isinstance(symptoms, list) or not symptoms:
            return jsonify({'error': 'Provide a non-empty list of symptoms.'}), 400

        result = predict(symptoms, top_k=min(top_k, 5))

        if 'error' in result and not result.get('predictions'):
            return jsonify(result), 422

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


@app.route('/api/metrics', methods=['GET'])
def metrics_route():
    """Return model evaluation metrics."""
    try:
        return jsonify(get_metrics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/static/models/<path:filename>')
def model_images(filename):
    """Serve generated chart images from frontend/static/models."""
    charts_dir = os.path.join(FRONTEND_DIR, 'static', 'models')
    return send_from_directory(charts_dir, filename)


# ── Dev server ─────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Disease Predictor API running at http://localhost:{port}")
    print(f"   Dashboard: http://localhost:{port}/dashboard\n")
    app.run(host='0.0.0.0', port=port, debug=True)
