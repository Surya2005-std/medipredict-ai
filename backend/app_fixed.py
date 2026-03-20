"""
app_fixed.py - MediPredict AI (Complete Fixed Version)
Run with: python app_fixed.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template_string
import joblib, numpy as np, pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

# Load model once at startup
print("Loading model...")
model    = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
encoder  = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
features = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
print(f"Model loaded! Features: {len(features)}")

# Load supporting data
prec_df = pd.read_csv(os.path.join(DATA_DIR, 'disease_precautions.csv'))
sev_df  = pd.read_csv(os.path.join(DATA_DIR, 'symptom_severity.csv'))
sev_map = dict(zip(sev_df['Symptom'], sev_df['Severity']))

with open(os.path.join(MODELS_DIR, 'model_metrics.json')) as f:
    metrics = json.load(f)

def get_precautions(disease):
    row = prec_df[prec_df['Disease'] == disease]
    if row.empty: return []
    cols = ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
    return [str(row.iloc[0][c]).strip() for c in cols 
            if c in row.columns and str(row.iloc[0][c]).strip() not in ('','nan')]

app = Flask(__name__)

@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MediPredict AI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#0a0f1e;color:#e2e8f0;min-height:100vh}
.navbar{background:rgba(10,15,30,0.95);border-bottom:1px solid rgba(255,255,255,0.08);padding:1rem 2rem;display:flex;align-items:center;gap:.75rem;position:sticky;top:0;z-index:100;backdrop-filter:blur(10px)}
.nav-icon{font-size:1.5rem}
.nav-title{font-size:1.2rem;font-weight:700;background:linear-gradient(135deg,#00c9a7,#4fc3f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero{text-align:center;padding:3rem 1rem 2rem;background:radial-gradient(ellipse 80% 50% at 50% 0%,rgba(0,201,167,0.08),transparent)}
.hero h1{font-size:2.5rem;font-weight:700;margin-bottom:.75rem;line-height:1.2}
.hero h1 span{background:linear-gradient(135deg,#00c9a7,#4fc3f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:#94a3b8;font-size:1rem;max-width:500px;margin:0 auto 1.5rem}
.stats{display:flex;justify-content:center;gap:2rem}
.stat-n{font-size:1.8rem;font-weight:700;color:#00c9a7;display:block}
.stat-l{font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em}
.main{max-width:1100px;margin:0 auto;padding:1.5rem;display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
.panel{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:1.5rem}
.panel h2{font-size:1.1rem;font-weight:600;margin-bottom:.25rem}
.panel p{font-size:.8rem;color:#64748b;margin-bottom:1.25rem}
.search-wrap{position:relative;margin-bottom:1rem}
.search-wrap input{width:100%;padding:.7rem 1rem .7rem 2.5rem;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:10px;color:#e2e8f0;font-family:inherit;font-size:.875rem;outline:none;transition:border .2s}
.search-wrap input:focus{border-color:rgba(0,201,167,0.5)}
.search-wrap input::placeholder{color:#475569}
.search-icon{position:absolute;left:.75rem;top:50%;transform:translateY(-50%);color:#475569;font-size:.9rem}
.dropdown{position:absolute;top:calc(100% + 4px);left:0;right:0;background:#1e293b;border:1px solid rgba(255,255,255,0.1);border-radius:10px;max-height:200px;overflow-y:auto;z-index:50;display:none;box-shadow:0 8px 32px rgba(0,0,0,.5)}
.dropdown.open{display:block}
.dd-item{padding:.55rem 1rem;font-size:.82rem;cursor:pointer;color:#94a3b8;transition:background .15s}
.dd-item:hover{background:rgba(0,201,167,0.1);color:#00c9a7}
.chips-label{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#475569;margin-bottom:.5rem;display:flex;align-items:center;gap:.5rem}
.chip-count{background:#00c9a7;color:#0a0f1e;border-radius:20px;padding:.1rem .45rem;font-size:.68rem;font-weight:700}
.chips{min-height:72px;display:flex;flex-wrap:wrap;gap:.4rem;padding:.6rem;background:rgba(0,0,0,.2);border:1px solid rgba(255,255,255,0.06);border-radius:10px;margin-bottom:1rem;align-content:flex-start}
.chip{display:flex;align-items:center;gap:.3rem;padding:.25rem .6rem;background:rgba(0,201,167,0.12);border:1px solid rgba(0,201,167,0.25);border-radius:20px;font-size:.75rem;color:#00c9a7;animation:popIn .2s ease}
.chip-rm{background:none;border:none;color:rgba(0,201,167,.6);cursor:pointer;font-size:.9rem;padding:0;line-height:1}
.chip-rm:hover{color:#ff6b6b}
.chips-empty{color:#334155;font-size:.8rem;width:100%;text-align:center;align-self:center}
.sev-wrap{margin-bottom:1rem;display:none}
.sev-hdr{display:flex;justify-content:space-between;font-size:.75rem;color:#475569;margin-bottom:.4rem}
.sev-num{font-weight:600;color:#e2e8f0}
.sev-track{height:5px;background:rgba(255,255,255,.07);border-radius:10px;overflow:hidden}
.sev-fill{height:100%;background:linear-gradient(90deg,#00c9a7,#ffb830,#ff6b6b);transition:width .5s cubic-bezier(.34,1.56,.64,1)}
.btn-row{display:flex;gap:.6rem;margin-bottom:1rem}
.btn-clear{padding:.6rem 1rem;background:none;border:1px solid rgba(255,255,255,.1);border-radius:10px;color:#64748b;font-family:inherit;font-size:.82rem;cursor:pointer;transition:all .2s}
.btn-clear:hover{border-color:rgba(255,107,107,.4);color:#ff6b6b}
.btn-analyse{flex:1;padding:.7rem;background:linear-gradient(135deg,#00c9a7,#00a88a);border:none;border-radius:10px;color:#0a0f1e;font-family:inherit;font-size:.875rem;font-weight:700;cursor:pointer;transition:all .2s;display:flex;align-items:center;justify-content:center;gap:.5rem}
.btn-analyse:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 6px 20px rgba(0,201,167,.3)}
.btn-analyse:disabled{opacity:.4;cursor:not-allowed;transform:none}
.presets{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center}
.preset-lbl{font-size:.72rem;color:#475569}
.preset{padding:.2rem .65rem;background:none;border:1px solid rgba(255,255,255,.08);border-radius:20px;color:#475569;font-family:inherit;font-size:.72rem;cursor:pointer;transition:all .2s}
.preset:hover{border-color:rgba(0,201,167,.4);color:#00c9a7}
.placeholder{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.75rem;min-height:280px;color:#334155;text-align:center}
.ph-icon{font-size:3rem}
.ph-title{font-size:.95rem;font-weight:600;color:#475569}
.ph-sub{font-size:.8rem}
.res-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.25rem}
.res-hdr h2{font-size:1.1rem;font-weight:600}
.model-tag{font-size:.68rem;padding:.2rem .65rem;background:rgba(0,201,167,.1);border:1px solid rgba(0,201,167,.2);border-radius:20px;color:#00c9a7}
.rcard{background:rgba(0,0,0,.2);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:1.1rem;margin-bottom:.75rem;animation:fadeUp .35s ease}
.rcard.r1{border-color:rgba(0,201,167,.3);background:rgba(0,201,167,.05)}
.rcard.r2{border-color:rgba(79,195,247,.2)}
.rcard.r3{border-color:rgba(167,139,250,.2)}
.rc-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.6rem}
.rc-rank{font-size:.68rem;color:#475569;margin-bottom:.2rem}
.rc-name{font-size:1rem;font-weight:600}
.r1 .rc-name{color:#00c9a7}
.r2 .rc-name{color:#4fc3f7}
.r3 .rc-name{color:#a78bfa}
.rc-prob{font-size:1.4rem;font-weight:700}
.r1 .rc-prob{color:#00c9a7}
.r2 .rc-prob{color:#4fc3f7}
.r3 .rc-prob{color:#a78bfa}
.rc-bar{height:3px;background:rgba(255,255,255,.07);border-radius:10px;overflow:hidden;margin-bottom:.6rem}
.rc-fill{height:100%;border-radius:10px;transition:width .8s cubic-bezier(.34,1.56,.64,1)}
.r1 .rc-fill{background:#00c9a7}
.r2 .rc-fill{background:#4fc3f7}
.r3 .rc-fill{background:#a78bfa}
.precs{display:flex;flex-wrap:wrap;gap:.3rem}
.prec{font-size:.7rem;padding:.15rem .55rem;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:20px;color:#64748b}
.matched-sec{margin-bottom:.75rem}
.matched-lbl{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#334155;margin-bottom:.4rem}
.m-chip{display:inline-block;padding:.15rem .5rem;background:rgba(79,195,247,.1);border:1px solid rgba(79,195,247,.2);border-radius:20px;font-size:.7rem;color:#4fc3f7;margin:.15rem}
.disclaimer{font-size:.72rem;color:rgba(255,184,48,.8);background:rgba(255,184,48,.06);border:1px solid rgba(255,184,48,.15);border-radius:8px;padding:.65rem .85rem;line-height:1.5}
.spinner{width:14px;height:14px;border:2px solid rgba(10,15,30,.3);border-top-color:#0a0f1e;border-radius:50%;animation:spin .6s linear infinite}
.error-box{display:flex;flex-direction:column;align-items:center;gap:.75rem;min-height:200px;justify-content:center;text-align:center;color:#ff6b6b}
footer{text-align:center;padding:2rem;font-size:.75rem;color:#1e293b;border-top:1px solid rgba(255,255,255,.04);margin-top:2rem}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes popIn{from{opacity:0;transform:scale(.8)}to{opacity:1;transform:scale(1)}}
@keyframes spin{to{transform:rotate(360deg)}}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:10px}
@media(max-width:768px){.main{grid-template-columns:1fr}.hero h1{font-size:1.8rem}}
</style>
</head>
<body>
<nav class="navbar">
  <span class="nav-icon">⚕</span>
  <span class="nav-title">MediPredict AI</span>
</nav>

<section class="hero">
  <h1>Describe your symptoms,<br><span>know your condition</span></h1>
  <p>AI-powered disease prediction using machine learning trained on 41 diseases and 132 symptoms</p>
  <div class="stats">
    <div><span class="stat-n" id="sAcc">99.4%</span><span class="stat-l">Accuracy</span></div>
    <div><span class="stat-n">41</span><span class="stat-l">Diseases</span></div>
    <div><span class="stat-n">132</span><span class="stat-l">Symptoms</span></div>
  </div>
</section>

<div class="main">
  <!-- INPUT PANEL -->
  <div class="panel">
    <h2>Select Your Symptoms</h2>
    <p>Type to search, click to add. Add all symptoms you have.</p>

    <div class="search-wrap">
      <span class="search-icon">🔍</span>
      <input type="text" id="sym-search" placeholder="Type symptom e.g. fever, headache, cough..." autocomplete="off">
      <div class="dropdown" id="dropdown"></div>
    </div>

    <div class="chips-label">
      Selected <span class="chip-count" id="chip-count">0</span>
    </div>
    <div class="chips" id="chips">
      <span class="chips-empty" id="chips-empty">No symptoms selected yet</span>
    </div>

    <div class="sev-wrap" id="sev-wrap">
      <div class="sev-hdr">
        <span>Severity Score</span>
        <span class="sev-num" id="sev-val">0%</span>
      </div>
      <div class="sev-track"><div class="sev-fill" id="sev-fill" style="width:0%"></div></div>
    </div>

    <div class="btn-row">
      <button class="btn-clear" onclick="clearAll()">Clear All</button>
      <button class="btn-analyse" id="btn-go" onclick="runPredict()" disabled>
        <span id="btn-txt">Analyse Symptoms</span>
        <span class="spinner" id="btn-spin" style="display:none"></span>
      </button>
    </div>

    <div class="presets">
      <span class="preset-lbl">Quick test:</span>
      <button class="preset" onclick="setPreset(['high_fever','chills','sweating','headache','nausea','muscle_pain'])">Malaria</button>
      <button class="preset" onclick="setPreset(['fatigue','weight_loss','polyuria','excessive_hunger','blurred_and_distorted_vision'])">Diabetes</button>
      <button class="preset" onclick="setPreset(['cough','breathlessness','high_fever','phlegm','chest_pain'])">Respiratory</button>
      <button class="preset" onclick="setPreset(['itching','skin_rash','nodal_skin_eruptions'])">Skin</button>
      <button class="preset" onclick="setPreset(['vomiting','headache','nausea','dizziness'])">Vertigo</button>
    </div>
  </div>

  <!-- RESULTS PANEL -->
  <div class="panel" id="result-panel">
    <div class="placeholder" id="placeholder">
      <span class="ph-icon">🩺</span>
      <div class="ph-title">Your analysis appears here</div>
      <div class="ph-sub">Select symptoms and click Analyse</div>
    </div>
    <div id="results" style="display:none"></div>
    <div class="error-box" id="err-box" style="display:none">
      <span style="font-size:2rem">⚠️</span>
      <span id="err-msg"></span>
    </div>
  </div>
</div>

<footer>MediPredict AI &middot; Built with Python, Flask &amp; scikit-learn &middot; For educational purposes only</footer>

<script>
let allSymptoms = [];
let selected = new Set();

// Load symptoms on page load
fetch('/api/symptoms')
  .then(r => r.json())
  .then(d => {
    allSymptoms = d.symptoms || [];
    console.log('Loaded symptoms:', allSymptoms.length);
  })
  .catch(e => console.error('Failed to load symptoms:', e));

// Load accuracy
fetch('/api/metrics')
  .then(r => r.json())
  .then(d => {
    const best = d.best_model;
    if (best && d[best]) {
      document.getElementById('sAcc').textContent = (d[best].accuracy * 100).toFixed(1) + '%';
    }
  }).catch(() => {});

const searchEl  = document.getElementById('sym-search');
const ddEl      = document.getElementById('dropdown');
const chipsEl   = document.getElementById('chips');
const emptyEl   = document.getElementById('chips-empty');
const countEl   = document.getElementById('chip-count');
const btnGo     = document.getElementById('btn-go');
const sevWrap   = document.getElementById('sev-wrap');
const sevFill   = document.getElementById('sev-fill');
const sevVal    = document.getElementById('sev-val');

// Search input handler
searchEl.addEventListener('input', function() {
  const q = this.value.trim().toLowerCase();
  if (!q) { ddEl.classList.remove('open'); return; }

  const matches = allSymptoms
    .filter(s => {
      const label = (s.label || s.value || s).toLowerCase();
      const val   = (s.value || s).toLowerCase();
      return (label.includes(q) || val.includes(q.replace(/ /g,'_'))) && !selected.has(s.value || s);
    })
    .slice(0, 12);

  if (!matches.length) { ddEl.classList.remove('open'); return; }

  ddEl.innerHTML = matches.map(s => {
    const val   = s.value || s;
    const label = s.label || val.replace(/_/g,' ').replace(/\\b\\w/g, c => c.toUpperCase());
    return `<div class="dd-item" onmousedown="addSym('${val}','${label}')">${label}</div>`;
  }).join('');
  ddEl.classList.add('open');
});

searchEl.addEventListener('blur', () => setTimeout(() => ddEl.classList.remove('open'), 200));
searchEl.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    const first = ddEl.querySelector('.dd-item');
    if (first) first.dispatchEvent(new MouseEvent('mousedown'));
  }
});

function addSym(val, label) {
  selected.add(val);
  searchEl.value = '';
  ddEl.classList.remove('open');
  renderChips();
  updateUI();
}

function removeSym(val) {
  selected.delete(val);
  renderChips();
  updateUI();
}

function renderChips() {
  chipsEl.querySelectorAll('.chip').forEach(e => e.remove());
  countEl.textContent = selected.size;
  if (!selected.size) { emptyEl.style.display = ''; return; }
  emptyEl.style.display = 'none';
  selected.forEach(val => {
    const label = val.replace(/_/g,' ').replace(/\\b\\w/g, c => c.toUpperCase());
    const chip  = document.createElement('div');
    chip.className = 'chip';
    chip.innerHTML = `${label}<button class="chip-rm" onclick="removeSym('${val}')">×</button>`;
    chipsEl.appendChild(chip);
  });
}

function updateUI() {
  btnGo.disabled = selected.size === 0;
  if (!selected.size) { sevWrap.style.display = 'none'; return; }
  sevWrap.style.display = '';
  const pct = Math.min(selected.size * 10, 100);
  sevFill.style.width = pct + '%';
  sevVal.textContent  = pct + '%';
}

function clearAll() {
  selected.clear();
  renderChips();
  updateUI();
  document.getElementById('placeholder').style.display = '';
  document.getElementById('results').style.display = 'none';
  document.getElementById('err-box').style.display = 'none';
}

function setPreset(syms) {
  selected = new Set(syms);
  renderChips();
  updateUI();
  runPredict();
}

async function runPredict() {
  if (!selected.size) return;

  document.getElementById('btn-txt').style.display  = 'none';
  document.getElementById('btn-spin').style.display = '';
  btnGo.disabled = true;

  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('results').style.display     = 'none';
  document.getElementById('err-box').style.display     = 'none';

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({symptoms: [...selected], top_k: 3})
    });
    const data = await res.json();

    if (data.error && !data.predictions.length) {
      document.getElementById('err-msg').textContent = data.error;
      document.getElementById('err-box').style.display = '';
    } else {
      renderResults(data);
    }
  } catch(e) {
    document.getElementById('err-msg').textContent = 'Connection error. Is the server running?';
    document.getElementById('err-box').style.display = '';
  } finally {
    document.getElementById('btn-txt').style.display  = '';
    document.getElementById('btn-spin').style.display = 'none';
    btnGo.disabled = selected.size === 0;
  }
}

function renderResults(data) {
  // Update severity with real score
  if (data.severity_score) {
    sevFill.style.width = data.severity_score + '%';
    sevVal.textContent  = data.severity_score + '%';
  }

  const mi    = data.model_info || {};
  const rDiv  = document.getElementById('results');
  const ranks = ['r1','r2','r3'];
  const icons = ['🥇','🥈','🥉'];

  let html = `
    <div class="res-hdr">
      <h2>Analysis Results</h2>
      <span class="model-tag">${mi.name || 'ML Model'} · ${mi.accuracy || 99.4}%</span>
    </div>`;

  (data.predictions || []).forEach((p, i) => {
    const precs = (p.precautions || []).map(pr => `<span class="prec">✓ ${pr}</span>`).join('');
    html += `
      <div class="rcard ${ranks[i]}">
        <div class="rc-top">
          <div>
            <div class="rc-rank">${icons[i]} #${i+1} Most Likely</div>
            <div class="rc-name">${p.disease}</div>
          </div>
          <div class="rc-prob">${p.probability.toFixed(1)}%</div>
        </div>
        <div class="rc-bar"><div class="rc-fill" style="width:${p.probability}%"></div></div>
        ${precs ? `<div class="precs">${precs}</div>` : ''}
      </div>`;
  });

  if (data.matched_symptoms && data.matched_symptoms.length) {
    const mchips = data.matched_symptoms.map(s =>
      `<span class="m-chip">${s.replace(/_/g,' ')}</span>`).join('');
    html += `<div class="matched-sec"><div class="matched-lbl">Matched Symptoms</div>${mchips}</div>`;
  }

  html += `<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This is for informational purposes only. Always consult a qualified doctor for proper medical advice.</div>`;

  rDiv.innerHTML = html;
  rDiv.style.display = '';
}
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/api/symptoms')
def symptoms():
    syms = [{'value': f, 'label': f.replace('_',' ').title()} for f in features]
    return jsonify({'symptoms': syms, 'count': len(syms)})

@app.route('/api/predict', methods=['POST'])
def predict_route():
    try:
        data     = request.get_json(force=True)
        syms_in  = data.get('symptoms', [])
        top_k    = int(data.get('top_k', 3))

        if not syms_in:
            return jsonify({'error': 'No symptoms provided.', 'predictions': []}), 400

        clean = [s.strip().lower().replace(' ','_') for s in syms_in]

        vec     = np.zeros(len(features), dtype=int)
        matched = []
        unknown = []
        for s in clean:
            if s in features:
                vec[features.index(s)] = 1
                matched.append(s)
            else:
                unknown.append(s)

        if not matched:
            return jsonify({
                'error': 'None of the symptoms were recognised. Please select from the dropdown.',
                'predictions': [], 'matched_symptoms': [], 'unknown_symptoms': unknown,
                'severity_score': 0, 'model_info': {}
            }), 422

        proba       = model.predict_proba(vec.reshape(1,-1))[0]
        top_indices = np.argsort(proba)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            disease = encoder.inverse_transform([idx])[0]
            conf    = round(float(proba[idx]) * 100, 2)
            predictions.append({'disease': disease, 'probability': conf, 'precautions': get_precautions(disease)})

        sev_vals = [sev_map.get(s, 3) for s in matched]
        sev_score = round(min(np.mean(sev_vals) / 7 * 100, 100), 1) if sev_vals else 0

        best_name = metrics.get('best_model', 'Random Forest')
        model_acc = metrics.get(best_name, {}).get('accuracy', 0.994)

        return jsonify({
            'predictions':      predictions,
            'matched_symptoms': matched,
            'unknown_symptoms': unknown,
            'severity_score':   sev_score,
            'model_info':       {'name': best_name, 'accuracy': round(model_acc * 100, 2)}
        })

    except Exception as e:
        return jsonify({'error': str(e), 'predictions': []}), 500

@app.route('/api/metrics')
def metrics_route():
    return jsonify(metrics)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n🚀 MediPredict AI running at http://localhost:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False)
