/**
 * app.js – MediPredict AI  (Main Predictor Page)
 * ──────────────────────────────────────────────
 * Handles:
 *  • Symptom search + autocomplete dropdown
 *  • Chip management (add / remove selected symptoms)
 *  • Severity meter
 *  • API call to /api/predict
 *  • Result rendering (disease cards, probabilities, precautions)
 *  • Chatbot interaction (NLP extraction → predict)
 *  • Preset buttons
 */

'use strict';

// ── State ─────────────────────────────────────────────────────
let allSymptoms    = [];   // {value, label}[]
let selectedSyms   = new Set();
let highlightedIdx = -1;
let filteredItems  = [];

// ── DOM refs ──────────────────────────────────────────────────
const searchInput    = document.getElementById('symptom-search');
const dropdown       = document.getElementById('search-dropdown');
const chipsContainer = document.getElementById('chips-container');
const chipsEmpty     = document.getElementById('chips-empty');
const chipCount      = document.getElementById('chip-count');
const btnPredict     = document.getElementById('btn-predict');
const btnClear       = document.getElementById('btn-clear');
const predictText    = document.getElementById('predict-text');
const predictSpinner = document.getElementById('predict-spinner');

const resultsPanel      = document.getElementById('results-panel');
const resultsPlaceholder= document.getElementById('results-placeholder');
const resultsContent    = document.getElementById('results-content');
const resultsError      = document.getElementById('results-error');
const resultCards       = document.getElementById('result-cards');
const matchedChips      = document.getElementById('matched-chips');
const modelBadge        = document.getElementById('model-badge');

const severityBlock = document.getElementById('severity-block');
const severityBar   = document.getElementById('severity-bar');
const severityValue = document.getElementById('severity-value');

const statAcc = document.getElementById('stat-acc');

// Chatbot
const chatbotFab   = document.getElementById('chatbot-fab');
const chatbotPanel = document.getElementById('chatbot-panel');
const chatbotClose = document.getElementById('chatbot-close');
const chatbotMsgs  = document.getElementById('chatbot-messages');
const chatbotInput = document.getElementById('chatbot-input');
const chatbotSend  = document.getElementById('chatbot-send');


// ══════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════
async function init() {
  try {
    const [symRes, metRes] = await Promise.all([
      fetch('/api/symptoms'),
      fetch('/api/metrics'),
    ]);
    const symData = await symRes.json();
    const metData = await metRes.json();

    allSymptoms = symData.symptoms || [];

    // Update hero accuracy stat
    const best = metData.best_model;
    if (best && metData[best]) {
      statAcc.textContent = (metData[best].accuracy * 100).toFixed(1) + '%';
    }
  } catch (e) {
    console.warn('Init fetch failed – running in standalone mode:', e);
  }
}

// ══════════════════════════════════════════════════════════════
// SEARCH & AUTOCOMPLETE
// ══════════════════════════════════════════════════════════════
searchInput.addEventListener('input', () => {
  const q = searchInput.value.trim().toLowerCase();
  highlightedIdx = -1;

  if (!q) { closeDropdown(); return; }

  filteredItems = allSymptoms
    .filter(s => s.value.includes(q.replace(/ /g,'_')) || s.label.toLowerCase().includes(q))
    .filter(s => !selectedSyms.has(s.value))
    .slice(0, 12);

  if (!filteredItems.length) { closeDropdown(); return; }

  dropdown.innerHTML = filteredItems.map((s, i) => {
    const lbl = highlight(s.label, q);
    return `<div class="dropdown-item" data-idx="${i}">${lbl}</div>`;
  }).join('');
  dropdown.classList.add('open');

  dropdown.querySelectorAll('.dropdown-item').forEach(el => {
    el.addEventListener('mousedown', e => {
      e.preventDefault();
      addSymptom(filteredItems[+el.dataset.idx].value);
    });
  });
});

searchInput.addEventListener('keydown', e => {
  if (!dropdown.classList.contains('open')) return;
  const items = dropdown.querySelectorAll('.dropdown-item');

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    highlightedIdx = (highlightedIdx + 1) % items.length;
    refreshHighlight(items);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    highlightedIdx = (highlightedIdx - 1 + items.length) % items.length;
    refreshHighlight(items);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (highlightedIdx >= 0) addSymptom(filteredItems[highlightedIdx].value);
    else if (filteredItems.length) addSymptom(filteredItems[0].value);
  } else if (e.key === 'Escape') {
    closeDropdown();
  }
});

document.addEventListener('click', e => {
  if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) closeDropdown();
});

function refreshHighlight(items) {
  items.forEach((el, i) => el.classList.toggle('highlighted', i === highlightedIdx));
  if (highlightedIdx >= 0) items[highlightedIdx].scrollIntoView({ block: 'nearest' });
}

function closeDropdown() {
  dropdown.classList.remove('open');
  dropdown.innerHTML = '';
}

function highlight(text, query) {
  const re = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')})`, 'gi');
  return text.replace(re, '<mark>$1</mark>');
}


// ══════════════════════════════════════════════════════════════
// CHIP MANAGEMENT
// ══════════════════════════════════════════════════════════════
function addSymptom(value) {
  if (selectedSyms.has(value)) return;
  selectedSyms.add(value);
  searchInput.value = '';
  closeDropdown();
  renderChips();
  updateUI();
}

function removeSymptom(value) {
  selectedSyms.delete(value);
  renderChips();
  updateUI();
}

function renderChips() {
  const chips = [...selectedSyms];
  chipCount.textContent = chips.length;

  if (!chips.length) {
    chipsEmpty.style.display = '';
    // Remove all chip elements
    chipsContainer.querySelectorAll('.chip').forEach(el => el.remove());
    return;
  }
  chipsEmpty.style.display = 'none';

  // Rebuild chips
  chipsContainer.querySelectorAll('.chip').forEach(el => el.remove());
  chips.forEach(val => {
    const label = labelOf(val);
    const chip  = document.createElement('div');
    chip.className = 'chip';
    chip.innerHTML = `${label}<button class="chip-remove" aria-label="Remove ${label}">×</button>`;
    chip.querySelector('.chip-remove').addEventListener('click', () => removeSymptom(val));
    chipsContainer.appendChild(chip);
  });
}

function labelOf(val) {
  const found = allSymptoms.find(s => s.value === val);
  return found ? found.label : val.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
}

function updateUI() {
  const n = selectedSyms.size;
  btnPredict.disabled = n === 0;

  if (n === 0) {
    severityBlock.style.display = 'none';
    return;
  }

  // Show rough severity from symptom count (real value comes from API)
  severityBlock.style.display = '';
  const roughSev = Math.min(n * 12, 100);
  severityBar.style.width  = roughSev + '%';
  severityValue.textContent = roughSev + '%';
}


// ══════════════════════════════════════════════════════════════
// PREDICT
// ══════════════════════════════════════════════════════════════
btnPredict.addEventListener('click', runPrediction);

async function runPrediction(symptoms = null) {
  const symsToSend = symptoms || [...selectedSyms];
  if (!symsToSend.length) return;

  // Loading state
  predictText.style.display    = 'none';
  predictSpinner.style.display = '';
  btnPredict.disabled = true;
  showState('loading');

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ symptoms: symsToSend, top_k: 3 }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || 'Prediction failed.');
    } else {
      renderResults(data);
    }
  } catch (err) {
    showError('Network error – is the server running?');
  } finally {
    predictText.style.display    = '';
    predictSpinner.style.display = 'none';
    btnPredict.disabled = selectedSyms.size === 0;
  }
}

function renderResults(data) {
  // Update severity meter with real score
  const sev = data.severity_score || 0;
  severityBar.style.width  = sev + '%';
  severityValue.textContent = sev + '%';
  severityBlock.style.display = '';

  // Model badge
  const mi = data.model_info || {};
  modelBadge.textContent = `${mi.name || 'ML Model'}  •  ${mi.accuracy || 0}% accuracy`;

  // Disease cards
  resultCards.innerHTML = '';
  (data.predictions || []).forEach((p, i) => {
    const rankClass = ['rank-1','rank-2','rank-3'][i] || 'rank-3';
    const card = document.createElement('div');
    card.className = `result-card ${rankClass}`;
    card.style.animationDelay = `${i * 0.1}s`;

    const precs = (p.precautions || []).map(pr =>
      `<span class="prec-pill">${pr}</span>`
    ).join('');

    card.innerHTML = `
      <div class="result-card-top">
        <div>
          <div class="result-rank">#${i+1} Prediction</div>
          <div class="result-disease">${p.disease}</div>
        </div>
        <div class="result-prob">
          <div class="prob-circle">${p.probability.toFixed(1)}%</div>
        </div>
      </div>
      <div class="result-bar-track">
        <div class="result-bar-fill" style="width:${p.probability}%"></div>
      </div>
      ${precs ? `<div class="result-precautions">${precs}</div>` : ''}
    `;
    resultCards.appendChild(card);
  });

  // Matched symptoms
  matchedChips.innerHTML = (data.matched_symptoms || []).map(s =>
    `<span class="matched-chip">${labelOf(s)}</span>`
  ).join('');

  showState('results');
}

function showState(state) {
  resultsPlaceholder.style.display = state === 'placeholder' ? '' : 'none';
  resultsContent.style.display     = state === 'results'     ? '' : 'none';
  resultsError.style.display       = state === 'error'       ? '' : 'none';
}

function showError(msg) {
  document.getElementById('error-message').textContent = msg;
  showState('error');
}

window.resetError = () => showState('placeholder');


// ══════════════════════════════════════════════════════════════
// CLEAR
// ══════════════════════════════════════════════════════════════
btnClear.addEventListener('click', () => {
  selectedSyms.clear();
  renderChips();
  updateUI();
  showState('placeholder');
});


// ══════════════════════════════════════════════════════════════
// PRESETS
// ══════════════════════════════════════════════════════════════
document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const syms = btn.dataset.symptoms.split(',');
    selectedSyms.clear();
    syms.forEach(s => selectedSyms.add(s.trim()));
    renderChips();
    updateUI();
    runPrediction();
  });
});


// ══════════════════════════════════════════════════════════════
// CHATBOT
// ══════════════════════════════════════════════════════════════
chatbotFab.addEventListener('click', () => chatbotPanel.classList.toggle('open'));
chatbotClose.addEventListener('click', () => chatbotPanel.classList.remove('open'));

chatbotSend.addEventListener('click', handleChatSend);
chatbotInput.addEventListener('keydown', e => { if (e.key === 'Enter') handleChatSend(); });

async function handleChatSend() {
  const msg = chatbotInput.value.trim();
  if (!msg) return;
  chatbotInput.value = '';

  appendChatMsg(msg, 'user');

  // Extract symptoms using keyword matching
  const extracted = extractSymptomsFromText(msg);

  if (!extracted.length) {
    appendChatMsg("I couldn't identify any known symptoms in your message. Try describing symptoms like fever, headache, cough, vomiting, etc.", 'bot');
    return;
  }

  appendChatMsg(`I found these symptoms: ${extracted.map(labelOf).join(', ')}. Running analysis…`, 'bot');

  // Add to selected & run prediction
  extracted.forEach(s => selectedSyms.add(s));
  renderChips();
  updateUI();

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symptoms: extracted, top_k: 3 }),
    });
    const data = await res.json();
    if (data.predictions && data.predictions.length) {
      const top = data.predictions[0];
      appendChatMsg(
        `Based on your symptoms, the most likely condition is <strong>${top.disease}</strong> with ${top.probability.toFixed(1)}% confidence. ` +
        (top.precautions.length ? `Suggested precaution: ${top.precautions[0]}.` : '') +
        ' Please consult a doctor for proper diagnosis.',
        'bot'
      );
      renderResults(data);
    } else {
      appendChatMsg('Unable to identify a specific condition. Please add more symptoms.', 'bot');
    }
  } catch {
    appendChatMsg('Server error. Please try again.', 'bot');
  }
}

function appendChatMsg(text, role) {
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.innerHTML = text;
  chatbotMsgs.appendChild(div);
  chatbotMsgs.scrollTop = chatbotMsgs.scrollHeight;
}

function extractSymptomsFromText(text) {
  const norm = text.toLowerCase().replace(/[^a-z0-9\s]/g,' ');
  const found = [];
  // Try multi-word then single-word matches
  allSymptoms.forEach(({ value }) => {
    const readable = value.replace(/_/g,' ');
    if (norm.includes(readable) || norm.includes(value)) found.push(value);
  });
  return [...new Set(found)].slice(0, 8);
}


// ── Bootstrap ────────────────────────────────────────────────
init();
