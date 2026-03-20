/**
 * dashboard.js – MediPredict AI  (Evaluation Dashboard)
 * ──────────────────────────────────────────────────────
 * Loads /api/metrics and renders KPI cards + table.
 * Charts are served as static images from /static/models/*.png
 */

'use strict';

async function loadDashboard() {
  try {
    const res  = await fetch('/api/metrics');
    const data = await res.json();

    const bestName = data.best_model;
    const best     = data[bestName] || {};

    // ── KPI Cards ──────────────────────────────────────────
    const kpiRow = document.getElementById('kpi-row');
    kpiRow.innerHTML = `
      <div class="kpi-card best" style="animation-delay:0s">
        <div class="kpi-label">Best Accuracy</div>
        <div class="kpi-value">${(best.accuracy * 100).toFixed(1)}%</div>
        <div class="kpi-model">${bestName}</div>
      </div>
      <div class="kpi-card" style="animation-delay:0.1s">
        <div class="kpi-label">F1-Score</div>
        <div class="kpi-value">${(best.f1 * 100).toFixed(1)}%</div>
        <div class="kpi-model">Weighted avg</div>
      </div>
      <div class="kpi-card" style="animation-delay:0.2s">
        <div class="kpi-label">Precision</div>
        <div class="kpi-value">${(best.precision * 100).toFixed(1)}%</div>
        <div class="kpi-model">Weighted avg</div>
      </div>
      <div class="kpi-card" style="animation-delay:0.3s">
        <div class="kpi-label">CV Mean ± Std</div>
        <div class="kpi-value">${(best.cv_mean * 100).toFixed(1)}%</div>
        <div class="kpi-model">±${(best.cv_std * 100).toFixed(1)}%</div>
      </div>
    `;

    // ── Model Table ────────────────────────────────────────
    const tbody   = document.getElementById('metrics-tbody');
    const models  = Object.keys(data).filter(k => k !== 'best_model');
    const metrics = ['accuracy','precision','recall','f1'];

    tbody.innerHTML = models.map(name => {
      const m   = data[name];
      const isBest = name === bestName;

      const cells = metrics.map(metric => {
        const val  = (m[metric] * 100).toFixed(2) + '%';
        const cls  = m[metric] >= 0.95 ? 'high' : m[metric] >= 0.85 ? 'medium' : 'low';
        return `<td class="score-cell ${cls}">${val}</td>`;
      }).join('');

      const cvCell = `<td>${(m.cv_mean*100).toFixed(2)}% ± ${(m.cv_std*100).toFixed(2)}%</td>`;
      const badge  = isBest ? `<td><span class="best-badge">★ Best</span></td>` : `<td>—</td>`;

      return `<tr><td><strong>${name}</strong></td>${cells}${cvCell}${badge}</tr>`;
    }).join('');

  } catch (e) {
    document.getElementById('metrics-tbody').innerHTML =
      `<tr><td colspan="7" style="text-align:center;color:#ff6b6b;padding:2rem">
        Failed to load metrics. Make sure the server is running and models are trained.
       </td></tr>`;
    document.getElementById('kpi-row').innerHTML =
      ['Best Accuracy','F1-Score','Precision','CV Mean'].map(l =>
        `<div class="kpi-card"><div class="kpi-label">${l}</div><div class="kpi-value" style="color:#ff6b6b">Error</div></div>`
      ).join('');
  }
}

// Handle missing chart images gracefully
document.querySelectorAll('.chart-img').forEach(img => {
  img.addEventListener('error', () => {
    img.style.display = 'none';
    const note = document.createElement('p');
    note.style.cssText = 'color:var(--text-muted);font-size:0.85rem;padding:1rem;text-align:center';
    note.textContent = 'Chart not available — run the training script first.';
    img.parentNode.appendChild(note);
  });
});

loadDashboard();
