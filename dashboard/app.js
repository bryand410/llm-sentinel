// ==========================================
// LLM Sentinel — Dashboard JavaScript
// ==========================================

const API_BASE = window.location.origin;

// ==========================================
// Analyze prompt
// ==========================================
async function analyzePrompt() {
    const input = document.getElementById('prompt-input');
    const btn = document.getElementById('analyze-btn');
    const text = input.value.trim();

    if (!text) {
        input.focus();
        return;
    }

    // UI loading state
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Analyse...';
    btn.querySelector('.btn-loader').hidden = false;

    try {
        const res = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        displayResult(data);
        refreshStats();
        refreshHistory();

    } catch (err) {
        alert(`Erreur : ${err.message}\n\nVérifiez que l'API est lancée.`);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').textContent = 'Analyser';
        btn.querySelector('.btn-loader').hidden = true;
    }
}

// ==========================================
// Display result
// ==========================================
function displayResult(data) {
    const section = document.getElementById('result-section');
    const card = document.getElementById('result-card');
    const icon = document.getElementById('result-icon');
    const label = document.getElementById('result-label');

    section.hidden = false;

    // Reset animation
    card.style.animation = 'none';
    card.offsetHeight; // trigger reflow
    card.style.animation = '';

    // Set class
    card.classList.remove('safe-result', 'injection-result');
    card.classList.add(data.label === 'safe' ? 'safe-result' : 'injection-result');

    // Set icon & label
    if (data.label === 'safe') {
        icon.textContent = '✅';
        label.textContent = 'PROMPT SÛR';
    } else {
        icon.textContent = '🚨';
        label.textContent = 'INJECTION DÉTECTÉE';
    }

    // Confidence bar
    const safePct = (data.probabilities.safe * 100).toFixed(1);
    const injPct = (data.probabilities.injection * 100).toFixed(1);
    document.getElementById('safe-fill').style.width = safePct + '%';
    document.getElementById('injection-fill').style.width = injPct + '%';
    document.getElementById('safe-pct').textContent = safePct + '%';
    document.getElementById('injection-pct').textContent = injPct + '%';

    // Details
    document.getElementById('detail-confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
    document.getElementById('detail-latency').textContent = data.latency_ms + ' ms';
    document.getElementById('detail-text').textContent = data.input_text;

    // Mini stats
    document.getElementById('mini-latency').querySelector('.mini-stat-value').textContent = data.latency_ms + 'ms';

    // Smooth scroll
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ==========================================
// Quick test
// ==========================================
function quickTest(text) {
    const input = document.getElementById('prompt-input');
    input.value = text;
    analyzePrompt();
}

// ==========================================
// Refresh stats
// ==========================================
async function refreshStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        const data = await res.json();

        document.getElementById('stat-total').textContent = data.total_analyses;
        document.getElementById('stat-injections').textContent = data.injections_detected;
        document.getElementById('stat-safe').textContent = data.safe_prompts;
        document.getElementById('stat-latency').textContent = data.avg_latency_ms ? data.avg_latency_ms + ' ms' : '—';

        // Mini stats
        document.getElementById('mini-total').querySelector('.mini-stat-value').textContent = data.total_analyses;
        document.getElementById('mini-blocked').querySelector('.mini-stat-value').textContent = data.injections_detected;
    } catch (err) {
        console.error('Stats error:', err);
    }
}

// ==========================================
// Refresh history
// ==========================================
async function refreshHistory() {
    try {
        const res = await fetch(`${API_BASE}/history`);
        const data = await res.json();
        const list = document.getElementById('history-list');

        if (data.history.length === 0) {
            list.innerHTML = '<div class="history-empty">Aucune analyse pour le moment</div>';
            return;
        }

        list.innerHTML = data.history.map((item, i) => `
            <div class="history-item" style="animation-delay: ${i * 0.05}s">
                <span class="history-badge ${item.label}">${item.label === 'safe' ? '✅ Safe' : '🚨 Injection'}</span>
                <span class="history-text" title="${escapeHtml(item.input_text)}">${escapeHtml(item.input_text)}</span>
                <div class="history-meta">
                    <span>${(item.confidence * 100).toFixed(1)}%</span>
                    <span>${item.latency_ms}ms</span>
                </div>
            </div>
        `).join('');

    } catch (err) {
        console.error('History error:', err);
    }
}

// ==========================================
// Utilities
// ==========================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==========================================
// Keyboard shortcut (Ctrl+Enter)
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('prompt-input');
    input.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            analyzePrompt();
        }
    });

    // Check API status
    checkStatus();
    refreshStats();
    refreshHistory();
});

async function checkStatus() {
    const badge = document.getElementById('status-badge');
    try {
        const res = await fetch(`${API_BASE}/`);
        if (res.ok) {
            badge.innerHTML = '<span class="pulse"></span> Connecté';
            badge.style.color = '#10b981';
        }
    } catch {
        badge.innerHTML = '⚠️ Déconnecté';
        badge.style.color = '#ef4444';
    }
}
