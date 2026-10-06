import { pipeline, env } from '@huggingface/transformers';

env.localModelPath = '/models/';
env.allowRemoteModels = false;
env.allowLocalModels = true;
const originalFetch = globalThis.fetch.bind(globalThis);

globalThis.fetch = async (...args) => {
  const response = await originalFetch(...args);

  if ((response.headers.get('content-type') || '').includes('text/html')) {
    console.error('MODEL REQUEST RETURNED HTML:', args[0], response.status);
  }

  return response;
};

let imageDetectorPromise;

function getImageDetector() {
  if (!imageDetectorPromise) {
    imageDetectorPromise = pipeline(
      'image-classification',
      'onnx-community/ai-image-detection-ONNX',
      { device: 'webgpu', dtype: 'q4' }
    );
  }

  return imageDetectorPromise;
}

const API = '/api';
const $ = (selector) => document.querySelector(selector);
let active = 'text';

function setTab(tab) {
  active = tab;

  document.querySelectorAll('.tab, .nav').forEach((element) => {
    element.classList.toggle('active', element.dataset.tab === tab);
  });

  document.querySelectorAll('.pane').forEach((element) => {
    element.classList.toggle('active', element.id === `pane-${tab}`);
  });

  $('#error').hidden = true;
}

document.querySelectorAll('.tab, .nav').forEach((element) => {
  element.addEventListener('click', () => setTab(element.dataset.tab));
});

$('#message').addEventListener('input', (event) => {
  $('#count').textContent =
    `${event.target.value.length.toLocaleString()} / 20,000`;
});

$('#file').addEventListener('change', (event) => {
  $('#file-label').textContent =
    event.target.files[0]?.name || 'Drop a file here or browse';
});

$('#analyze').addEventListener('click', run);

async function run() {
  const button = $('#analyze');
  const error = $('#error');
  error.hidden = true;

  try {
    button.disabled = true;
    button.textContent = 'Analyzing…';

    if (active === 'text') {
      const text = $('#message').value.trim();
      if (!text) throw new Error('Paste a message or transcript first.');

      const response = await fetch(`${API}/analyze/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Text analysis failed.');
      render(data);
      return;
    }

    if (active === 'url') {
      const url = $('#url').value.trim();
      if (!url) throw new Error('Enter a URL to inspect.');

      const response = await fetch(`${API}/analyze/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'URL analysis failed.');
      render(data);
      return;
    }

    const file = $('#file').files[0];
    if (!file) throw new Error('Choose an image file first.');
    if (!file.type.startsWith('image/')) {
      throw new Error('This model currently analyzes images only.');
    }

    const imageUrl = URL.createObjectURL(file);

    try {
      const detector = await getImageDetector();
      const predictions = await detector(imageUrl);
      renderImageResult(file, predictions);
    } finally {
      URL.revokeObjectURL(imageUrl);
    }
  } catch (err) {
    error.textContent = err.message || 'Analysis failed.';
    console.error(err);
    error.hidden = false;
  } finally {
    button.disabled = false;
    button.innerHTML = '<span>Analyze risk</span><span>?</span>';
  }
}

function renderImageResult(file, predictions) {
  const aiPrediction = predictions.find((item) =>
    /fake|ai|synthetic|generated|artificial/i.test(item.label)
  );

  const realPrediction = predictions.find((item) =>
    /real|authentic|human|natural/i.test(item.label)
  );

  const formatScore = (item) =>
    item ? `${Math.round(item.score * 100)}%` : 'Not available';

  const aiScore = formatScore(aiPrediction);
  const realScore = formatScore(realPrediction);

  const indicators = predictions.map((item) => ({
    title: item.label,
    severity: item.score >= 0.75 ? 'High' : item.score >= 0.45 ? 'Medium' : 'Low',
    explanation:
      'This is the model’s relative score for this class. It is not a verified probability.',
    evidence: `${Math.round(item.score * 100)}% model score`,
  }));

  render({
    mode: 'media',
    risk_level: 'Needs Verification',
    score: aiPrediction ? Math.round(aiPrediction.score * 100) : 0,
    filename: file.name,
    size_bytes: file.size,
    media_analysis:
      `AI/synthetic class score: ${aiScore}. ` +
      `Real/authentic class score: ${realScore}. ` +
      'These model scores can be wrong and do not prove who created the image.',
    indicators,
    progression: [],
    guidance: [
      'Check the image source and its original publication.',
      'Do not treat a model score as proof that an image is real or AI-generated.',
      'This model analyzes images only; it does not analyze video or audio.',
    ],
    disclaimer:
      'TRUSTLENS uses a local image model. Its scores are not calibrated probabilities, can be wrong, and may be less accurate for newer AI generators or edited images.',
  });
}

function escapeHtml(value = '') {
  return String(value).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[character]));
}

function render(data) {
  const indicators = data.indicators || [];
  const progression = data.progression || [];
  const guidance = data.guidance || [];
  const level = (data.risk_level || 'Needs Verification')
    .toLowerCase()
    .replaceAll(' ', '-');

  let html = `
    <div class="report-head">
      <div>
        <h3>Explainable analysis report</h3>
        <div class="score-line">
          ${indicators.length} observed model signal${indicators.length === 1 ? '' : 's'}
        </div>
      </div>
      <span class="badge ${escapeHtml(level)}">
        ${escapeHtml(data.risk_level || 'Needs Verification')}
      </span>
    </div>
  `;

  if (data.mode === 'media') {
    html += `
      <div class="section-title">IMAGE INSPECTION</div>
      <div class="indicator">
        <div class="indicator-top">
          <strong>${escapeHtml(data.filename || 'Image')}</strong>
          <span class="severity low">
            ${((data.size_bytes || 0) / 1024 / 1024).toFixed(2)} MB
          </span>
        </div>
        <p>${escapeHtml(data.media_analysis || '')}</p>
      </div>
    `;
  }

  html += '<div class="section-title">OBSERVED INDICATORS</div>';

  if (indicators.length) {
    html += indicators.map((item) => `
      <div class="indicator">
        <div class="indicator-top">
          <strong>${escapeHtml(item.title)}</strong>
          <span class="severity ${escapeHtml((item.severity || 'Low').toLowerCase())}">
            ${escapeHtml(item.severity || 'Low')}
          </span>
        </div>
        <p>${escapeHtml(item.explanation)}</p>
        <div class="evidence">Model output: ${escapeHtml(item.evidence)}</div>
      </div>
    `).join('');
  } else {
    html += '<div class="no-signals">No indicators were returned.</div>';
  }

  if (progression.length) {
    html += '<div class="section-title">SOCIAL-ENGINEERING PROGRESSION</div>';
    html += '<div class="progress">';
    html += progression.map((step) => `
      <div class="stage ${step.observed ? 'done' : ''}">
        ${escapeHtml(step.step)}
      </div>
    `).join('');
    html += '</div>';
  }

  html += '<div class="section-title">INDEPENDENT VERIFICATION</div>';
  html += guidance.map((tip) =>
    `<div class="tip">${escapeHtml(tip)}</div>`
  ).join('');

  html += `<div class="disclaimer">${escapeHtml(data.disclaimer || '')}</div>`;
  $('#results').innerHTML = html;
}