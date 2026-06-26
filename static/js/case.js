// case.js — standalone follow-up/recovery screen logic for /case/<case_id>.
// Deliberately tiny and dependency-free (no build step, no app.js coupling)
// since a farmer may open this from a saved link days after the original
// chat session, on a different device/tab.
(function () {
  const OUTCOME_ICONS = { better: '🙂', same: '😐', worse: '☹️', not_sure: '🤔' };

  function caseIdFromUrl() {
    const parts = window.location.pathname.split('/').filter(Boolean);
    return parts[parts.length - 1] || '';
  }

  function pick(dict, lang) {
    if (!dict) return '';
    return dict[lang] || dict.en || '';
  }

  async function init() {
    const root = document.getElementById('caseRoot');
    const caseId = caseIdFromUrl();

    let data;
    try {
      const res = await fetch(`/api/case/${encodeURIComponent(caseId)}`);
      data = await res.json();
    } catch (err) {
      root.innerHTML = `<div class="not-found">Connection error. Please try again.</div>`;
      return;
    }

    const text = data.text || {};

    if (!data.found) {
      root.innerHTML = `<div class="not-found">${pick(text.not_found, 'en')}</div>`;
      return;
    }

    const lang = data.case.lang || 'en';
    renderForm(root, caseId, data.case, text, lang);
  }

  function renderForm(root, caseId, caseRecord, text, lang) {
    root.innerHTML = '';

    const title = document.createElement('div');
    title.className = 'case-title';
    title.textContent = pick(text.title, lang);
    root.appendChild(title);

    if (caseRecord.display_name) {
      const sub = document.createElement('div');
      sub.className = 'case-sub';
      sub.textContent = caseRecord.display_name;
      root.appendChild(sub);
    }

    let selectedOutcome = null;

    const grid = document.createElement('div');
    grid.className = 'outcome-grid';
    Object.keys(OUTCOME_ICONS).forEach((key) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'outcome-btn';
      btn.innerHTML = `<span class="icon">${OUTCOME_ICONS[key]}</span><span>${pick(text.options && text.options[key], lang)}</span>`;
      btn.onclick = () => {
        selectedOutcome = key;
        grid.querySelectorAll('.outcome-btn').forEach((b) => b.classList.remove('selected'));
        btn.classList.add('selected');
        submitBtn.disabled = false;
      };
      grid.appendChild(btn);
    });
    root.appendChild(grid);

    const uploadRow = document.createElement('div');
    uploadRow.className = 'upload-row';
    const uploadLabel = document.createElement('label');
    uploadLabel.className = 'upload-label';
    uploadLabel.textContent = '📷 ' + pick(text.upload_prompt, lang);
    const photoInput = document.createElement('input');
    photoInput.type = 'file';
    photoInput.accept = 'image/*';
    photoInput.id = 'photoInput';
    uploadLabel.appendChild(photoInput);
    uploadRow.appendChild(uploadLabel);
    root.appendChild(uploadRow);

    const submitBtn = document.createElement('button');
    submitBtn.type = 'button';
    submitBtn.className = 'submit-btn';
    submitBtn.disabled = true;
    submitBtn.textContent = pick(text.submit, lang);
    submitBtn.onclick = () => submitOutcome(caseId, selectedOutcome, photoInput.files[0], root, text, lang);
    root.appendChild(submitBtn);
  }

  async function submitOutcome(caseId, outcome, file, root, text, lang) {
    const form = new FormData();
    form.append('outcome', outcome);
    if (file) form.append('image', file);

    let result;
    try {
      const res = await fetch(`/api/case/${encodeURIComponent(caseId)}/outcome`, { method: 'POST', body: form });
      result = await res.json();
    } catch (err) {
      result = { recorded: false };
    }

    root.innerHTML = '';
    const thanks = document.createElement('div');
    thanks.className = 'thanks-note';
    thanks.textContent = pick(text.thanks, lang);
    root.appendChild(thanks);

    if (result && result.escalate && result.escalation_note) {
      const note = document.createElement('div');
      note.className = 'escalation-note';
      note.textContent = result.escalation_note;
      root.appendChild(note);
    }
  }

  init();
})();
