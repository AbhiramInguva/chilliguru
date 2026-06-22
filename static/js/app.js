  // ── State ──────────────────────────────────────────────────────────────────
  const API_BASE = '';

  // ── Security: escape any dynamic string before it touches innerHTML ─────────
  // Detection labels and streamed model text are untrusted (the model echoes
  // the farmer's own message back), so anything interpolated into an HTML
  // template must be escaped to prevent stored/reflected XSS.
  function escapeHtml(value) {
    if (value == null) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  let conversation  = [];
  let selectedImage = null;
  let isLoading     = false;
  let userLat       = null;
  let userLng       = null;
  let lastImageFile = null;
  let lastResponseTextNode = null;

  function isNonChilliResponse(status, errorText, label) {
    const txt = (errorText || '').toLowerCase() + ' ' + (label || '').toLowerCase();
    return status === 422 || 
           txt.includes('chilli plant') || 
           txt.includes('chili plant') || 
           txt.includes('cannot identify crop') || 
           txt.includes('non_chilli') ||
           label === 'non_chilli';
  }

  function detectUserLanguage() {
    const userLang = navigator.language || navigator.userLanguage || '';
    const match = userLang.match(/^([a-z]{2})/i);
    if (match) {
      const code = match[1].toLowerCase();
      if (['en', 'te', 'hi', 'kn', 'ta'].includes(code)) {
        return code;
      }
    }
    return 'en'; // default fallback
  }

  let currentLang   = detectUserLanguage();

  const LANG_PROMPTS = {
    en: '',
    te: 'Please respond in Telugu (తెలుగు).',
    hi: 'Please respond in Hindi (हिंदी).',
    kn: 'Please respond in Kannada (ಕನ್ನಡ).',
    ta: 'Please respond in Tamil (தமிழ்).',
  };

  document.addEventListener('DOMContentLoaded', () => {
    const activeLang = currentLang;
    const btn = document.querySelector(`.lang-capsule[onclick*="'${activeLang}'"]`) ||
                document.querySelector(`.lang-btn[onclick*="'${activeLang}'"]`);
    if (btn) {
      btn.click();
    }
  });

  function captureGeolocation() {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          userLat = position.coords.latitude;
          userLng = position.coords.longitude;
          console.log(`Captured high-accuracy location: ${userLat}, ${userLng}`);
        },
        (error) => {
          console.warn("Geolocation access failed or denied:", error);
        },
        {
          enableHighAccuracy: true,
          timeout: 5000,
          maximumAge: 0
        }
      );
    } else {
      console.warn("Geolocation not supported by this browser.");
    }
  }

  // NOTE: Chat advice is generated server-side (app.py SYSTEM_PROMPT), grounded
  // in a curated agronomy reference (knowledge/chilli_kb.json). No prompt or
  // KB content is embedded client-side.

  // ── Health check ───────────────────────────────────────────────────────────
  async function checkHealth() {
    try {
      const res  = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      const pill = document.getElementById('modelStatusPill');
      if (data.model_ready) {
        pill.textContent = '🟢 AI Vision On';
        pill.classList.add('online');
      } else {
        pill.textContent = '🟡 Text Mode';
      }
    } catch {
      document.getElementById('modelStatusPill').textContent = '🔴 Offline';
    }
  }
  checkHealth();

  // ── Flask backend calls ────────────────────────────────────────────────────
  async function callGroq(userMessage) {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userMessage, history: conversation })
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || 'API error');
    }
    const data  = await response.json();
    const reply = data.reply;
    conversation.push({ role: 'user', content: userMessage });
    conversation.push({ role: 'assistant', content: reply });
    return reply;
  }

  // ── Build leaf curl / leaf-age morphology strip ─────────────────────────────
  function buildCurlMorphologyHtml(det) {
    const m = det && det.morphology;
    if (!m || (m.curl_direction === 'unknown' && m.leaf_age === 'unknown')) return '';

    const curlIcon = { upward: '⬆️', downward: '⬇️', flat: '➖' };
    const curlText = {
      upward: 'Upward cupping', downward: 'Downward inverted-boat', flat: 'Flat (no curl)',
    };
    const ageIcon = { young: '🌱', adult: '🍃', old: '🍂' };

    const rows = [];
    if (m.curl_direction && m.curl_direction !== 'unknown') {
      rows.push(
        `<div class="morph-row">` +
          `<span class="morph-key">${curlIcon[m.curl_direction] || '🍃'} Leaf curl</span>` +
          `<span class="morph-val">${escapeHtml(curlText[m.curl_direction] || m.curl_direction)}` +
          (m.curl_confidence ? ` (${escapeHtml(m.curl_confidence)}%)` : '') + `</span>` +
        `</div>` +
        (m.curl_telugu ? `<div class="morph-telugu">${escapeHtml(m.curl_telugu)}</div>` : '')
      );
    }
    if (m.leaf_age && m.leaf_age !== 'unknown') {
      rows.push(
        `<div class="morph-row">` +
          `<span class="morph-key">${ageIcon[m.leaf_age] || '🍃'} Leaf age</span>` +
          `<span class="morph-val">${escapeHtml(m.leaf_age)}` +
          (m.age_confidence ? ` (${escapeHtml(m.age_confidence)}%)` : '') + `</span>` +
        `</div>`
      );
    }
    if (m.juvenile_mimicry) {
      rows.push(
        `<div class="morph-mimicry">⚠️ Mature leaf showing juvenile traits — possible viral stunting</div>`
      );
    }
    return `<div class="morph-strip">${rows.join('')}</div>`;
  }

  // ── Build detection card ───────────────────────────────────────────────────
  function buildDetectionCard(det, isLow) {
    const raw    = det.label || 'Unknown';
    const tMatch = raw.match(/\[(.+?)\]$/);
    const english = tMatch ? raw.replace(/\s*\[.+?\]$/, '').trim() : raw;
    const telugu  = det.telugu || (tMatch ? tMatch[1] : '');

    const conf    = det.confidence != null ? det.confidence : '?';
    const kind    = (det.type || 'pest').toUpperCase();
    const confNum = typeof conf === 'number' ? conf : (parseInt(conf) || 0);
    const lowCls  = isLow ? ' low' : '';

    // Escape every interpolated value — these come from the model/detector.
    const engEsc    = escapeHtml(english);
    const teluguEsc = escapeHtml(telugu);
    const confEsc   = escapeHtml(conf);
    const kindEsc   = escapeHtml(kind);
    const curlHtml  = buildCurlMorphologyHtml(det);

    const warnBadge = isLow
      ? `<div class="warning-badge-inline">⚠️ Low Confidence Warning</div>`
      : '';

    const teluguHtml = telugu
      ? `<div class="det-telugu">${teluguEsc}</div>`
      : '';

    return `<div class="detection-card${isLow ? ' low-conf' : ''}">` +
      warnBadge +
      `<div class="det-pest-name">🔍 ${engEsc}</div>` +
      teluguHtml +
      `<div class="det-meta">` +
        `<span class="det-confidence${lowCls}">${confEsc}%</span>` +
        `<span class="det-type">${kindEsc}</span>` +
      `</div>` +
      curlHtml +
      `<div class="conf-bar-wrap">` +
        `<div class="conf-bar-label">Detection confidence</div>` +
        `<div class="conf-track">` +
          `<div class="conf-fill${lowCls}" style="width:${confNum}%"></div>` +
        `</div>` +
      `</div>` +
    `</div>`;
  }

  // ── UI helpers ─────────────────────────────────────────────────────────────
  function hideWelcome() {
    const w = document.getElementById('welcomeState');
    if (w) w.style.display = 'none';
  }

  function addMessage(role, content, imageUrl = null, detectionCard = null) {
    hideWelcome();
    const container = document.getElementById('chatMessages');
    const wrap      = document.createElement('div');
    wrap.className  = `message ${role}`;

    const av        = document.createElement('div');
    av.className    = role === 'bot' ? 'avatar bot' : 'avatar user-av';
    av.textContent  = role === 'bot' ? '🌶️' : 'You';

    const bubble    = document.createElement('div');
    bubble.className = 'bubble';

    if (role === 'bot') {
      const lbl = document.createElement('div');
      lbl.className   = 'bubble-name';
      lbl.textContent = 'ChilliGuru';
      bubble.appendChild(lbl);
    }

    if (imageUrl) {
      const img     = document.createElement('img');
      img.src       = imageUrl;
      img.alt       = 'Uploaded plant photo';
      bubble.appendChild(img);
    }

    if (detectionCard) {
      const cardEl = document.createElement('div');
      cardEl.innerHTML = detectionCard;
      bubble.appendChild(cardEl);
    }

    const txt = document.createElement('div');
    if (role === 'bot') {
      txt.innerHTML = content.startsWith('<div class="alert-card-warning"') ? content : marked.parse(content);
      lastResponseTextNode = txt;
    } else {
      txt.style.whiteSpace = 'pre-wrap';
      txt.textContent = content;
    }
    bubble.appendChild(txt);

    wrap.appendChild(av);
    wrap.appendChild(bubble);
    container.appendChild(wrap);
    container.scrollTop = container.scrollHeight;
    return wrap;
  }

  function addTyping() {
    hideWelcome();
    const container = document.getElementById('chatMessages');
    const wrap      = document.createElement('div');
    wrap.className  = 'message bot';
    wrap.id         = 'typingMsg';

    const av        = document.createElement('div');
    av.className    = 'avatar bot';
    av.textContent  = '🌶️';

    const typ       = document.createElement('div');
    typ.className   = 'typing';
    typ.innerHTML   = '<span></span><span></span><span></span>';

    wrap.appendChild(av);
    wrap.appendChild(typ);
    container.appendChild(wrap);
    container.scrollTop = container.scrollHeight;
  }

  function removeTyping() {
    const t = document.getElementById('typingMsg');
    if (t) t.remove();
  }

  // Creates an empty bot bubble and returns the live text <div> so the
  // SSE reader can append chunks to it word-by-word.
  function addStreamingBubble(detectionCardHtml) {
    hideWelcome();
    const container = document.getElementById('chatMessages');

    const wrap      = document.createElement('div');
    wrap.className  = 'message bot';

    const av        = document.createElement('div');
    av.className    = 'avatar bot';
    av.textContent  = '🌶️';

    const bubble    = document.createElement('div');
    bubble.className = 'bubble';

    const lbl       = document.createElement('div');
    lbl.className   = 'bubble-name';
    lbl.textContent = 'ChilliGuru';
    bubble.appendChild(lbl);

    if (detectionCardHtml) {
      const cardEl = document.createElement('div');
      cardEl.innerHTML = detectionCardHtml;
      bubble.appendChild(cardEl);
    }

    // This is the node we write chunks into — returned to the caller
    const txt = document.createElement('div');
    txt.style.whiteSpace = 'pre-wrap';
    txt.className = 'streaming-cursor';
    bubble.appendChild(txt);

    wrap.appendChild(av);
    wrap.appendChild(bubble);
    container.appendChild(wrap);
    container.scrollTop = container.scrollHeight;

    return txt;
  }

  // ── Send message ───────────────────────────────────────────────────────────
  async function sendMessage() {
    const input    = document.getElementById('userInput');
    const text     = input.value.trim();
    const langMsg  = LANG_PROMPTS[currentLang];

    if (!text && !selectedImage) return;
    if (isLoading) return;

    isLoading = true;
    document.getElementById('sendBtn').disabled = true;

    const userText    = text || 'Please analyse this photo of my chilli plant.';
    const displayText = text || '📷 Photo uploaded for analysis';
    const imageUrl    = selectedImage ? URL.createObjectURL(selectedImage) : null;
    const imageToSend = selectedImage;

    addMessage('user', displayText, imageUrl);
    input.value = '';
    autoResize(input);
    removeImage();
    addTyping();

    try {
      if (imageToSend) {
        // ── Image path: SSE streaming /detect ─────────────────────────────────
        const form = new FormData();
        form.append('image',   imageToSend);
        form.append('message', userText);
        form.append('history', JSON.stringify(conversation));
        form.append('lang', currentLang);
        // Farmer's MCQ field answers — folded into detection ranking server-side
        const fieldAnswers = collectFieldAnswers();
        for (const [k, v] of Object.entries(fieldAnswers)) form.append(k, v);
        if (userLat !== null) {
          form.append('latitude', parseFloat(userLat));
        }
        if (userLng !== null) {
          form.append('longitude', parseFloat(userLng));
        }

        const res = await fetch(`${API_BASE}/detect`, { method: 'POST', body: form });

        // Non-2xx responses are always small JSON (validation / rate-limit errors)
        if (!res.ok) {
          let errMsg = 'Detection error';
          let checkNonChilli = false;
          try {
            const errData = await res.json();
            errMsg = errData.error || errMsg;
            if (res.status === 422 || isNonChilliResponse(res.status, errMsg, errData.label)) {
              checkNonChilli = true;
            }
          } catch {}
          
          if (checkNonChilli) {
            removeTyping();
            addMessage('bot', '<div class="alert-card-warning" style="background:#fef2f2; border:1px solid #f87171; color:#991b1b; padding:16px; border-radius:12px; font-weight:500; margin:8px 0; font-family:\'DM Sans\',sans-serif; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">⚠️ Clear Capture Required: Please upload a close-up photo of a chilli leaf or fruit for analysis.</div>');
            isLoading = false;
            document.getElementById('sendBtn').disabled = false;
            selectedImage = null;
            return;
          }
          throw new Error(errMsg);
        }

        const contentType = res.headers.get('content-type') || '';

        if (contentType.includes('text/event-stream')) {
          // ── SSE streaming path ─────────────────────────────────────────────
          const reader  = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer    = '';   // partial SSE data accumulator
          let fullText  = '';   // complete assembled response text
          let textNode  = null; // live DOM node we write chunks into
          let metaSeen  = false;

          outer: while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // SSE events are separated by double newlines
            const parts = buffer.split('\n\n');
            buffer = parts.pop(); // keep any incomplete trailing fragment

            for (const part of parts) {
              const line = part.trim();
              if (!line.startsWith('data: ')) continue;
              let event;
              try { event = JSON.parse(line.slice(6)); } catch { continue; }

              if (event.type === 'meta') {
                // First frame: detection card data → build bubble shell now
                const cardHtml = event.detection
                  ? buildDetectionCard(event.detection, event.low_confidence)
                  : null;
                removeTyping();
                textNode = addStreamingBubble(cardHtml);
                conversation.push({ role: 'user', content: userText });
                metaSeen = true;

              } else if (event.type === 'text' && textNode) {
                fullText += event.text;
                updateStreamingDisplay(textNode, fullText);
                const msgs = document.getElementById('chatMessages');
                msgs.scrollTop = msgs.scrollHeight;

              } else if (event.type === 'done') {
                if (textNode) textNode.classList.remove('streaming-cursor');
                conversation.push({ role: 'assistant', content: fullText });
                break outer;

              } else if (event.type === 'error') {
                // Mid-stream Groq error: update existing bubble or show new one
                const errMsg = event.error || 'Streaming failed. Please try again.';
                if (textNode) {
                  textNode.classList.remove('streaming-cursor');
                  textNode.textContent = errMsg;
                } else {
                  removeTyping();
                  addMessage('bot', errMsg);
                }
                break outer;
              }
            }
          }

          // Stream closed before we ever received a meta frame (network drop)
          if (!metaSeen) {
            removeTyping();
            addMessage('bot', 'Connection interrupted. Please try again.');
          }

        } else {
          // ── JSON path: guardrail rejections returned with HTTP 200 ──────────
          // (success:false / phase:3 — backend emits plain JSON, not a stream)
          const data = await res.json();
          removeTyping();
          let reply;
          let detectionCard = null;

          if (data.success === false) {
            if (isNonChilliResponse(200, data.error, data.label || (data.top_detection || {}).label)) {
              addMessage('bot', '<div class="alert-card-warning" style="background:#fef2f2; border:1px solid #f87171; color:#991b1b; padding:16px; border-radius:12px; font-weight:500; margin:8px 0; font-family:\'DM Sans\',sans-serif; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">⚠️ Clear Capture Required: Please upload a close-up photo of a chilli leaf or fruit for analysis.</div>');
              isLoading = false;
              document.getElementById('sendBtn').disabled = false;
              selectedImage = null;
              return;
            }
            reply = data.error || 'Unable to process this image. Please try again.';
            if (data.phase === 3) {
              detectionCard = `<div class="warning-badge-inline">⚠️ Low Confidence Warning</div>`;
            }
          } else {
            // Defensive fallback for any unexpected old-style JSON response
            reply = data.reply || '';
            detectionCard = data.detection
              ? buildDetectionCard(data.detection, data.low_confidence)
              : null;
          }
          conversation.push({ role: 'user',      content: userText });
          conversation.push({ role: 'assistant', content: reply });
          addMessage('bot', reply, null, detectionCard);
        }

      } else {
        // ── Text-only path: /chat endpoint (unchanged) ─────────────────────────
        let messageToSend = userText;
        if (langMsg) messageToSend = `${langMsg} ${userText}`;
        const reply = await callGroq(messageToSend);
        removeTyping();
        addMessage('bot', reply);
      }

    } catch (err) {
      removeTyping();
      addMessage('bot', `Sorry, something went wrong: ${err.message}`);
    }

    isLoading = false;
    document.getElementById('sendBtn').disabled = false;
    selectedImage = null;
  }

  // ── Quick ask ──────────────────────────────────────────────────────────────
  function askQuestion(q) {
    document.getElementById('userInput').value = q;
    sendMessage();
  }

  // ── Image handling ─────────────────────────────────────────────────────────
  function triggerImageUpload() {
    document.getElementById('fileInput').click();
  }

  // ── Field MCQ answers (sent with the image to refine detection) ─────────────
  const FIELD_MCQ_IDS = {
    plant_age:     'q_plant_age',
    observed:      'q_observed',
    affected_part: 'q_affected',
    curl_dir:      'q_curl',
  };

  function collectFieldAnswers() {
    const out = {};
    for (const [key, id] of Object.entries(FIELD_MCQ_IDS)) {
      const el = document.getElementById(id);
      if (el && el.value) out[key] = el.value;
    }
    return out;
  }

  function resetFieldAnswers() {
    for (const id of Object.values(FIELD_MCQ_IDS)) {
      const el = document.getElementById(id);
      if (el) el.value = '';
    }
  }

  function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    selectedImage = file;

    const preview    = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    document.getElementById('previewName').textContent = file.name;
    previewImg.src = URL.createObjectURL(file);
    preview.classList.add('show');
    document.getElementById('userInput').placeholder = 'Describe what you see, or press send to auto-analyse…';
    
    // Capture high-accuracy Geolocation
    captureGeolocation();
  }

  function removeImage() {
    selectedImage = null;
    document.getElementById('imagePreview').classList.remove('show');
    document.getElementById('fileInput').value = '';
    document.getElementById('userInput').placeholder = 'Ask about your chilli crop…';
    resetFieldAnswers();
  }

  // ── Language ───────────────────────────────────────────────────────────────
  function setLang(lang, btn) {
    currentLang = lang;
    document.documentElement.lang = lang;
    document.querySelectorAll('.lang-capsule').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
  }

  // ── Keyboard & auto-resize ─────────────────────────────────────────────────
  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  // ── Camera ─────────────────────────────────────────────────────────────────
  let cameraStream = null;

  async function openCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Camera not supported on this device. Use the gallery button to upload a photo.');
      document.getElementById('fileInput').click();
      return;
    }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      document.getElementById('cameraVideo').srcObject = cameraStream;
      document.getElementById('cameraModal').classList.add('show');
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        alert('Camera access denied. Allow camera permissions in your browser, or use the gallery button to upload a photo.');
      } else {
        alert('Could not open camera. Use the gallery button to upload a photo instead.');
        document.getElementById('fileInput').click();
      }
    }
  }

  function capturePhoto() {
    const video  = document.getElementById('cameraVideo');
    const canvas = document.createElement('canvas');
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    closeCamera();
    canvas.toBlob(blob => {
      const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
      selectedImage = file;
      const preview     = document.getElementById('imagePreview');
      const previewImg  = document.getElementById('previewImg');
      previewImg.src    = URL.createObjectURL(file);
      document.getElementById('previewName').textContent = 'camera-capture.jpg';
      preview.classList.add('show');
      document.getElementById('userInput').placeholder = 'Describe what you see, or press send to auto-analyse…';
      
      // Capture high-accuracy Geolocation
      captureGeolocation();
    }, 'image/jpeg', 0.85);
  }

  function closeCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
    }
    document.getElementById('cameraVideo').srcObject = null;
    document.getElementById('cameraModal').classList.remove('show');
  }

  // ── Stream Advisory Parsers ────────────────────────────────────────────────
  function parseAdvisorySections(text) {
    let climate = '';
    let organic = '';
    let inorganic = '';

    const climateRegex = /(?:Climate-Pest Correlation Analysis|Climate-Pest Correlation)[:\s*#-]*([\s\S]*?)(?=(?:Targeted Organic Regulation|Targeted Organic|Biological & Organic Interventions|Targeted Inorganic|Targeted Chemical Interventions|$))/i;
    const organicRegex = /(?:Targeted Organic Regulation|Targeted Organic|Biological & Organic Interventions)[:\s*#-]*([\s\S]*?)(?=(?:Targeted Inorganic Regulation|Targeted Inorganic|Targeted Chemical Interventions|$))/i;
    const inorganicRegex = /(?:Targeted Inorganic Regulation|Targeted Inorganic|Targeted Chemical Interventions)[:\s*#-]*([\s\S]*?)$/i;

    const mClimate = text.match(climateRegex);
    const mOrganic = text.match(organicRegex);
    const mInorganic = text.match(inorganicRegex);

    if (mClimate) climate = mClimate[1];
    if (mOrganic) organic = mOrganic[1];
    if (mInorganic) inorganic = mInorganic[1];

    const clean = (str) => {
      if (!str) return '';
      let cleaned = str.replace(/^[:\s*#-]+/, '').trim();
      cleaned = cleaned.replace(/[:\s*#-]+$/, '').trim();
      return cleaned;
    };

    return {
      climate: clean(climate),
      organic: clean(organic),
      inorganic: clean(inorganic)
    };
  }

  function updateStreamingDisplay(textNode, rawText) {
    const sections = parseAdvisorySections(rawText);

    if (sections.climate || sections.organic || sections.inorganic) {
      // Section bodies are raw model output — escape before injecting. The
      // model is instructed to echo the farmer's message, so this is an
      // untrusted XSS sink.
      textNode.innerHTML = `
        <div class="advisory-stream-container">
          ${sections.climate ? `
          <div class="advisory-card climate-card">
            <div class="card-header">🌦️ Climate-Pest Correlation Analysis</div>
            <div class="card-body">${marked.parse(escapeHtml(sections.climate))}</div>
          </div>` : ''}
          ${sections.organic ? `
          <div class="advisory-card organic-card">
            <div class="card-header">🌿 Biological & Organic Interventions</div>
            <div class="card-body">${marked.parse(escapeHtml(sections.organic))}</div>
          </div>` : ''}
          ${sections.inorganic ? `
          <div class="advisory-card inorganic-card">
            <div class="card-header">🧪 Targeted Chemical Interventions</div>
            <div class="card-body">${marked.parse(escapeHtml(sections.inorganic))}</div>
          </div>` : ''}
        </div>
      `;
    } else {
      textNode.innerHTML = marked.parse(escapeHtml(rawText));
    }
  }

  let mapInitialized = false;
  let leafletMap = null;

  window.switchTab = function(tabName) {
    const chatBtn = document.querySelector(".tab-btn[onclick*=\"'chat'\"]");
    const mapBtn = document.querySelector(".tab-btn[onclick*=\"'map'\"]");
    const chatPanel = document.getElementById("chat-tab-panel");
    const mapPanel = document.getElementById("map-tab-panel");

    if (tabName === 'chat') {
      if (chatBtn) chatBtn.classList.add('active');
      if (mapBtn) mapBtn.classList.remove('active');
      if (chatPanel) {
        chatPanel.style.display = 'flex';
        chatPanel.classList.add('active');
      }
      if (mapPanel) {
        mapPanel.style.display = 'none';
        mapPanel.classList.remove('active');
      }
    } else if (tabName === 'map') {
      if (chatBtn) chatBtn.classList.remove('active');
      if (mapBtn) mapBtn.classList.add('active');
      if (chatPanel) {
        chatPanel.style.display = 'none';
        chatPanel.classList.remove('active');
      }
      if (mapPanel) {
        mapPanel.style.display = 'flex';
        mapPanel.classList.add('active');
      }
      // Initialize map on switch
      setTimeout(() => {
        initRegionalMap();
      }, 50);
    }
  };

  async function initRegionalMap() {
    if (mapInitialized && leafletMap) {
      leafletMap.invalidateSize();
      return;
    }
    
    const statusText = document.getElementById("map-location-status");
    if (statusText) statusText.textContent = "Detecting current coordinates...";

    // Default centroid coordinates if geolocation is not shared
    let lat = 16.5;
    let lon = 79.5;

    // Use navigator.geolocation high accuracy
    if (navigator.geolocation) {
      try {
        const position = await new Promise((resolve, reject) => {
          navigator.geolocation.getCurrentPosition(resolve, reject, {
            enableHighAccuracy: true,
            timeout: 5000,
            maximumAge: 0
          });
        });
        lat = position.coords.latitude;
        lon = position.coords.longitude;
        if (statusText) statusText.textContent = `Centered on field: ${lat.toFixed(4)}, ${lon.toFixed(4)}`;
      } catch (err) {
        console.warn("Map Geolocation failed, using AP/Telangana Centroid:", err);
        if (statusText) statusText.textContent = "Using regional baseline center coordinates";
      }
    } else {
      if (statusText) statusText.textContent = "Using regional baseline center coordinates";
    }

    try {
      // Initialize Leaflet map
      leafletMap = L.map('regional-pest-map').setView([lat, lon], 10);
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      }).addTo(leafletMap);

      mapInitialized = true;

      // Fetch regional risk data
      const response = await fetch(`/api/regional-risk?lat=${lat}&lon=${lon}`);
      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }
      const data = await response.json();

      data.forEach(loc => {
        // Determine the highest risk level to color the circle
        let highestLevel = 'Low';
        if (loc.risks && loc.risks.length > 0) {
          const levels = loc.risks.map(r => r.level);
          if (levels.includes('Critical')) {
            highestLevel = 'Critical';
          } else if (levels.includes('High')) {
            highestLevel = 'High';
          } else if (levels.includes('Moderate')) {
            highestLevel = 'Moderate';
          }
        }

        // Color mapping
        let color = '#3498db'; // blue for low/none
        let fillOpacity = 0.2;
        if (highestLevel === 'Critical') {
          color = '#e74c3c'; // red
          fillOpacity = 0.45;
        } else if (highestLevel === 'High') {
          color = '#e67e22'; // orange
          fillOpacity = 0.35;
        } else if (highestLevel === 'Moderate') {
          color = '#f1c40f'; // yellow
          fillOpacity = 0.25;
        }

        // Draw spatial coverage risk circle
        const circle = L.circle([loc.latitude, loc.longitude], {
          color: color,
          fillColor: color,
          fillOpacity: fillOpacity,
          radius: 6000 // 6km radius spatial coverage
        }).addTo(leafletMap);

        // Build popup content
        let risksHtml = '';
        if (loc.risks && loc.risks.length > 0) {
          risksHtml = loc.risks.map(r => {
            let badgeColor = 'background: #34495e; color: white;';
            if (r.level === 'Critical') badgeColor = 'background: #c0392b; color: white; font-weight: bold;';
            else if (r.level === 'High') badgeColor = 'background: #d35400; color: white;';
            else if (r.level === 'Moderate') badgeColor = 'background: #f39c12; color: white;';
            
            // Localized labels
            let label = currentLang === 'te' ? r.telugu || r.label : r.label;
            return `
              <div style="margin-bottom: 8px; border-bottom: 1px solid #f1f1f1; padding-bottom: 6px;">
                <span style="font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; ${badgeColor}">${r.level}</span>
                <strong style="margin-left: 6px; font-size: 0.9rem; color: #2c3e50;">${label}</strong>
                <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #7f8c8d; line-height: 1.3;">${r.description}</p>
              </div>
            `;
          }).join('');
        } else {
          risksHtml = '<p style="font-size: 0.85rem; color: #7f8c8d; margin: 0;">No significant pest risks identified for current climate.</p>';
        }

        const titleText = loc.name;
        const tempText = currentLang === 'te' ? `ఉష్ణోగ్రత` : `Temp`;
        const humText = currentLang === 'te' ? `తేమ` : `Humidity`;
        
        const popupContent = `
          <div style="font-family: 'DM Sans', sans-serif; width: 260px;">
            <h4 style="margin: 0 0 4px 0; color: #27ae60; font-size: 1rem; border-bottom: 2px solid #27ae60; padding-bottom: 4px;">${titleText}</h4>
            <div style="display: flex; justify-content: space-between; margin: 8px 0; font-size: 0.85rem; color: #34495e; font-weight: 500;">
              <span>🌡️ ${tempText}: <strong>${loc.temperature}°C</strong></span>
              <span>💧 ${humText}: <strong>${loc.humidity}%</strong></span>
            </div>
            <div style="max-height: 200px; overflow-y: auto; padding-right: 4px; margin-top: 8px;">
              ${risksHtml}
            </div>
          </div>
        `;
        circle.bindPopup(popupContent);
      });

    } catch (mapErr) {
      console.error("Error loading regional map data:", mapErr);
      if (statusText) statusText.textContent = "Failed to load risk mapping. Please retry.";
    }
  }
