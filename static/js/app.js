  // ── State ──────────────────────────────────────────────────────────────────
  const API_BASE = '';
  let conversation  = [];
  let selectedImage = null;
  let isLoading     = false;
  let userLat       = null;
  let userLng       = null;

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

  const SYSTEM_PROMPT = `You are ChilliGuru, a friendly and helpful farming assistant for chilli farmers in Andhra Pradesh and Telangana. You talk like a trusted friend who knows a lot about farming — simple, warm, and easy to understand. No complicated words.

VARIETIES: Teja, Guntur Sannam, LCA 334, Wonder Hot, Pusa Jwala, Byadgi, and local varieties across Guntur, Khammam, Warangal, Krishna, Prakasam, Kurnool.

SEASONS: Kharif (Jun-Oct) rainy season; Rabi (Nov-Feb) cool season; Zaid (Mar-May) hot season.

PESTS: Thrips, Spider Mites, Aphids, Whiteflies, Fruit Borer, Mealybugs, Leaf Miners, Armyworm, Cutworm, Broad Mite.

DISEASES: Leaf Curl Virus, Powdery Mildew, Anthracnose, Damping Off, Phytophthora, Cercospora Leaf Spot, Bacterial Wilt, Mosaic Virus.

SOLUTION FORMAT — always use this:
Solution name (Home-made OR Shop):
How to make/use it: [simple steps]
How well it works: X out of 10
Days to see results: X-X days
Cost: Rs X to Rs X
How often: every X days for X weeks
Where to get: [AP/Telangana]

Always give 2-3 solutions. End with one prevention tip.
ORGANIC ONLY — never suggest chemicals.
LANGUAGE: reply in the same language the user writes in.`;

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

    const warnBadge = isLow
      ? `<div class="warning-badge-inline">⚠️ Low Confidence Warning</div>`
      : '';

    const teluguHtml = telugu
      ? `<div class="det-telugu">${telugu}</div>`
      : '';

    return `<div class="detection-card${isLow ? ' low-conf' : ''}">` +
      warnBadge +
      `<div class="det-pest-name">🔍 ${english}</div>` +
      teluguHtml +
      `<div class="det-meta">` +
        `<span class="det-confidence${lowCls}">${conf}%</span>` +
        `<span class="det-type">${kind}</span>` +
      `</div>` +
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
    txt.style.whiteSpace = 'pre-wrap';
    txt.textContent = content;
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
          try { errMsg = (await res.json()).error || errMsg; } catch {}
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
  }

  // ── Language ───────────────────────────────────────────────────────────────
  function setLang(lang, btn) {
    currentLang = lang;
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

    const climateRegex = /(?:Climate-Pest Correlation Analysis|Climate-Pest Correlation)[:\s*#-]*([\s\S]*?)(?=(?:Targeted Organic Regulation|Targeted Organic|Targeted Inorganic|$))/i;
    const organicRegex = /(?:Targeted Organic Regulation|Targeted Organic)[:\s*#-]*([\s\S]*?)(?=(?:Targeted Inorganic Regulation|Targeted Inorganic|$))/i;
    const inorganicRegex = /(?:Targeted Inorganic Regulation|Targeted Inorganic)[:\s*#-]*([\s\S]*?)$/i;

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
      textNode.innerHTML = `
        <div class="advisory-stream-container">
          ${sections.climate ? `
          <div class="advisory-card climate-card">
            <div class="card-header">🌦️ Climate-Pest Correlation Analysis</div>
            <div class="card-body">${sections.climate}</div>
          </div>` : ''}
          ${sections.organic ? `
          <div class="advisory-card organic-card">
            <div class="card-header">🌿 Targeted Organic Regulation</div>
            <div class="card-body">${sections.organic}</div>
          </div>` : ''}
          ${sections.inorganic ? `
          <div class="advisory-card inorganic-card">
            <div class="card-header">🧪 Targeted Inorganic Regulation</div>
            <div class="card-body">${sections.inorganic}</div>
          </div>` : ''}
        </div>
      `;
    } else {
      textNode.textContent = rawText;
    }
  }
