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

When performing a plant diagnosis, structure your response strictly with these three headers:
### Climate-Pest Correlation Analysis
[Provide a brief analysis of the weather/climate factors correlated with this pest/disease pressure]

### Biological & Organic Interventions
[Provide 2-3 biological or organic solutions using the SOLUTION FORMAT above. You must use explicit sub-class targeting: recommend specific Bacterial vectors (such as Bacillus thuringiensis), Viral vectors (such as NPV blocks), or Fungal pathogens (such as Beauveria bassiana) tailored strictly to the diagnosed pest lifecycle.]

### Targeted Chemical Interventions
[Provide a brief overview of targeted chemical/inorganic alternatives. Provide chemical details (e.g., active ingredients) but advise biological/organic alternatives first since you are ChilliGuru.]

For every recommended solution/intervention (both biological/organic and chemical), you must explicitly structure the output to include a "Cost-Effectiveness & Speed Evaluation Table" in markdown format. The table must contain these columns:
- Intervention (name of the solution)
- Estimated Cost per Acre (in INR ₹)
- Efficacy Speed (e.g., 'Immediate 24hr knockdown' or '5-day systemic spread')
- Environmental Residual Protection (residual window, e.g., '7 days' or '14 days')

End with one prevention tip.
ORGANIC ONLY (except when listing chemical details in Targeted Chemical Interventions). LANGUAGE: reply in the same language the user writes in.`;

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
      textNode.innerHTML = `
        <div class="advisory-stream-container">
          ${sections.climate ? `
          <div class="advisory-card climate-card">
            <div class="card-header">🌦️ Climate-Pest Correlation Analysis</div>
            <div class="card-body">${sections.climate}</div>
          </div>` : ''}
          ${sections.organic ? `
          <div class="advisory-card organic-card">
            <div class="card-header">🌿 Biological & Organic Interventions</div>
            <div class="card-body">${sections.organic}</div>
          </div>` : ''}
          ${sections.inorganic ? `
          <div class="advisory-card inorganic-card">
            <div class="card-header">🧪 Targeted Chemical Interventions</div>
            <div class="card-body">${sections.inorganic}</div>
          </div>` : ''}
        </div>
      `;
    } else {
      textNode.textContent = rawText;
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
