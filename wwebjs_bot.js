const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const http = require('http');
const fs = require('fs');

const NATLAS_URL = process.env.N_ATLAS_BASE_URL || 'https://ai.saynet.ng/v1/chat/completions';
const NATLAS_VOICE_URL = process.env.N_ATLAS_VOICE_URL || 'https://ai.saynet.ng/v1/audio/chat';
const NATLAS_API_KEY = process.env.N_ATLAS_API_KEY || 'KgRRl_6vEp7YRCY6agQ9HjH0RFk6T6hr9KYQq7YTRfRrKeWQIyFs_biogXFvK_Lo';
const NATLAS_MODEL = process.env.N_ATLAS_MODEL || 'n-atlas';
const HTTP_PORT = 3000;

console.log('====================================================');
console.log(' Confam AI - WhatsApp Bot (High-Availability Engine)');
console.log(' Text Endpoint:  ', NATLAS_URL);
console.log(' Voice Endpoint: ', NATLAS_VOICE_URL);
console.log(' Model:          ', NATLAS_MODEL);
console.log(' Web Dashboard:  ', `http://localhost:${HTTP_PORT}`);
console.log('====================================================\n');

// State tracking for the web UI
let botState = {
    status: 'initializing',
    qr: null,
    phone: '2349060970339',
    lastPing: Date.now()
};

// In-memory conversation session store (phone -> { messages, lastActive, preferredLang })
const sessions = new Map();
const SESSION_TTL_MS = 30 * 60 * 1000; // 30 minutes

function getSession(phone) {
    const now = Date.now();
    let session = sessions.get(phone);
    if (!session || (now - session.lastActive > SESSION_TTL_MS)) {
        session = { messages: [], lastActive: now, preferredLang: 'en' };
        sessions.set(phone, session);
    }
    session.lastActive = now;
    return session;
}

// Track processed message IDs to prevent double-replies
const processedMsgIds = new Set();
function markProcessed(id) {
    processedMsgIds.add(id);
    if (processedMsgIds.size > 1000) {
        const first = processedMsgIds.values().next().value;
        processedMsgIds.delete(first);
    }
}

// Intelligent language detection defaulting to English ('en')
function detectMessageLanguage(text, currentLang = 'en') {
    if (!text) return currentLang;
    const lower = text.toLowerCase().trim();

    // Explicit requests for English
    if (/in english|speak english|english please|reply in english|switch to english/i.test(lower)) {
        return 'en';
    }

    // Hausa detection (common greetings / vocabulary)
    if (/\b(sannu|ina kwana|yaya kake|yaya dai|barka|lafiya|nagode|don allah|yau|wannan|ba ni da|turanci|hausa)\b/i.test(lower)) {
        return 'ha';
    }

    // Yoruba detection
    if (/\b(bawo|e kaaro|e ku|eku|kilo nsele|nje|o ti|bẹẹni|jọwọ|yoruba)\b/i.test(lower)) {
        return 'yo';
    }

    // Igbo detection
    if (/\b(kedu|ndewo|daalu|kedụ|ị nọ|biko|igbo)\b/i.test(lower)) {
        return 'ig';
    }

    // Default to English whenever English words or standard NAFDAC queries appear
    return 'en';
}

function cleanAiResponse(text) {
    if (!text) return '';
    return text
        .replace(/<\|im_end\|>/g, '')
        .replace(/<\|im_start\|>/g, '')
        .replace(/<\|eom_id\|>/g, '')
        .replace(/<s>/g, '')
        .replace(/<\/s>/g, '')
        .trim();
}

// ----------------------------------------------------
// Live HTTP Web Dashboard for pairing & monitoring
// ----------------------------------------------------
const server = http.createServer((req, res) => {
    if (req.url === '/status') {
        res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
        res.end(JSON.stringify(botState));
        return;
    }

    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Confam AI - WhatsApp Link</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #f0f2f5;
      margin: 0;
      padding: 24px;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }
    .card {
      background: #ffffff;
      border-radius: 16px;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
      max-width: 440px;
      width: 100%;
      padding: 36px 28px;
      text-align: center;
    }
    .badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 16px;
    }
    .badge.scanning { background: #e8f5e9; color: #1b5e20; }
    .badge.connected { background: #dcf8c6; color: #075e54; }
    .badge.init { background: #e0f2fe; color: #0369a1; }
    h1 {
      font-size: 24px;
      color: #111b21;
      margin: 0 0 8px;
    }
    p {
      color: #667781;
      font-size: 14px;
      line-height: 1.5;
      margin: 0 0 24px;
    }
    #qrcode {
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 256px;
      margin: 16px 0;
    }
    .success-box {
      display: none;
      padding: 24px 16px;
    }
    .success-icon {
      font-size: 64px;
      margin-bottom: 12px;
    }
    .instructions {
      text-align: left;
      background: #f8fafc;
      border-radius: 10px;
      padding: 14px 18px;
      font-size: 13px;
      color: #334155;
      margin-top: 20px;
    }
    .instructions ol {
      margin: 6px 0 0 18px;
      padding: 0;
    }
    .instructions li {
      margin-bottom: 4px;
    }
    .pulse-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #22c55e;
      margin-right: 6px;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { opacity: 0.4; }
      50% { opacity: 1; }
      100% { opacity: 0.4; }
    }
  </style>
</head>
<body>
  <div class="card">
    <div id="badge-container">
      <span class="badge init" id="status-badge"><span class="pulse-dot"></span>Connecting...</span>
    </div>
    
    <div id="qr-section">
      <h1>Link Confam AI</h1>
      <p>Scan with WhatsApp on <strong>+234 906 097 0339</strong> to activate the bot.</p>
      
      <div id="qrcode">Loading fresh QR code...</div>
      
      <div class="instructions">
        <strong>How to link:</strong>
        <ol>
          <li>Open WhatsApp on your phone</li>
          <li>Tap <strong>Settings</strong> (or <strong>⋮</strong>) &rarr; <strong>Linked devices</strong></li>
          <li>Tap <strong>Link a device</strong></li>
          <li>Point your phone at this screen</li>
        </ol>
      </div>
    </div>

    <div id="success-section" class="success-box">
      <div class="success-icon">🎉</div>
      <h1 style="color: #075e54;">WhatsApp Connected!</h1>
      <p>Confam AI is actively online. Send any voice note or text message to <strong>+234 906 097 0339</strong> to verify products.</p>
    </div>
  </div>

  <script>
    let currentQr = null;

    async function poll() {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        
        const badge = document.getElementById('status-badge');
        const qrSec = document.getElementById('qr-section');
        const succSec = document.getElementById('success-section');

        if (data.status === 'ready' || data.status === 'authenticated') {
          badge.className = 'badge connected';
          badge.innerHTML = '● Online & Active (High-Availability)';
          qrSec.style.display = 'none';
          succSec.style.display = 'block';
        } else if (data.status === 'qr' && data.qr) {
          badge.className = 'badge scanning';
          badge.innerHTML = '<span class="pulse-dot"></span>Live QR (Auto-refreshing)';
          qrSec.style.display = 'block';
          succSec.style.display = 'none';

          if (data.qr !== currentQr) {
            currentQr = data.qr;
            const container = document.getElementById('qrcode');
            container.innerHTML = '';
            new QRCode(container, {
              text: currentQr,
              width: 256,
              height: 256,
              correctLevel: QRCode.CorrectLevel.M
            });
          }
        }
      } catch (e) {}
    }

    setInterval(poll, 1500);
    poll();
  </script>
</body>
</html>`);
});

server.listen(HTTP_PORT, () => {
    console.log(`[DASHBOARD] Live pairing dashboard available at http://localhost:${HTTP_PORT}`);
});

// ----------------------------------------------------
// Client Initialization with Anti-Freeze Chromium Flags
// ----------------------------------------------------
let client = null;

function initClient() {
    console.log('[INITIALIZING] Launching WhatsApp Web client...');

    client = new Client({
        authStrategy: new LocalAuth({ dataPath: './.wwebjs_auth' }),
        puppeteer: {
            executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
            headless: true,
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--disable-gpu',
                // Anti-sleep and anti-throttling flags:
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding'
            ]
        }
    });

    client.on('qr', (qr) => {
        console.log('\n[QR CODE] Pairing QR code generated. Scan at http://localhost:' + HTTP_PORT + ':\n');
        qrcode.generate(qr, { small: true });
        botState.status = 'qr';
        botState.qr = qr;
    });

    client.on('authenticated', () => {
        console.log('[STATUS] WhatsApp authenticated successfully!');
        botState.status = 'authenticated';
    });

    client.on('auth_failure', (msg) => {
        console.error('[STATUS] WhatsApp authentication failure:', msg);
        botState.status = 'auth_failure';
    });

    client.on('ready', () => {
        console.log('\n========================================================');
        console.log(' >>> CONFAM AI BOT IS ONLINE (HIGH-AVAILABILITY ACTIVE) <<<');
        console.log('========================================================\n');
        botState.status = 'ready';
    });

    client.on('change_state', (state) => {
        console.log(`[STATE CHANGE] WhatsApp state changed to: ${state}`);
    });

    client.on('disconnected', async (reason) => {
        console.warn(`[DISCONNECTED] WhatsApp disconnected (${reason}). Reconnecting in 5s...`);
        botState.status = 'disconnected';
        try {
            await client.destroy();
        } catch (e) {}
        setTimeout(initClient, 5000);
    });

    // Handle messages
    async function handleMessage(msg) {
        if (!msg || !msg.from) return;

        // Skip groups, broadcast status, and outgoing messages
        if (msg.from.includes('@g.us') || msg.from === 'status@broadcast' || msg.fromMe) {
            return;
        }

        const msgId = msg.id?._serialized || String(msg.id);
        if (processedMsgIds.has(msgId)) {
            return; // Already processed
        }
        markProcessed(msgId);

        const sender = msg.from.replace('@c.us', '');
        const userText = (msg.body || '').trim();
        const isVoice = msg.hasMedia && (msg.type === 'ptt' || msg.type === 'audio');
        const isImage = msg.hasMedia && msg.type === 'image';

        console.log(`[INCOMING] From ${sender}: "${userText || '[' + msg.type + ']'}"`);

        const session = getSession(sender);

        // Send real-time typing / recording state
        let chat = null;
        try {
            chat = await msg.getChat();
            if (isVoice) {
                await chat.sendStateRecording();
            } else {
                await chat.sendStateTyping();
            }
        } catch (e) {}

        const clearTyping = async () => {
            try {
                if (chat) await chat.clearState();
            } catch (e) {}
        };

        // ----------------------------------------------------
        // VOICE NOTE PROCESSING
        // ----------------------------------------------------
        if (isVoice) {
            console.log(`[VOICE NOTE] Processing voice note from ${sender}...`);
            try {
                await new Promise(res => setTimeout(res, 1200));

                let media = null;
                for (let attempt = 1; attempt <= 4; attempt++) {
                    try {
                        media = await msg.downloadMedia();
                        if (media && media.data) break;
                    } catch (dmErr) {
                        console.warn(`[VOICE NOTE] Attempt ${attempt} failed: ${dmErr?.message || dmErr}`);
                    }
                    if (attempt < 4) await new Promise(res => setTimeout(res, 1200));
                }

                if (!media || !media.data) {
                    console.error('[VOICE NOTE] Empty or failed media download after retries');
                    await clearTyping();
                    await msg.reply("I couldn't load your voice note. Please speak clearly, or send the product name or NAFDAC number as text.");
                    return;
                }

                const audioBuffer = Buffer.from(media.data, 'base64');
                const mime = media.mimetype || 'audio/ogg';
                let ext = 'ogg';
                if (mime.includes('mp4')) ext = 'mp4';
                else if (mime.includes('wav')) ext = 'wav';
                else if (mime.includes('webm')) ext = 'webm';

                console.log(`[VOICE NOTE] Sending ${audioBuffer.length} bytes to ${NATLAS_VOICE_URL}...`);

                const form = new FormData();
                const blob = new Blob([audioBuffer], { type: mime });
                form.append('file', blob, `voice-note.${ext}`);
                form.append('model', NATLAS_MODEL);
                form.append('language', 'auto');

                const prior = session.messages.filter(m => m.role === 'user' || m.role === 'assistant').slice(-8);
                if (prior.length > 0) {
                    form.append('messages', JSON.stringify(prior));
                }

                const res = await fetch(NATLAS_VOICE_URL, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${NATLAS_API_KEY}`,
                        'User-Agent': 'confamai-bot/1.0 (+https://ai.saynet.ng)'
                    },
                    body: form
                });

                if (!res.ok) {
                    const errBody = await res.text();
                    console.error(`[VOICE ERROR] HTTP ${res.status}:`, errBody);
                    throw new Error(`Voice endpoint returned HTTP ${res.status}`);
                }

                const data = await res.json();
                let aiReply = '';
                if (Array.isArray(data.choices) && data.choices[0] && data.choices[0].message) {
                    aiReply = data.choices[0].message.content || '';
                }
                if (!aiReply) {
                    aiReply = data.reply || data.answer || data.message || '';
                }
                const transcript = (data.transcript || data.text || data.transcription || '').trim();

                console.log(`[VOICE TRANSCRIPT] "${transcript}"`);

                aiReply = cleanAiResponse(aiReply);
                if (!aiReply) {
                    aiReply = "I heard your voice note, but could not find a product match. Please mention the specific product name or NAFDAC registration number.";
                }

                session.messages.push({ role: 'user', content: transcript || '[Voice Note]' });
                session.messages.push({ role: 'assistant', content: aiReply });
                if (session.messages.length > 8) session.messages = session.messages.slice(-8);

                let finalReply = aiReply;
                if (transcript) {
                    finalReply = `🎤 _"${transcript}"_\n\n${aiReply}`;
                }

                console.log(`[OUTGOING VOICE REPLY] To ${sender}: "${aiReply.substring(0, 80)}..."`);
                await clearTyping();
                await msg.reply(finalReply);
                return;

            } catch (err) {
                console.error('[VOICE ERROR] Failed processing voice note:', err.message);
                await clearTyping();
                await msg.reply("I had trouble transcribing that voice note. Please speak clearly, or send the product name or NAFDAC number as text.");
                return;
            }
        }

        // ----------------------------------------------------
        // IMAGE PROCESSING GUIDANCE
        // ----------------------------------------------------
        if (isImage) {
            await clearTyping();
            await msg.reply("I received your product photo! To verify it against the NAFDAC database, please also type the product name or the NAFDAC registration number (e.g. 04-1234 or A4-5678) visible on the pack.");
            return;
        }

        // ----------------------------------------------------
        // TEXT MESSAGE PROCESSING
        // ----------------------------------------------------
        if (!userText) {
            await clearTyping();
            return;
        }

        const targetLang = detectMessageLanguage(userText, session.preferredLang);
        session.preferredLang = targetLang;

        if (targetLang === 'en' && /in english|speak english|english please/i.test(userText)) {
            session.messages = [];
        }

        session.messages.push({ role: 'user', content: userText });
        if (session.messages.length > 8) {
            session.messages = session.messages.slice(-8);
        }

        try {
            console.log(`[AI CALL] Querying N-ATLaS (lang=${targetLang})...`);
            const response = await axios.post(
                NATLAS_URL,
                {
                    model: NATLAS_MODEL,
                    messages: session.messages,
                    language: targetLang,
                    max_tokens: 512,
                    temperature: 0.2,
                    stream: false
                },
                {
                    headers: {
                        'Authorization': `Bearer ${NATLAS_API_KEY}`,
                        'Content-Type': 'application/json',
                        'User-Agent': 'confamai-bot/1.0 (+https://ai.saynet.ng)'
                    },
                    timeout: 30000
                }
            );

            const choices = response.data && response.data.choices;
            if (!choices || choices.length === 0) {
                throw new Error('N-ATLaS returned empty choices');
            }

            let aiReply = choices[0].message?.content || '';
            aiReply = cleanAiResponse(aiReply);

            if (!aiReply) {
                aiReply = "I could not process that request. Please send a product name or NAFDAC number to verify.";
            }

            session.messages.push({ role: 'assistant', content: aiReply });

            console.log(`[OUTGOING] To ${sender} (${targetLang}): "${aiReply.substring(0, 80)}..."`);
            await clearTyping();
            await msg.reply(aiReply);

        } catch (err) {
            console.error(`[ERROR] AI call failed for ${sender}:`, err.response?.data || err.message);
            await clearTyping();
            await msg.reply("Confam AI is currently experiencing high verification traffic. Please send the product name or NAFDAC number again in a moment.");
        }
    }

    client.on('message', handleMessage);
    client.on('message_create', handleMessage);

    client.initialize();
}

// ----------------------------------------------------
// Keep-Alive Ping & Anti-Sleep Watchdog (every 30 seconds)
// ----------------------------------------------------
setInterval(async () => {
    if (!client || !client.pupPage) return;

    try {
        await client.pupPage.evaluate(() => {
            return window.Store?.AppState?.state || 'active';
        });
        botState.lastPing = Date.now();
    } catch (e) {
        console.warn('[WATCHDOG] Keep-alive ping failed. Testing connection state...');
        try {
            const state = await client.getState();
            console.log('[WATCHDOG] Client state:', state);
            if (state !== 'CONNECTED') {
                throw new Error('State is ' + state);
            }
        } catch (stateErr) {
            console.error('[WATCHDOG] Client frozen or unreachable. Triggering self-recovery...');
            try {
                await client.destroy();
            } catch (err) {}
            setTimeout(initClient, 3000);
        }
    }
}, 30000);

// Launch bot
initClient();
