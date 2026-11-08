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
console.log(' Confam AI - WhatsApp Bot (Voice + Text N-ATLaS)');
console.log(' Text Endpoint: ', NATLAS_URL);
console.log(' Voice Endpoint:', NATLAS_VOICE_URL);
console.log(' Model:         ', NATLAS_MODEL);
console.log(' Web Dashboard: ', `http://localhost:${HTTP_PORT}`);
console.log('====================================================\n');

// State tracking for the web UI
let botState = {
    status: 'initializing',
    qr: null,
    phone: '2349060970339'
};

// In-memory conversation session store (phone -> { messages, lastActive })
const sessions = new Map();
const SESSION_TTL_MS = 30 * 60 * 1000; // 30 minutes

function getSession(phone) {
    const now = Date.now();
    let session = sessions.get(phone);
    if (!session || (now - session.lastActive > SESSION_TTL_MS)) {
        session = { messages: [], lastActive: now };
        sessions.set(phone, session);
    }
    session.lastActive = now;
    return session;
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
          badge.innerHTML = '● Online & Active (Voice + Text)';
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
// WhatsApp Client setup
// ----------------------------------------------------
const client = new Client({
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
            '--disable-gpu'
        ]
    }
});

client.on('qr', (qr) => {
    console.log('\n[QR CODE] New live QR generated. Scan from terminal or http://localhost:' + HTTP_PORT + ':\n');
    qrcode.generate(qr, { small: true });

    botState.status = 'qr';
    botState.qr = qr;

    try {
        fs.writeFileSync('C:\\Users\\USER\\Desktop\\whatsapp_qr.html', `<meta http-equiv="refresh" content="0; url=http://localhost:${HTTP_PORT}">`, 'utf8');
    } catch (e) {}
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
    console.log(' >>> CONFAM AI BOT IS ONLINE (VOICE + TEXT READY) <<<');
    console.log('========================================================\n');
    botState.status = 'ready';
});

client.on('message', async (msg) => {
    // Ignore group chats, status broadcasts, and bot's own messages
    if (msg.from.includes('@g.us') || msg.from === 'status@broadcast' || msg.fromMe) {
        return;
    }

    const sender = msg.from.replace('@c.us', '');
    const userText = (msg.body || '').trim();
    const isVoice = msg.hasMedia && (msg.type === 'ptt' || msg.type === 'audio');
    const isImage = msg.hasMedia && msg.type === 'image';

    console.log(`[INCOMING] From ${sender}: "${userText || '[' + msg.type + ']'}"`);

    const session = getSession(sender);

    // ----------------------------------------------------
    // VOICE NOTE PROCESSING (Same as Browser /v1/audio/chat)
    // ----------------------------------------------------
    if (isVoice) {
        console.log(`[VOICE NOTE] Receiving voice note from ${sender}...`);
        try {
            const media = await msg.downloadMedia();
            if (!media || !media.data) {
                console.error('[VOICE NOTE] Empty or failed media download');
                await msg.reply("I couldn't download your voice note. Please speak again or send your question as text.");
                return;
            }

            const audioBuffer = Buffer.from(media.data, 'base64');
            const mime = media.mimetype || 'audio/ogg';
            let ext = 'ogg';
            if (mime.includes('mp4')) ext = 'mp4';
            else if (mime.includes('wav')) ext = 'wav';
            else if (mime.includes('webm')) ext = 'webm';

            console.log(`[VOICE NOTE] Uploading ${audioBuffer.length} bytes to ${NATLAS_VOICE_URL}...`);

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

            // Save conversation turn
            session.messages.push({ role: 'user', content: transcript || '[Voice Note]' });
            session.messages.push({ role: 'assistant', content: aiReply });
            if (session.messages.length > 8) session.messages = session.messages.slice(-8);

            // Send reply with transcript quote
            let finalReply = aiReply;
            if (transcript) {
                finalReply = `🎤 _"${transcript}"_\n\n${aiReply}`;
            }

            console.log(`[OUTGOING VOICE REPLY] To ${sender}: "${aiReply.substring(0, 80)}..."`);
            await msg.reply(finalReply);
            return;

        } catch (err) {
            console.error('[VOICE ERROR] Failed to process voice note:', err.message);
            await msg.reply("I had trouble transcribing that voice note. Please speak clearly, or send the product name or NAFDAC number as text.");
            return;
        }
    }

    // ----------------------------------------------------
    // IMAGE PROCESSING GUIDANCE
    // ----------------------------------------------------
    if (isImage) {
        await msg.reply("I received your product photo! To verify it against the NAFDAC database, please also type the product name or the NAFDAC registration number (e.g. 04-1234 or A4-5678) visible on the pack.");
        return;
    }

    // ----------------------------------------------------
    // TEXT MESSAGE PROCESSING
    // ----------------------------------------------------
    if (!userText) {
        return;
    }

    session.messages.push({ role: 'user', content: userText });
    if (session.messages.length > 8) {
        session.messages = session.messages.slice(-8);
    }

    try {
        const response = await axios.post(
            NATLAS_URL,
            {
                model: NATLAS_MODEL,
                messages: session.messages,
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

        console.log(`[OUTGOING] To ${sender}: "${aiReply.substring(0, 80)}..."`);
        await msg.reply(aiReply);

    } catch (err) {
        console.error(`[ERROR] AI call failed for ${sender}:`, err.response?.data || err.message);
        await msg.reply("Confam AI is currently experiencing high verification traffic. Please send the product name or NAFDAC number again in a moment.");
    }
});

client.initialize();
