const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

const NATLAS_URL = process.env.N_ATLAS_BASE_URL || 'https://ai.saynet.ng/v1/chat/completions';
const NATLAS_API_KEY = process.env.N_ATLAS_API_KEY || 'KgRRl_6vEp7YRCY6agQ9HjH0RFk6T6hr9KYQq7YTRfRrKeWQIyFs_biogXFvK_Lo';
const NATLAS_MODEL = process.env.N_ATLAS_MODEL || 'n-atlas';

console.log('====================================================');
console.log(' Confam AI - WhatsApp Bot (Direct N-ATLaS Engine)');
console.log(' N-ATLaS Endpoint:', NATLAS_URL);
console.log(' Model:', NATLAS_MODEL);
console.log('====================================================\n');

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

// Strip token artifacts that might leak from raw LLM outputs
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
    console.log('\n[QR CODE] WhatsApp needs pairing. Scan this QR code:\n');
    qrcode.generate(qr, { small: true });

    try {
        const html = `<!DOCTYPE html>
<html>
<head>
  <title>Confam AI - WhatsApp QR Code</title>
  <meta charset="utf-8">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; background: #f0f2f5; margin: 0; }
    .card { background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); text-align: center; }
    h2 { color: #075e54; margin: 0 0 16px; }
  </style>
</head>
<body>
  <div class="card">
    <h2>Confam AI WhatsApp Link</h2>
    <p>Scan to link: <strong>+234 906 097 0339</strong></p>
    <div id="qrcode" style="display:flex;justify-content:center;margin:20px 0;"></div>
    <p style="color:#666;font-size:14px;">WhatsApp -> Linked Devices -> Link a Device</p>
  </div>
  <script>
    new QRCode(document.getElementById("qrcode"), {
      text: ${JSON.stringify(qr)},
      width: 256,
      height: 256
    });
  </script>
</body>
</html>`;
        fs.writeFileSync('C:\\Users\\USER\\Desktop\\whatsapp_qr.html', html, 'utf8');
    } catch (e) {}
});

client.on('authenticated', () => {
    console.log('[STATUS] WhatsApp authenticated successfully!');
});

client.on('auth_failure', (msg) => {
    console.error('[STATUS] WhatsApp authentication failure:', msg);
});

client.on('ready', () => {
    console.log('\n========================================================');
    console.log(' >>> CONFAM AI BOT IS ONLINE AND ACTIVELY RESPONDING <<<');
    console.log('========================================================\n');
});

client.on('message', async (msg) => {
    // Ignore group chats, status broadcasts, and messages from bot itself
    if (msg.from.includes('@g.us') || msg.from === 'status@broadcast' || msg.fromMe) {
        return;
    }

    const sender = msg.from.replace('@c.us', '');
    const userText = (msg.body || '').trim();

    console.log(`[INCOMING] From +${sender}: "${userText || '[' + msg.type + ']'}"`);

    // Handle voice notes / media gracefully
    if (msg.hasMedia && (msg.type === 'ptt' || msg.type === 'audio')) {
        await msg.reply("I received your voice note! For the fastest and most accurate NAFDAC verification, please type the product name or NAFDAC registration number (e.g. 04-1234 or A4-5678).");
        return;
    }

    if (!userText) {
        return;
    }

    const session = getSession(sender);
    session.messages.push({ role: 'user', content: userText });

    // Keep session context window bounded to last 8 messages (4 turns)
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

        // Save AI response in session history
        session.messages.push({ role: 'assistant', content: aiReply });

        console.log(`[OUTGOING] To +${sender}: "${aiReply.substring(0, 80)}..."`);
        await msg.reply(aiReply);

    } catch (err) {
        console.error(`[ERROR] AI call failed for +${sender}:`, err.response?.data || err.message);
        await msg.reply("Confam AI is currently experiencing high verification traffic. Please send the product name or NAFDAC number again in a moment.");
    }
});

client.initialize();
