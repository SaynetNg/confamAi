# The Ultimate Beginner's Guide: Confam AI WhatsApp + STT Integration

> **Note (current behaviour).** The bot no longer transcribes locally and there
> is no `audio.py`. A voice note is answered **voice-first**: it is downloaded
> and POSTed to the N-ATLaS server's `/v1/audio/chat`, which transcribes it with
> its preloaded NCAIR1 models *and* runs the chat model on the result, returning
> the reply. See `README.md` section 6 and `router.py:_ask_voice`. The steps
> below that build a local `audio.py` are kept only as background and are no
> longer how the code works.

This guide will walk you and your teammate through the entire process of connecting your `confamai_bot` to WhatsApp and enabling Speech-to-Text (STT) for voice notes, from absolutely zero.

---

## Phase 1: Get Your WhatsApp API Credentials

Before writing any code, you need permission from Meta (Facebook) to build a WhatsApp bot.

1. **Go to the Meta Developer Portal:** Visit [developers.facebook.com](https://developers.facebook.com/) and log in with your Facebook account.
2. **Create an App:** 
   - Click **My Apps** -> **Create App**.
   - Select **Other** -> **Next**.
   - Select **Business** -> **Next**.
   - Name your app (e.g., "Confam AI Bot") and click **Create App**.
3. **Add WhatsApp to Your App:**
   - On the next screen, scroll down to **WhatsApp** and click **Set Up**.
   - Meta will automatically give you a **Test Phone Number** to play with.
4. **Get Your Secret Tokens:**
   - On the left sidebar, click **WhatsApp** -> **API Setup**.
   - Here you will see three very important things. Copy them into a safe notepad:
     - **Temporary Access Token** (This is your `WHATSAPP_TOKEN`).
     - **Phone Number ID** (This is your `PHONE_NUMBER_ID`).
   - *Note: The temporary token expires in 24 hours. When you are ready for production, you will need to create a Permanent System User Token in Meta Business Settings.*

---

## Phase 2: Update Your `confamai_bot` Code

Your teammate has already built a great foundation in the `confamai_bot` folder! Now we just need to add the ability to download audio and transcribe it.

### Step 1: Add Media Downloading
Open `confamAi-main\confamai_bot\whatsapp.py`. Add this new function inside the `WhatsAppClient` class. This teaches your bot how to download `.ogg` audio files from Meta.

```python
    async def download_media(self, media_id: str) -> bytes:
        """Download an audio/image file from Meta's servers."""
        # Get the download URL from Meta
        url_resp = await self.client.get(
            f"https://graph.facebook.com/v18.0/{media_id}",
            headers={"Authorization": f"Bearer {self.cfg.whatsapp_token}"}
        )
        url_resp.raise_for_status()
        media_url = url_resp.json()["url"]
        
        # Download the actual audio file bytes
        media_resp = await self.client.get(
            media_url,
            headers={"Authorization": f"Bearer {self.cfg.whatsapp_token}"}
        )
        media_resp.raise_for_status()
        return media_resp.content
```

### Step 2: Add the STT (Speech-to-Text) Brain
Create a brand new file called `audio.py` inside the `confamai_bot` folder. This file loads your NCAIR1 AI models to convert voice to text.

```python
import torch
from transformers import pipeline

class VoiceTranscriber:
    def __init__(self):
        print("Loading NCAIR1 Speech-to-Text models... This will take a moment.")
        # Load the 4 Hugging Face models into memory
        self.models = {
            "en": pipeline("automatic-speech-recognition", model="NCAIR1/NigerianAccentedEnglish"),
            "ha": pipeline("automatic-speech-recognition", model="NCAIR1/Hausa-ASR"),
            "yo": pipeline("automatic-speech-recognition", model="NCAIR1/Yoruba-ASR"),
            "ig": pipeline("automatic-speech-recognition", model="NCAIR1/Igbo-ASR"),
        }

    def transcribe(self, audio_bytes: bytes, language_code: str) -> str:
        # Default to English if the user hasn't set a language yet
        if language_code not in self.models:
            language_code = "en"
            
        print(f"Transcribing using {language_code} model...")
        asr_pipeline = self.models[language_code]
        
        # Convert the audio into text!
        result = asr_pipeline(audio_bytes)
        return result["text"]

# Create a single instance to be used by the router
transcriber = VoiceTranscriber()
```

### Step 3: Tell the Router to Listen to Voice Notes
Open `confamAi-main\confamai_bot\router.py`. At the very top, add this import:
`from .audio import transcriber`

Then, scroll down to the `handle_non_text` function. Replace it with this code so it handles audio instead of rejecting it:

```python
    async def handle_non_text(self, item: dict) -> str:
        """Process incoming audio voice notes."""
        phone = normalize_phone(item.get("from", ""))
        if not phone or phone in self.opted_out or not self.cfg.is_allowed(phone):
            return ""
        if not await self.store.mark_seen(item.get("id", ""), DEDUPE_TTL):
            return ""
            
        session = await self.store.get(phone)
        msg_type = item.get("type")
        
        if msg_type == "audio":
            audio_id = item["audio"]["id"]
            
            # Let the user know we are thinking
            await self._send(phone, "🎧 Listening to your voice note...")
            
            # 1. Download the voice note
            audio_bytes = await self.whatsapp.download_media(audio_id)
            
            # 2. Transcribe it using the user's selected language
            user_lang = session.language or "en"
            transcribed_text = transcriber.transcribe(audio_bytes, user_lang)
            print(f"[router] Transcribed text: {transcribed_text}")
            
            # 3. Pretend the user typed the transcribed text, and send it to the AI!
            mock_text_item = item.copy()
            mock_text_item["type"] = "text"
            mock_text_item["text"] = {"body": transcribed_text}
            
            return await self.handle_text(mock_text_item)
            
        # If it's a photo or sticker, reject it
        return await self._send(phone, "Sorry, I can only understand text messages and voice notes right now!")
```

---

## Phase 3: Connect Meta to Your Bot

Now you have the code, but Meta doesn't know where your bot lives! You need to connect them.

1. **Run Your Bot Locally:**
   Open your terminal in the `confamai_bot` folder and start your webhook server. (Ensure your `.env` file has the `WHATSAPP_TOKEN` you copied earlier!).
   ```bash
   python -m confamai_bot
   ```
   *Your bot is now running on your computer, probably on port 8000 or 5000.*

2. **Expose Your Bot to the Internet (Ngrok):**
   Meta cannot send messages to "localhost". You need a public URL. Download a free tool called **ngrok** and run:
   ```bash
   ngrok http 8000
   ```
   *(Replace 8000 with whatever port your bot is running on). Ngrok will give you a public URL like `https://a1b2-c3d4.ngrok.io`.*

3. **Tell Meta Your URL:**
   - Go back to the Meta Developer Portal -> **WhatsApp** -> **Configuration**.
   - Click **Edit** next to Webhook.
   - **Callback URL:** Paste your ngrok URL here, and add your webhook path at the end (e.g., `https://a1b2-c3d4.ngrok.io/webhook`).
   - **Verify Token:** Type any secret password here (e.g., `my_secret_confam_password`). Make sure this exact password is saved in your `.env` file so your bot can verify it!
   - Click **Verify and Save**. (If it succeeds, your bot is officially connected!)

4. **Subscribe to Messages:**
   - Right below that, click **Manage** under Webhook fields.
   - Check the box next to **messages** and click Done.

---

## Phase 4: The Final Test!

1. Go to the Meta Developer Portal -> **WhatsApp** -> **API Setup**.
2. Scroll down to the **To** field, and add your personal WhatsApp phone number. Meta will send you a code on WhatsApp to verify your number.
3. Once verified, save the test phone number Meta gave you into your phone's contacts as "Confam AI".
4. **Send a WhatsApp message!** Type "Hello" to the bot. 
   - Meta will instantly ping your ngrok URL.
   - Your bot will receive the ping, ask the Kaggle AI for an answer, and send the answer back to your phone!
5. **Test Voice!** Send a voice note saying a NAFDAC number. The bot will download the audio, run your STT model, transcribe it, and reply!

### Summary Checklist for Production
When you are ready to move this off your laptop and make it available 24/7 to the public:
- [ ] Move the `confamai_bot` code to a real server (like Render, Heroku, or an AWS VPS).
- [ ] Generate a Permanent System User Token in Meta instead of the temporary 24-hour one.
- [ ] Update the Webhook URL in Meta to point to your real server instead of ngrok.
- [ ] Turn off `N_ATLAS_LANGUAGE_TAGGING=0` on your Kaggle server so the AI doesn't hallucinate!
