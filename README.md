# Aries Health Assistant - Deployment Guide

A health assistant chatbot powered by HuggingFace Phi-3, with medicine price comparison and emergency support.

## 🚀 Quick Deploy

### Step 1: Get HuggingFace Token
1. Go to [HuggingFace](https://huggingface.co)
2. Sign up/login
3. Go to Settings → Access Tokens
4. Create a **Read** token
5. Copy the token (starts with `hf_...`)

### Step 2: Deploy Backend (Railway)
1. Push this code to GitHub
2. Go to [Railway](https://railway.app)
3. Click "New Project" → "Deploy from GitHub"
4. Select your repository
5. Add Environment Variable:
   - Key: `HF_TOKEN`
   - Value: Your HuggingFace token
6. Deploy!

You'll get a URL like: `https://your-app.up.railway.app`

### Step 3: Deploy Frontend (Vercel)
1. Go to [Vercel](https://vercel.com)
2. Import from GitHub
3. Select your repository
4. **IMPORTANT**: Update `index.html` line 341:
   ```javascript
   const SERVER_URL = 'https://your-actual-railway-url.up.railway.app';
   ```
5. Deploy!

## 🧪 Test Your Deployment

### Backend Test
```bash
curl -X POST https://your-backend.up.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Aries"}'
```

### Frontend Test
Visit your Vercel URL and try:
- "What's the price of paracetamol?"
- "I have a headache"
- "Find nearest hospital"

## 📁 Project Structure

```
├── app.py              # Flask backend (Railway)
├── requirements.txt    # Python dependencies
├── runtime.txt         # Python version
├── Procfile           # Railway deployment config
├── index.html         # Frontend (Vercel)
├── package.json       # Node.js config for Vercel
└── README.md          # This file
```

## 🔧 Features

- **Chat Interface**: Talk to Aries about health concerns
- **Medicine Prices**: Compare prices across pharmacies
- **Emergency Detection**: Automatic escalation for critical situations
- **State Detection**: CASUAL, CARE_MODE, MEDICAL_MODE, CRITICAL_MODE
- **Browser TTS**: Aries speaks replies using browser's speech synthesis
- **Persistent Memory**: Chat history saved to file (survives server restarts)
- **Mobile Responsive**: Works on all devices

## 💾 Memory Storage

Chat conversations are stored in:
- **Location**: Railway's temp directory (`/tmp/aries_memory.json`)
- **Persistence**: Survives server restarts (Railway keeps temp files for ~24-48 hours)
- **Capacity**: Last 200 conversations
- **Access**: Visit `/memory` endpoint to view recent chats

**Note**: For longer-term storage, consider upgrading to Railway's persistent volumes or external database.

## 🛠️ Local Development

### Backend
```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here
python app.py
```

### Frontend
Open `index.html` in browser or:
```bash
npm install
npm start
```

## 🔒 Security Notes

- HF_TOKEN is server-side only
- No sensitive data in frontend
- CORS enabled for cross-origin requests
- Rate limiting recommended for production

## 📝 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | HuggingFace API token | Yes |
| `PORT` | Server port (auto-set by Railway) | No |

## 🚨 Important Notes

1. **Update Frontend URL**: Change `SERVER_URL` in `index.html` after backend deployment
2. **HuggingFace Limits**: Free tier has request limits
3. **No TTS**: Text-to-speech removed for deployment simplicity
4. **Demo Medicine Prices**: Using sample data instead of live scraping

## 🆘 Troubleshooting

### Backend Issues
- Check Railway logs for errors
- Verify HF_TOKEN is set correctly
- Test `/health` endpoint

### Frontend Issues
- Check browser console for errors
- Verify SERVER_URL points to your Railway app
- Test with simple curl request first

### CORS Issues
- Backend has CORS enabled
- If issues persist, check Railway deployment logs

## 📞 Support

If you encounter issues:
1. Check Railway/Vercel deployment logs
2. Test backend endpoints directly
3. Verify HuggingFace token permissions
4. Check browser network tab for errors

---

**Made with ❤️ using HuggingFace Phi-3, Railway, and Vercel**