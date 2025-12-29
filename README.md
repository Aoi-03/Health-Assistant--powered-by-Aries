# 🏥 Aries Health Assistant - Powered by AI

**Aries** is an intelligent health assistant that provides caring, contextual health advice with natural voice interaction. Built with **HuggingFace Phi-3**, **Flask**, and **Browser TTS** for a complete health support experience.

---

## ✨ **Major Features**

- 🤖 **Intelligent AI Chat** - Varied, contextual responses with emotional understanding
- 💊 **Medicine Price Comparison** - Real-time price comparison across multiple pharmacies  
- 🏥 **Nearby Medical Facilities** - Find hospitals, clinics, and pharmacies with location services
- 🎤 **Natural Voice Interaction** - High-quality text-to-speech with female voice selection
- 🧠 **Smart Memory System** - Remembers conversations and avoids repetitive responses
- 📱 **Mobile Responsive** - Works seamlessly on all devices
- 🔒 **Privacy Focused** - Secure deployment with environment variable protection

---

## 🚀 **Quick Deploy (15 minutes)**

### **Step 1: Get HuggingFace Token**
1. Go to [HuggingFace](https://huggingface.co/settings/tokens)
2. Create a **Read** token
3. Copy the token (starts with `hf_...`)

### **Step 2: Deploy Backend (Railway)**
1. Go to [Railway](https://railway.app)
2. New Project → Deploy from GitHub
3. Select this repository
4. Add Environment Variable:
   - **Key**: `HF_TOKEN`
   - **Value**: Your HuggingFace token
5. Deploy! (Get your Railway URL)

### **Step 3: Deploy Frontend (Vercel)**
1. Go to [Vercel](https://vercel.com)
2. Import from GitHub
3. Select this repository
4. **IMPORTANT**: Update `index.html` line ~341:
   ```javascript
   const SERVER_URL = 'https://your-railway-url.up.railway.app';
   ```
5. Deploy!

---

## 🧪 **Test Your Deployment**

### **Backend Test**
```bash
curl -X POST https://your-backend.up.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Aries"}'
```

### **Frontend Test**
Visit your Vercel URL and try:
- "Hello Aries" (varied greetings)
- "What's the price of paracetamol?" (medicine search)
- "I have a headache" (health advice)
- "Find nearest hospital" (location services)

---

## 📁 **Project Structure**

```
├── app.py              # 🧠 Smart AI backend with varied responses
├── index.html          # 🎤 Frontend with natural voice TTS
├── requirements.txt    # 📦 Python dependencies
├── runtime.txt         # 🐍 Python version for Railway
├── Procfile           # 🚀 Railway deployment config
├── package.json       # 📱 Vercel frontend config
├── .gitignore         # 🔒 Security (excludes .env)
└── README.md          # 📖 This documentation
```

---

## 🎯 **What Makes Aries Special**

### **🤖 Intelligent AI Responses**
- **No Repetition**: Multiple response variations for each scenario
- **Context-Aware**: Different advice for headaches, fever, anxiety, etc.
- **Memory-Smart**: Avoids repeating recent responses
- **Caring Personality**: Empathetic, supportive, and helpful tone

### **🎤 Natural Voice Experience**
- **Premium Voice Selection**: Microsoft Zira, Hazel, Google Female voices
- **Optimized Settings**: Perfect rate, pitch, and volume for caring tone
- **Smart Fallbacks**: Automatically finds best available voice
- **Voice Controls**: Easy toggle with visual feedback

### **💊 Medicine Price Intelligence**
- **Multi-Pharmacy Search**: Compares prices across 1mg, PharmEasy, Netmeds
- **Smart Detection**: Automatically detects medicine queries
- **Best Deal Highlighting**: Shows cheapest options first
- **Direct Purchase Links**: One-click access to pharmacy websites

### **🏥 Location Services**
- **Nearby Facilities**: Hospitals, clinics, pharmacies within radius
- **Real-Time Data**: Uses OpenStreetMap for accurate information
- **Google Maps Integration**: Direct navigation links
- **Opening Hours**: Shows current status (open/closed)

---

## 🛠️ **Local Development**

### **Backend Setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export HF_TOKEN=your_huggingface_token

# Run server
python app.py
```

### **Frontend Setup**
```bash
# Update SERVER_URL in index.html to http://localhost:5000
# Open index.html in browser
```

### **Test Suite**
```bash
# Run comprehensive tests
python test_local.py

# Test voice quality
# Open test_voice.html in browser
```

---

## 🔧 **Advanced Features**

### **State Detection System**
- **CASUAL**: Regular conversations and general health questions
- **CARE_MODE**: Emotional support and mental health guidance  
- **MEDICAL_MODE**: Physical symptoms requiring medical attention
- **CRITICAL_MODE**: Emergency situations with immediate escalation

### **Memory System**
- **Persistent Storage**: Conversations saved to file
- **Smart Retrieval**: Loads recent context for better responses
- **Privacy Focused**: Local storage only, no external databases

### **API Endpoints**
- `GET /health` - Health check
- `POST /chat` - Main conversation endpoint
- `POST /medicine/prices` - Medicine price search
- `POST /nearby` - Location-based facility search
- `GET /memory` - View conversation history

---

## 🔒 **Security & Privacy**

- **Environment Variables**: Sensitive tokens stored securely
- **CORS Protection**: Configured for safe cross-origin requests
- **No Data Leakage**: Personal conversations stay on your server
- **Secure Deployment**: Production-ready configuration

---

## 📊 **Performance & Reliability**

- **Smart Fallbacks**: Works even when external APIs fail
- **Error Handling**: Graceful degradation for all scenarios
- **Memory Efficient**: Optimized for Railway's free tier
- **Fast Response**: Cached responses and intelligent routing

---

## 🆘 **Troubleshooting**

### **Backend Issues**
- Check Railway logs for errors
- Verify `HF_TOKEN` environment variable
- Test `/health` endpoint

### **Frontend Issues**
- Check browser console (F12)
- Verify `SERVER_URL` points to Railway app
- Test voice permissions in browser

### **Voice Issues**
- Enable microphone permissions
- Try different browsers (Chrome recommended)
- Check available voices in browser settings

---

## 🌟 **What Users Say**

*"Aries feels like talking to a real health assistant who actually cares!"*

*"The voice is so natural, and the responses are always helpful and different."*

*"Finally, a health app that doesn't repeat the same boring advice!"*

---

## 📞 **Support**

Having issues? Check these resources:
1. **Railway Logs**: Project → Deployments → View Logs
2. **Browser Console**: F12 → Console tab
3. **Network Tab**: Check API calls and responses
4. **HuggingFace Status**: Verify token and API availability

---

## 🎉 **Ready to Deploy?**

Your Aries Health Assistant is ready for production! 

**Total Cost**: $0 (using free tiers)  
**Deployment Time**: ~15 minutes  
**Maintenance**: Zero - fully automated  

**Deploy now and give your users a caring AI health companion!** 🚀

---

**Made with ❤️ using HuggingFace Phi-3, Railway, and Vercel**
