#  Aries Health Assistant

**Aries** is a local AI-powered health assistant built with **Flask**, **Ollama (Phi-3)**, and **Kokoro TTS**.  
It provides natural, emotional health conversations, nearby hospital lookup, and real-time medicine price comparison — all while keeping your data local.

---

##  Prerequisites

Before running Aries, make sure you have the following installed and set up:

###  1️⃣ **Docker**
Used to run **Kokoro TTS** (text-to-speech engine).  
- Download Docker from: [https://www.docker.com/get-started](https://www.docker.com/get-started)  
- After installation, make sure **Docker Desktop** is running in the background.

###  2️⃣ **Ollama**
Used to run the **local LLM (Phi-3)** for natural conversation.  
- Download Ollama from: [https://ollama.com/download](https://ollama.com/download)  
- Once installed, run these commands:

```bash
ollama pull phi3:latest
ollama serve
```

---

##  Features

-  **AI Chat** – Emotional + medical context detection using a local LLM (Phi-3)
-  **Medicine Price Search** – Scrapes 1mg, PharmEasy, and NetMeds for live price comparison
-  **Nearby Hospitals/Pharmacies** – Uses Overpass API (OpenStreetMap) to find nearby facilities
-  **TTS Output** – Real-time Kokoro voice response (via Docker)
-  **Local Memory** – Stores short-term chat history in JSON
-  **Privacy First** – Runs entirely on your local machine, no external API calls for chat or TTS

---

##  System Architecture

```text
Frontend (index.html)
   ↕
Backend (Flask - ICC_logic.py)
   ↕
Ollama (phi3:latest) — Local LLM
Kokoro TTS (Docker)
Overpass API — Map Data
1mg / PharmEasy — Web Scraping for Medicine Prices
