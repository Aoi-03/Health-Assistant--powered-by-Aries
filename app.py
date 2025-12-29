# Deployment-ready Aries server for Railway
import os
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import base64
import re
import time
import json
import logging
import math
import random
from datetime import datetime
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import statistics
from huggingface_hub import InferenceClient

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, skip (fine for production)
    pass

# ----- Configuration -----
# Use environment variables for deployment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize HuggingFace client
hf_client = InferenceClient(
    model="microsoft/Phi-3-mini-4k-instruct",
    token=HF_TOKEN
)

SYSTEM_PROMPT_BASE = (
    "You are Aries — a caring, protective, encouraging elder-sibling figure. "
    "You speak calmly and with firm reassurance. Be concise, clear, and supportive. "
)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ---- logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
CORS(app)

# ---- Memory with file persistence for Railway ----
import os
import tempfile

# Use Railway's persistent volume or temp directory
MEMORY_DIR = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', tempfile.gettempdir())
MEMORY_FILE = os.path.join(MEMORY_DIR, 'aries_memory.json')

def load_memory(limit=50):
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                mem = json.load(f)
                return mem[-limit:] if mem else []
    except Exception as e:
        logging.warning(f"Failed to load memory: {e}")
    return []

def save_memory_entry(user_text, ai_text, state, confidence):
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": user_text,
        "output": ai_text,
        "state": state,
        "confidence": confidence
    }
    
    # Load existing memory
    memory_store = load_memory(200)  # Keep last 200 entries
    memory_store.append(entry)
    
    # Save back to file
    try:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_store, f, indent=2, ensure_ascii=False)
        logging.info(f"Memory saved: {len(memory_store)} entries")
    except Exception as e:
        logging.warning(f"Failed to save memory: {e}")

# ---- State detection patterns ----
CRITICAL_PATTERNS = [
    r"\b(?:i want to|going to|will|planning to)\s+(?:kill myself|end my life|commit suicide|take my life)\b",
    r"\b(?:suicide|suicidal thoughts|want to die)\b",
    r"\b(?:hanging|overdose|jump off|shoot myself)\b"
]

MEDICAL_PATTERNS = [
    r"\b(?:severe|unbearable|extreme|intense)\s+(?:pain|bleeding|vomiting)\b",
    r"\b(?:chest pain|heart attack|stroke|seizure|difficulty breathing|can't breathe)\b",
    r"\b(?:broken bone|fracture|deep cut|severe burn)\b",
    r"\b(?:high fever|temperature over|fever\s+\d{3})\b",
    r"\b(?:uncontrolled bleeding|heavy bleeding|blood in)\b"
]

CARE_PATTERNS = [
    r"\b(?:feeling|i feel|i am|i'm)\s+(?:depressed|anxious|sad|lonely|stressed|overwhelmed)\b",
    r"\b(?:depression|anxiety|panic attack|mental health)\b",
    r"\b(?:can't cope|struggling|having a hard time)\b"
]

NEARBY_KEYWORDS = re.compile(
    r"\b(?:near|nearest|nearby|closest|where (?:is|are).+hospital|find.+hospital|hospital near|clinic near|pharmacy near|medical store|chemist)\b",
    re.IGNORECASE
)

MEDICINE_KEYWORDS = re.compile(
    r"\b(?:medicine|tablet|capsule|syrup|injection|drug|medication|price|cost|buy|"
    r"paracetamol|dolo|crocin|aspirin|ibuprofen|amoxicillin|azithromycin|"
    r"metformin|vitamin|supplement|ointment|cream)\b",
    re.IGNORECASE
)

def detect_state(text):
    """Conservative state detection"""
    t = text.lower()
    
    critical_score = 0
    medical_score = 0
    care_score = 0
    
    for pattern in CRITICAL_PATTERNS:
        if re.search(pattern, t):
            critical_score += 2.0
    
    for pattern in MEDICAL_PATTERNS:
        if re.search(pattern, t):
            medical_score += 1.5
    
    for pattern in CARE_PATTERNS:
        if re.search(pattern, t):
            care_score += 1.0
    
    simple_symptoms = ['headache', 'fever', 'cold', 'cough', 'tired', 'stomach ache']
    has_simple_symptom = any(symptom in t for symptom in simple_symptoms)
    has_severe_modifier = any(word in t for word in ['severe', 'unbearable', 'extreme', 'intense', 'can\'t', 'unable'])
    
    if has_simple_symptom and not has_severe_modifier:
        medical_score = max(0, medical_score - 1.0)
    
    if critical_score >= 2.0:
        return "CRITICAL_MODE", min(0.95, 0.7 + (critical_score * 0.1))
    
    if medical_score >= 1.5:
        return "MEDICAL_MODE", min(0.90, 0.6 + (medical_score * 0.1))
    
    if care_score >= 1.0:
        return "CARE_MODE", min(0.85, 0.5 + (care_score * 0.1))
    
    return "CASUAL", 0.3

def build_prompt(user_message, state, confidence):
    instructions = ""
    if state == "CASUAL":
        instructions = (
            "Respond in a friendly, concise way. Keep tone warm and casual. No medical diagnosis. "
            "For simple symptoms like headaches or minor issues, provide general wellness advice "
            "(rest, hydration, over-the-counter remedies) but always suggest seeing a doctor if symptoms persist or worsen."
        )
    elif state == "CARE_MODE":
        instructions = (
            "The user is showing emotional distress but not immediate danger. Respond with calm empathy. "
            "Use supportive language, encourage self-care (breathing, taking breaks, talking to someone), "
            "and suggest professional support if feelings persist. Keep it warm and non-clinical."
        )
    elif state == "MEDICAL_MODE":
        instructions = (
            "The user reports significant physical symptoms. Ask focused questions about severity, duration, and red flags. "
            "Provide conservative advice (rest, fluids) but STRONGLY recommend seeing a doctor soon. "
            "Offer to find nearby hospitals if appropriate."
        )
    elif state == "CRITICAL_MODE":
        instructions = (
            "HIGH-RISK situation indicating possible self-harm or imminent danger. Use calm, grounding language. "
            "IMMEDIATELY encourage calling emergency services (108/112). Ask if they are safe now and if anyone is with them. "
            "Keep reply short, stabilizing, and focused on immediate safety. Do not over-explain."
        )
    
    prompt = (
        SYSTEM_PROMPT_BASE
        + f"\nCONTEXT: User state={state} (confidence={confidence:.2f}).\n"
        + f"INSTRUCTIONS: {instructions}\n\n"
        + f"User: {user_message}\nAries:"
    )
    return prompt

def generate_llm_response(prompt):
    """Use HuggingFace API with multiple fallback strategies"""
    try:
        # Try multiple HuggingFace endpoints and models
        models_to_try = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small", 
            "facebook/blenderbot-400M-distill",
            "microsoft/Phi-3-mini-4k-instruct"
        ]
        
        for model in models_to_try:
            try:
                # Use the serverless inference API
                API_URL = f"https://api-inference.huggingface.co/models/{model}"
                headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                
                # Extract just the user message from the prompt
                user_message = prompt.split("User: ")[-1].split("\nAries:")[0].strip()
                
                payload = {
                    "inputs": user_message,
                    "parameters": {
                        "max_length": 200,
                        "temperature": 0.8,
                        "do_sample": True,
                        "pad_token_id": 50256
                    }
                }
                
                response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "").strip()
                        # Clean up the response
                        if generated_text and len(generated_text) > 10:
                            # Remove the input from the output if it's repeated
                            if user_message in generated_text:
                                generated_text = generated_text.replace(user_message, "").strip()
                            return generated_text[:300]  # Limit length
                    elif isinstance(result, dict) and "generated_text" in result:
                        generated_text = result["generated_text"].strip()
                        if generated_text and len(generated_text) > 10:
                            return generated_text[:300]
                
                logging.info(f"Model {model} failed: {response.status_code}")
                
            except Exception as e:
                logging.info(f"Model {model} error: {e}")
                continue
        
        # If all models fail, use contextual responses
        return generate_contextual_response(prompt)
        
    except Exception as e:
        logging.warning("All HuggingFace attempts failed: %s", e)
        return generate_contextual_response(prompt)

def generate_contextual_response(prompt):
    """Generate varied contextual responses based on conversation history and user input"""
    user_input = prompt.split("User: ")[-1].split("\nAries:")[0].strip().lower()
    
    # Load recent memory to avoid repetition
    recent_memory = load_memory(5)
    recent_responses = [entry.get("output", "") for entry in recent_memory]
    
    # Health symptom responses
    if any(word in user_input for word in ['headache', 'head hurt', 'migraine']):
        responses = [
            "Headaches can be caused by stress, dehydration, or lack of sleep. Try drinking water, resting in a dark room, and taking a break from screens. If it persists, consider seeing a doctor.",
            "For headaches, I'd suggest staying hydrated, getting some rest, and maybe trying a cold compress on your forehead. Over-the-counter pain relievers can help too, but follow the dosage instructions.",
            "Headaches are often related to tension or dehydration. Have you been drinking enough water today? Sometimes a short walk in fresh air can help too."
        ]
        return get_varied_response(responses, recent_responses)
    
    if any(word in user_input for word in ['fever', 'temperature', 'hot', 'burning up']):
        responses = [
            "Fever is your body's way of fighting infection. Stay hydrated, rest, and monitor your temperature. If it goes above 103°F (39.4°C) or persists for more than 3 days, see a doctor.",
            "For fever, focus on staying cool and hydrated. Light clothing, room temperature fluids, and rest are key. Keep track of your temperature and watch for other symptoms.",
            "A fever means your immune system is working. Drink plenty of fluids, get rest, and take your temperature regularly. Seek medical care if you feel very unwell or the fever is very high."
        ]
        return get_varied_response(responses, recent_responses)
    
    if any(word in user_input for word in ['cold', 'cough', 'sneezing', 'runny nose']):
        responses = [
            "Cold symptoms are usually manageable at home. Rest, warm fluids like tea or soup, and honey can soothe a cough. If symptoms worsen or last more than 10 days, consider seeing a doctor.",
            "For cold symptoms, I recommend plenty of rest, staying hydrated, and maybe some warm salt water gargles for a sore throat. Most colds resolve on their own in 7-10 days.",
            "Sounds like you might have a cold. Warm liquids, rest, and time are your best remedies. Zinc lozenges and vitamin C might help too. Take care of yourself!"
        ]
        return get_varied_response(responses, recent_responses)
    
    # Medicine and pharmacy queries
    if any(word in user_input for word in ['medicine', 'tablet', 'price', 'pharmacy', 'drug']):
        responses = [
            "I can help you compare medicine prices across different pharmacies. Just tell me the specific medicine name you're looking for, and I'll find the best deals.",
            "Looking for medicine prices? I can search across multiple pharmacies to find you the best prices. What medication do you need?",
            "I'd be happy to help you find competitive medicine prices. Which medication are you interested in? I can compare prices from different pharmacies for you."
        ]
        return get_varied_response(responses, recent_responses)
    
    # Emergency and urgent care
    if any(word in user_input for word in ['emergency', 'urgent', 'hospital', 'serious', 'severe']):
        responses = [
            "For urgent medical situations, please call emergency services (108/112) immediately. I can also help you find the nearest hospital if you share your location.",
            "If this is a medical emergency, don't hesitate to call 108 or 112 right away. Your safety is the priority. I can help locate nearby hospitals if needed.",
            "Urgent medical issues require immediate professional care. Please contact emergency services (108/112) or go to the nearest emergency room. I'm here to help find nearby facilities if you need."
        ]
        return get_varied_response(responses, recent_responses)
    
    # Mental health and emotional support
    if any(word in user_input for word in ['anxious', 'anxiety', 'depressed', 'sad', 'stressed', 'worried', 'overwhelmed']):
        responses = [
            "I hear that you're going through a difficult time. It's brave of you to reach out. Consider talking to someone you trust - a friend, family member, or mental health professional. You don't have to face this alone.",
            "Mental health is just as important as physical health. If you're feeling overwhelmed, try some deep breathing exercises, take a short walk, or call someone who cares about you. Professional support is also available if you need it.",
            "It sounds like you're dealing with some tough emotions right now. That's completely normal, and it's okay to not be okay sometimes. Reaching out for support, whether from loved ones or professionals, is a sign of strength."
        ]
        return get_varied_response(responses, recent_responses)
    
    # Greetings and general conversation
    if any(word in user_input for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
        responses = [
            "Hello! I'm Aries, your health assistant. I'm here to help with health questions, medicine prices, and finding medical facilities. What's on your mind today?",
            "Hi there! I'm Aries. I can help you with health advice, compare medicine prices, or find nearby hospitals and pharmacies. How can I assist you?",
            "Hey! Good to see you. I'm Aries, and I'm here to support you with health-related questions and information. What would you like to know?",
            "Hello! I'm Aries, your caring health companion. Whether you need health advice, medicine price comparisons, or help finding medical facilities, I'm here for you. What can I help with?"
        ]
        return get_varied_response(responses, recent_responses)
    
    # Default varied responses
    default_responses = [
        "I'm here to help with your health questions. You can ask me about symptoms, medicine prices, or finding nearby medical facilities. What's concerning you?",
        "As your health assistant, I can provide advice on common health issues, help you compare medicine prices, or locate nearby hospitals and pharmacies. How can I support you today?",
        "I'm Aries, and I care about your wellbeing. Whether you have health questions, need medicine price information, or want to find medical facilities nearby, I'm here to help. What do you need?",
        "I'm ready to assist you with health-related questions and information. I can discuss symptoms, compare medication prices, or help you find medical services. What would you like to know?"
    ]
    return get_varied_response(default_responses, recent_responses)

def get_varied_response(possible_responses, recent_responses):
    """Select a response that hasn't been used recently"""
    # Filter out responses that were used recently
    available_responses = [resp for resp in possible_responses 
                          if not any(resp[:50] in recent[:50] for recent in recent_responses)]
    
    # If all responses were used recently, use any response
    if not available_responses:
        available_responses = possible_responses
    
    # Return a random response from available ones
    import random
    return random.choice(available_responses)

# ---- Medicine price functions (same as before) ----
def extract_medicine_name(text: str) -> Optional[str]:
    patterns = [
        r"(?:price|cost|buy|find|get|need|looking for)\s+(?:of\s+)?([a-zA-Z0-9\s-]+?)(?:\s+(?:tablet|medicine|capsule|syrup|mg|near me)|\?|$)",
        r"(?:medicine|tablet|drug)\s+(?:called\s+|named\s+)?([a-zA-Z0-9\s-]+?)(?:\s|$|\?)",
        r"\b([a-zA-Z]{3,}(?:\s+\d+)?)\s+(?:price|cost|tablet|medicine)\b"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            med_name = match.group(1).strip()
            if len(med_name) > 2 and med_name.lower() not in ['the', 'for', 'and', 'this', 'that']:
                return med_name
    
    known_meds = ['paracetamol', 'dolo', 'crocin', 'azithromycin', 'ibuprofen', 'aspirin', 'amoxicillin']
    for med in known_meds:
        if re.search(rf"\b{med}\b", text, re.IGNORECASE):
            return med
    
    return None

def get_medicine_prices(medicine: str) -> Dict:
    """Simplified medicine price lookup with demo data"""
    # For deployment, using demo data since web scraping might be blocked
    demo_results = [
        {
            "name": f"{medicine.title()} 500mg Strip of 10 Tablets",
            "price": 45.50,
            "url": f"https://www.1mg.com/drugs/{medicine.lower().replace(' ', '-')}",
            "pharmacy": "1mg",
            "rating": 4.3,
            "reviews": 234,
            "in_stock": True
        },
        {
            "name": f"{medicine.title()} 500mg Bottle of 15 Tablets",
            "price": 52.00,
            "url": f"https://pharmeasy.in/search/all?name={medicine}",
            "pharmacy": "PharmEasy",
            "rating": 4.1,
            "reviews": 156,
            "in_stock": True
        },
        {
            "name": f"{medicine.title()} 650mg Strip of 15 Tablets",
            "price": 38.75,
            "url": f"https://www.netmeds.com/catalogsearch/result/{medicine}",
            "pharmacy": "Netmeds",
            "rating": 4.5,
            "reviews": 412,
            "in_stock": True
        }
    ]

    prices = [r["price"] for r in demo_results]
    summary = {
        "min_price": min(prices),
        "max_price": max(prices),
        "avg_price": round(statistics.mean(prices), 2),
        "best_deal": min(demo_results, key=lambda x: x.get("price", 1e9)),
        "total_options": len(demo_results)
    }

    demo_results.sort(key=lambda x: x.get("price", 1e9))

    return {
        "medicine": medicine,
        "results": demo_results,
        "summary": summary
    }

def recommended_action_for_state(state, confidence):
    """Only escalate for truly critical situations"""
    if state == "CRITICAL_MODE" and confidence >= 0.7:
        return (True, "URGENT: Call emergency services immediately (108/112). Do not wait. Your safety is the priority.")
    
    if state == "MEDICAL_MODE" and confidence >= 0.6:
        return (True, "Seek medical attention soon. I can help find nearby hospitals if needed.")
    
    if state == "CARE_MODE":
        return (False, "Consider talking to someone you trust or seeking professional support if these feelings continue.")
    
    return (False, "")

# ---- Overpass / Nearby helpers ----
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def parse_opening_hours_simple(oh):
    if not oh or not isinstance(oh, str):
        return {"raw": None, "status": "unknown", "note": None}
    txt = oh.strip().lower()
    if "24/7" in txt or "24h" in txt or "24 hr" in txt:
        return {"raw": oh, "status": "open", "note": "24/7"}
    
    today = datetime.utcnow().weekday()
    day_short = ["mo", "tu", "we", "th", "fr", "sa", "su"][today]
    parts = [p.strip() for p in re.split(r"[;\/]", txt) if p.strip()]
    
    for p in parts:
        if re.search(rf"\b({day_short}|{day_short}[\w-]*)\b", p):
            m = re.search(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", p)
            if m:
                try:
                    now = datetime.utcnow().time()
                    start = datetime.strptime(m.group(1), "%H:%M").time()
                    end = datetime.strptime(m.group(2), "%H:%M").time()
                    if start <= now <= end:
                        return {"raw": oh, "status": "open", "note": f"Open today {m.group(1)}-{m.group(2)}"}
                    else:
                        return {"raw": oh, "status": "closed", "note": f"Opens at {m.group(1)}"}
                except Exception:
                    pass
    
    return {"raw": oh, "status": "unknown", "note": "Check opening hours"}

def overpass_nearby(lat, lon, radius=2000, limit=5):
    q = f"""
    [out:json][timeout:25];
    (
      node["amenity"~"hospital|clinic|pharmacy"](around:{radius},{lat},{lon});
      way["amenity"~"hospital|clinic|pharmacy"](around:{radius},{lat},{lon});
      relation["amenity"~"hospital|clinic|pharmacy"](around:{radius},{lat},{lon});
    );
    out center tags geom;
    """
    try:
        r = requests.post(OVERPASS_URL, data=q.encode("utf-8"), timeout=60)
        r.raise_for_status()
        j = r.json()
        elements = j.get("elements", [])
    except Exception as e:
        logging.warning("Overpass error: %s", e)
        return {"ok": False, "error": str(e), "places": []}

    places = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        if el.get("type") == "node":
            plat = el.get("lat")
            plon = el.get("lon")
        else:
            center = el.get("center") or el.get("bounds") or {}
            plat = center.get("lat")
            plon = center.get("lon")
        
        if plat is None or plon is None:
            continue
        
        name = tags.get("name") or tags.get("operator") or "Unknown"
        amen = tags.get("amenity") or "unknown"
        phone = tags.get("phone") or tags.get("contact:phone")
        website = tags.get("website") or tags.get("contact:website")
        opening = tags.get("opening_hours")
        dist = haversine(lat, lon, plat, plon)
        oh_info = parse_opening_hours_simple(opening)
        google_maps = f"https://www.google.com/maps/dir/?api=1&destination={plat},{plon}"
        
        place = {
            "osm_id": el.get("id"),
            "name": name,
            "amenity": amen,
            "lat": plat,
            "lon": plon,
            "distance_m": int(dist),
            "phone": phone,
            "website": website,
            "opening_hours_raw": opening,
            "opening_status": oh_info["status"],
            "opening_note": oh_info.get("note"),
            "google_maps": google_maps
        }
        places.append(place)

    places = sorted(places, key=lambda x: x["distance_m"])
    return {"ok": True, "places": places[:limit]}

# ---- API Routes ----
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Aries Online — Ready to help", "version": "1.0"})

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route("/medicine/prices", methods=["POST", "OPTIONS"])
def medicine_prices():
    if request.method == "OPTIONS":
        return make_response("", 200)
    
    j = request.get_json(force=True, silent=True) or {}
    medicine = j.get("medicine", "").strip()
    
    if not medicine:
        return jsonify({"ok": False, "error": "Medicine name required"}), 400
    
    logging.info(f"Fetching prices for: {medicine}")
    result = get_medicine_prices(medicine)
    
    return jsonify({"ok": True, **result})

@app.route("/memory", methods=["GET"])
def get_memory():
    """Get recent chat memory (for debugging)"""
    try:
        memory = load_memory(20)  # Last 20 conversations
        return jsonify({
            "ok": True,
            "total_entries": len(memory),
            "recent_entries": memory
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/nearby", methods=["POST", "OPTIONS"])
def nearby_route():
    if request.method == "OPTIONS":
        return make_response("", 200)
    
    j = request.get_json(force=True, silent=True) or {}
    lat = j.get("lat")
    lon = j.get("lon")
    
    if lat is None or lon is None:
        return make_response(jsonify({"ok": False, "error": "lat and lon required"}), 400)
    
    radius = int(j.get("radius", 2000))
    limit = int(j.get("limit", 5))
    res = overpass_nearby(lat, lon, radius=radius, limit=limit)
    
    return jsonify(res)

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return make_response("", 200)
    
    j = request.get_json(force=True, silent=True) or {}
    user_text = (j.get("message") or "").strip()
    lat = j.get("lat")
    lon = j.get("lon")
    
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    logging.info("USER: %s", user_text)

    # Check for medicine query first
    medicine_name = None
    if MEDICINE_KEYWORDS.search(user_text):
        medicine_name = extract_medicine_name(user_text)
    
    if medicine_name:
        logging.info("Detected medicine query: %s", medicine_name)
        price_data = get_medicine_prices(medicine_name)
        
        if price_data["results"]:
            summary = price_data["summary"]
            best = summary["best_deal"] if summary else price_data["results"][0]
            
            reply = f"I found {summary['total_options'] if summary else len(price_data['results'])} options for {medicine_name}.\n\n"
            reply += f"💰 Best Price: ₹{best['price']} at {best['pharmacy']}\n"
            if summary:
                reply += f"📊 Price Range: ₹{summary['min_price']} - ₹{summary['max_price']}\n"
                reply += f"📈 Average: ₹{summary['avg_price']}\n\n"
            reply += "Check the Medicine Prices tab for full comparison!"
            
            return jsonify({
                "reply": reply,
                "audio": None,  # No TTS in deployment version
                "state": "MEDICINE_QUERY",
                "confidence": 1.0,
                "escalate": False,
                "medicine_data": price_data,
                "recommended_action": ""
            })

    # Detect state
    state, confidence = detect_state(user_text)
    logging.info("Detected state=%s confidence=%.2f", state, confidence)

    # Build prompt and get response
    prompt = build_prompt(user_text, state, confidence)
    reply = generate_llm_response(prompt)
    logging.info("ARIES_REPLY: %s", reply[:200])

    # Determine escalation
    escalate, action_text = recommended_action_for_state(state, confidence)

    # Check if nearby lookup needed
    want_nearby = False
    if escalate or NEARBY_KEYWORDS.search(user_text):
        want_nearby = True

    nearest = None
    
    # Only ask for location if we actually need it
    if want_nearby and (lat is None or lon is None):
        return jsonify({
            "reply": reply,
            "audio": None,
            "state": state,
            "confidence": round(confidence, 2),
            "escalate": escalate,
            "recommended_action": action_text,
            "nearest": None,
            "need_location": True
        })

    # Get nearby places if we have location and it's needed
    if want_nearby and lat is not None and lon is not None:
        try:
            nr = overpass_nearby(lat, lon, radius=5000, limit=5)
            if nr.get("ok") and nr.get("places"):
                nearest = nr["places"][0]
        except Exception as e:
            logging.warning("Nearby lookup failed: %s", e)

    # Save memory
    try:
        save_memory_entry(user_text, reply, state, confidence)
    except Exception:
        pass

    payload = {
        "reply": reply,
        "audio": None,  # No TTS in deployment version
        "state": state,
        "confidence": round(confidence, 2),
        "escalate": escalate,
        "recommended_action": action_text,
        "nearest": nearest
    }
    
    return jsonify(payload)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)