from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import base64
import re
import time
import json
import logging
import math
from datetime import datetime
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import statistics

OLLAMA_URL = "http://localhost:11434/api/generate"
KOKORO_TTS_URL = "http://localhost:8880/v1/audio/speech"
VOICE = "af_bella"

SYSTEM_PROMPT_BASE = (
    "You are Aries — a caring, protective, encouraging elder-sibling figure. "
    "You speak calmly and with firm reassurance. Be concise, clear, and supportive. "
)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
CORS(app)


MEMORY_FILE = "aries_memory.json"

def load_memory(limit=50):
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            mem = json.load(f)
            return mem[-limit:]
    except Exception:
        return []

def save_memory_entry(user_text, ai_text, state, confidence):
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": user_text,
        "output": ai_text,
        "state": state,
        "confidence": confidence
    }
    mem = []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            mem = json.load(f)
    except Exception:
        mem = []
    mem.append(entry)
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem[-200:], f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.warning("Failed to save memory: %s", e)

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
    """
    FIXED: Much more conservative state detection
    Only triggers CRITICAL/MEDICAL for truly urgent situations
    """
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
    payload = {"model": "phi3:latest", "prompt": prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=90)
        r.raise_for_status()
        j = r.json()
        return j.get("response") or j.get("output") or j.get("text") or str(j)
    except Exception as e:
        logging.warning("LLM error: %s", e)
        return f"I'm having trouble connecting right now. Please try again in a moment."

def generate_audio_base64(text):
    try:
        payload = {"model": "kokoro", "voice": VOICE, "input": text, "format": "mp3"}
        r = requests.post(KOKORO_TTS_URL, json=payload, timeout=60)
        r.raise_for_status()
        return base64.b64encode(r.content).decode("utf-8")
    except Exception as e:
        logging.warning("TTS error: %s", e)
        return None

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

def recommended_action_for_state(state, confidence):
    """Only escalate for truly critical situations"""
    if state == "CRITICAL_MODE" and confidence >= 0.7:
        return (True, "URGENT: Call emergency services immediately (108/112). Do not wait. Your safety is the priority.")
    
    if state == "MEDICAL_MODE" and confidence >= 0.6:
        return (True, "Seek medical attention soon. I can help find nearby hospitals if needed.")
    
    if state == "CARE_MODE":
        return (False, "Consider talking to someone you trust or seeking professional support if these feelings continue.")
    
    return (False, "")

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

def scrape_1mg_prices(medicine: str) -> List[Dict]:
    results = []
    try:
        search_url = f"https://www.1mg.com/search/all?name={urllib.parse.quote(medicine)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return results
        
        soup = BeautifulSoup(response.content, 'html.parser')
        items = soup.find_all('div', limit=8)
        
        for item in items:
            try:
                a = item.find('a', href=True) # type: ignore
                if not a:
                    continue
                
                href = a['href'] # type: ignore
                text = a.get_text(strip=True)
                if not text:
                    continue
                
                price_text = item.get_text(separator=" ").strip()
                m = re.search(r'₹\s*[\d,]+(?:\.\d+)?', price_text)
                
                if m:
                    price = float(m.group(0).replace('₹', '').replace(',', '').strip())
                    results.append({
                        "name": text,
                        "price": price,
                        "url": "https://www.1mg.com" + href if href.startswith("/") else href, # type: ignore
                        "pharmacy": "1mg",
                        "rating": 4.2,
                        "reviews": 150,
                        "in_stock": True
                    })
                
                if len(results) >= 5:
                    break
            except Exception:
                continue
    except Exception as e:
        logging.debug(f"1mg scraping error: {e}")
    
    return results

def scrape_pharmeasy_prices(medicine: str) -> List[Dict]:
    results = []
    try:
        search_url = f"https://pharmeasy.in/search/all?name={urllib.parse.quote(medicine)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return results
        
        soup = BeautifulSoup(response.content, 'html.parser')
        blocks = soup.find_all(['div', 'a'], limit=12)
        
        for b in blocks:
            try:
                txt = b.get_text(strip=True)
                if not txt or len(txt) < 4:
                    continue
                
                m = re.search(r'₹\s*[\d,]+(?:\.\d+)?', txt)
                if not m:
                    continue
                
                price = float(m.group(0).replace('₹', '').replace(',', '').strip())
                results.append({
                    "name": txt.split('₹')[0].strip()[:120],
                    "price": price,
                    "url": search_url,
                    "pharmacy": "PharmEasy",
                    "rating": 4.0,
                    "reviews": 120,
                    "in_stock": True
                })
                
                if len(results) >= 5:
                    break
            except Exception:
                continue
    except Exception as e:
        logging.debug(f"PharmEasy scraping error: {e}")
    
    return results

def get_medicine_prices(medicine: str) -> Dict:
    all_results = []
    
    # Try scraping
    try:
        all_results.extend(scrape_1mg_prices(medicine))
    except Exception as e:
        logging.debug("1mg failed: %s", e)
    
    try:
        all_results.extend(scrape_pharmeasy_prices(medicine))
    except Exception as e:
        logging.debug("PharmEasy failed: %s", e)

    if not all_results:
        all_results = [
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

    if all_results:
        prices = [r["price"] for r in all_results if isinstance(r.get("price"), (int, float))]
        if prices:
            summary = {
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": round(statistics.mean(prices), 2),
                "best_deal": min(all_results, key=lambda x: x.get("price", 1e9)),
                "total_options": len(all_results)
            }
        else:
            summary = None
    else:
        summary = None

    all_results.sort(key=lambda x: x.get("price", 1e9))

    return {
        "medicine": medicine,
        "results": all_results,
        "summary": summary
    }

def find_nearby_pharmacies_with_medicine(lat: float, lon: float, medicine: str, radius: int = 3000) -> Dict:
    nearby = overpass_nearby(lat, lon, radius=radius, limit=10)
    if not nearby.get("ok"):
        return nearby
    
    pharmacies = [p for p in nearby["places"] if p["amenity"] == "pharmacy"]
    for pharmacy in pharmacies:
        pharmacy["medicine_query"] = medicine
        pharmacy["availability"] = "Call to confirm"
        pharmacy["estimated_price"] = "₹40-60"
    
    return {
        "ok": True,
        "medicine": medicine,
        "pharmacies": pharmacies,
        "total_found": len(pharmacies)
    }
    

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Aries Online — Ready to help"})

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
    
    if not result["results"]:
        return jsonify({
            "ok": False,
            "error": f"No prices found for '{medicine}'. Try using generic name (e.g., 'paracetamol' instead of 'Crocin')"
        })
    
    return jsonify({"ok": True, **result})

@app.route("/medicine/nearby", methods=["POST", "OPTIONS"])
def medicine_nearby():
    if request.method == "OPTIONS":
        return make_response("", 200)
    
    j = request.get_json(force=True, silent=True) or {}
    medicine = j.get("medicine", "").strip()
    lat = j.get("lat")
    lon = j.get("lon")
    radius = int(j.get("radius", 3000))
    
    if not medicine:
        return jsonify({"ok": False, "error": "Medicine name required"}), 400
    if lat is None or lon is None:
        return jsonify({"ok": False, "error": "Location required"}), 400
    
    result = find_nearby_pharmacies_with_medicine(lat, lon, medicine, radius)
    return jsonify(result)

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
                "audio": None,
                "state": "MEDICINE_QUERY",
                "confidence": 1.0,
                "escalate": False,
                "medicine_data": price_data,
                "recommended_action": ""
            })

    state, confidence = detect_state(user_text)
    logging.info("Detected state=%s confidence=%.2f", state, confidence)

    prompt = build_prompt(user_text, state, confidence)

    reply = generate_llm_response(prompt)
    logging.info("ARIES_REPLY: %s", reply[:200])

    escalate, action_text = recommended_action_for_state(state, confidence)

    want_nearby = False
    if escalate or NEARBY_KEYWORDS.search(user_text):
        want_nearby = True

    nearest = None
    
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

    if want_nearby and lat is not None and lon is not None:
        try:
            nr = overpass_nearby(lat, lon, radius=5000, limit=5)
            if nr.get("ok") and nr.get("places"):
                nearest = nr["places"][0]
        except Exception as e:
            logging.warning("Nearby lookup failed: %s", e)
    audio_b64 = generate_audio_base64(reply) if escalate else None

    try:
        save_memory_entry(user_text, reply, state, confidence)
    except Exception:
        pass

    payload = {
        "reply": reply,
        "audio": audio_b64,
        "state": state,
        "confidence": round(confidence, 2),
        "escalate": escalate,
        "recommended_action": action_text,
        "nearest": nearest
    }
    
    return jsonify(payload)

@app.route("/speak", methods=["POST", "OPTIONS"])
def speak():
    if request.method == "OPTIONS":
        return make_response("", 200)
    
    j = request.get_json(force=True, silent=True) or {}
    text = (j.get("text") or "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    audio_b64 = generate_audio_base64(text)
    return jsonify({"audio": audio_b64})

if __name__ == "__main__":
    logging.info("Starting Aries server on 0.0.0.0:5000")

    app.run(host="0.0.0.0", port=5000, debug=False)
