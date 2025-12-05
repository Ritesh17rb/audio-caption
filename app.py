import streamlit as st
import os
import librosa
import numpy as np
import warnings
import requests
import json

warnings.filterwarnings("ignore")

# Optional heavy models - import only if installed
try:
    import whisper
except Exception:
    whisper = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

# --------------------
# Basic paths & config
# --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SONGS_DIR = os.path.join(BASE_DIR, "songs")

st.set_page_config(page_title="Audio Captioning", page_icon="🎵", layout="wide")

# --------------------
# LLM Configuration
# --------------------
LLM_API_URL = "https://llmfoundry.straive.com/openai/v1/chat/completions"

# --------------------
# Styling
# --------------------
st.markdown("""
<style>
:root { --accent: #6f9eff; --muted: #9fb3ff; --card-bg: rgba(255,255,255,0.03); }
body { background: #07101a; color: #e8f4ff; }
.metric-card {
    background: var(--card-bg);
    border: 1px solid rgba(255,255,255,0.04);
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    min-height: 100px;
    transition: transform 0.2s;
    cursor: help;
}
.metric-card:hover { transform: translateY(-3px); border-color: var(--accent); }
.metric-label { color: var(--muted); font-size: 13px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { color: var(--accent); font-weight:700; font-size: 24px; }
.row-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: 14px; margin-top:10px; }
.demo-card { background: rgba(255,255,255,0.02); padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); }
.tooltip-note { font-size:11px; color:#6e7f9e; margin-top: 4px; }
.recommendation-tag { display:inline-block; background: rgba(111,160,255,0.12); padding:4px 10px; border-radius:4px; margin:4px; color:#cfe1ff; font-size:12px; border: 1px solid rgba(111,160,255,0.2); }
.insight-box { background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding:20px; border-radius:12px; border:1px solid rgba(255,255,255,0.05); margin-bottom: 20px; }
.llm-response { font-family: sans-serif; font-size: 16px; line-height: 1.6; color: #d0d7e5; border-left: 3px solid #6f9eff; padding-left: 15px; }
</style>
""", unsafe_allow_html=True)

# --------------------
# Demo songs config
# --------------------
DEMO_SONGS = {
    "song-1": {"path": os.path.join(SONGS_DIR, "song-1.mp3"), "name": "Demo Song 1", "desc": "End of Beginning"},
    "song-2": {"path": os.path.join(SONGS_DIR, "song-2.mp3"), "name": "Demo Song 2", "desc": "Attention"},
    "song-3": {"path": os.path.join(SONGS_DIR, "song-3.mp3"), "name": "Demo Song 3", "desc": "Let Me Down Slowly"},
}

# --------------------
# Sidebar
# --------------------
with st.sidebar:
    st.header("🎛️ Audio Captioning")
    st.caption("Signal Processing + GenAI")
    st.markdown("---")
    LLM_TOKEN = st.text_input("LLM API Token", type="password", help="Required for the AI summary")
    st.markdown("---")
    st.info("ℹ️ Hover over cards for definitions.")

# --------------------
# Models
# --------------------
@st.cache_resource
def load_models():
    models = {}
    if whisper is not None:
        try: models['whisper'] = whisper.load_model("base")
        except: models['whisper'] = None
    else: models['whisper'] = None

    if pipeline is not None:
        try: models['genre'] = pipeline("audio-classification", model="dima806/music_genres_classification")
        except: models['genre'] = None
        try: models['instrument'] = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
        except: models['instrument'] = None
    return models

# --------------------
# Logic: Feature Extraction
# --------------------
def compute_rhythm_stability(beat_frames):
    if beat_frames is None or len(beat_frames) < 2: return 0.0
    try:
        intervals = np.diff(beat_frames).astype(float)
        mean_i = np.mean(intervals)
        std_i = np.std(intervals)
        stability = 1.0 - (std_i / (mean_i + 1e-9))
        return float(np.clip(stability, 0.0, 1.0))
    except: return 0.0

def extract_spotify_features(y, sr):
    features = {}
    
    # 1. Tempo & Beats
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    except: tempo, beat_frames = 0.0, np.array([])

    features['tempo'] = float(tempo) if not isinstance(tempo, np.ndarray) else float(np.mean(tempo))
    features['rhythm_stability'] = compute_rhythm_stability(np.array(beat_frames))

    # 2. Energy
    try:
        rms = librosa.feature.rms(y=y)[0]
        features['energy'] = float(np.clip(np.mean(rms) * 1.5, 0, 1))
    except: features['energy'] = 0.0

    # 3. Danceability
    try:
        if len(beat_frames) > 0:
            onset_env = onset_env if 'onset_env' in locals() else librosa.onset.onset_strength(y=y, sr=sr)
            beat_strength = np.mean(onset_env[beat_frames])
            features['danceability'] = float(np.clip(beat_strength / 6.0, 0, 1))
        else: features['danceability'] = 0.0
    except: features['danceability'] = 0.0

    # 4. Valence
    v = 0.5
    if features['energy'] > 0.6: v += 0.2
    if features['tempo'] > 120: v += 0.15
    if features['tempo'] < 80: v -= 0.15
    features['valence'] = float(np.clip(v, 0, 1))

    # 5. Acousticness & Instrumentalness
    try:
        spec_flat = librosa.feature.spectral_flatness(y=y)[0]
        features['acousticness'] = float(1.0 - np.clip(np.mean(spec_flat) * 3.0, 0, 1))
        
        y_h, y_p = librosa.effects.hpss(y)
        h_e, p_e = np.mean(y_h**2), np.mean(y_p**2)
        features['instrumentalness'] = float(np.clip(1.0 - (h_e/(h_e+p_e+1e-9)), 0, 1))
    except: 
        features['acousticness'] = 0.5
        features['instrumentalness'] = 0.5

    # 6. Speechiness
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['speechiness'] = float(np.clip(np.mean(zcr)*2, 0, 1))
    except: features['speechiness'] = 0.0

    return features

def analyze_audio_quality(y, sr):
    quality = {}
    try:
        rms = librosa.feature.rms(y=y)[0]
        loudness_db = librosa.amplitude_to_db(rms, ref=np.max)
        quality['dynamic_range_db'] = float(np.max(loudness_db) - np.min(loudness_db))
    except: quality['dynamic_range_db'] = 0.0

    if quality['dynamic_range_db'] < 6: quality['compression'] = "Heavy (Loud)"
    elif quality['dynamic_range_db'] < 12: quality['compression'] = "Moderate"
    else: quality['compression'] = "Natural (Dynamic)"

    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg = float(np.mean(cent))
        quality['brightness_hz'] = int(avg)
        quality['timbre'] = "Bright/Sharp" if avg > 3000 else ("Dark/Warm" if avg < 1500 else "Balanced")
    except: 
        quality['brightness_hz'] = 0
        quality['timbre'] = "Unknown"
        
    return quality

def detect_key(y, sr):
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        avg = np.mean(chroma, axis=1)
        notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        key = notes[np.argmax(avg)]
        return key
    except: return "Unknown"

def get_instruments(path, models):
    insts = []
    if not models.get('instrument'): return insts
    try:
        preds = models['instrument'](path, top_k=8)
        ignored = ['music', 'song', 'musical instrument', 'performing arts']
        for p in preds:
            if p['score'] > 0.05 and p['label'].lower() not in ignored:
                insts.append((p['label'], int(p['score']*100)))
        return insts[:5]
    except: return []

# --------------------
# LLM INTEGRATION
# --------------------
def generate_llm_insight(data, token):
    if not token:
        return "🔒 Enter LLM Token in sidebar to see AI Analysis."

    f = data['features']
    prompt = f"""
    Analyze the technical audio data below and write a 2-3 sentence summary.
    
    DATA:
    - Energy: {int(f['energy']*100)}% (Intensity)
    - Tempo: {int(f['tempo'])} BPM
    - Brightness: {data['quality']['brightness_hz']}Hz ({data['quality']['timbre']})
    - Dynamic Range: {data['quality']['dynamic_range_db']:.1f}dB
    - Key: {data['key']}
    - Genre: {data['primary_genre']}
    
    TASK:
    Briefly describe the song's character based on these inputs. Mention how the Energy, Brightness, or Tempo contributes to the overall feel. Do not list numbers, just the insight.
    """

    try:
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 100
        }
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        response = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=20)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error from LLM: {response.text}"
    except Exception as e:
        return f"LLM Connection Failed: {str(e)}"

# --------------------
# Main Analysis Logic
# --------------------
def run_analysis(path, llm_token=None):
    models = load_models()
    try: y, sr = librosa.load(path, duration=30)
    except: return None

    feats = extract_spotify_features(y, sr)
    qual = analyze_audio_quality(y, sr)
    key = detect_key(y, sr)

    try:
        g_preds = models['genre'](path, top_k=2) if models.get('genre') else []
        p_genre = g_preds[0]['label'] if g_preds else "Unknown"
        all_genres = ", ".join([f"{p['label']} ({int(p['score']*100)}%)" for p in g_preds])
    except: 
        p_genre = "Unknown"
        all_genres = ""

    insts = get_instruments(path, models)

    lyrics = ""
    try:
        if models.get('whisper'):
            tr = models['whisper'].transcribe(path)
            lyrics = tr.get('text', "").strip()
    except: pass

    results = {
        "features": feats,
        "quality": qual,
        "key": key,
        "primary_genre": p_genre,
        "all_genres": all_genres,
        "instruments": insts,
        "lyrics": lyrics
    }

    # Generate LLM Insight
    if llm_token:
        with st.spinner("🤖 Synthesizing inputs..."):
            results['llm_summary'] = generate_llm_insight(results, llm_token)
    else:
        results['llm_summary'] = None

    return results

# --------------------
# UI Display
# --------------------
def metric_card(title, value, tooltip, sub):
    return f"""
    <div class="metric-card" title="{tooltip}">
        <div class="metric-label">{title} ⓘ</div>
        <div class="metric-value">{value}</div>
        <div class="tooltip-note">{sub}</div>
    </div>
    """

def display_results(res):
    if not res: return
    f = res['features']
    q = res['quality']

    # 1. Metrics Grid
    st.markdown("### 📊 Audio Signal DNA")
    st.markdown("<div class='row-grid'>", unsafe_allow_html=True)
    
    st.markdown(metric_card("Energy", f"{int(f['energy']*100)}%", "Intensity/Loudness", "Higher = More Intense"), unsafe_allow_html=True)
    st.markdown(metric_card("Danceability", f"{int(f['danceability']*100)}%", "Rhythm Strength", "Higher = Groovier"), unsafe_allow_html=True)
    st.markdown(metric_card("Valence", f"{int(f['valence']*100)}%", "Musical Positivity", "Higher = Happy"), unsafe_allow_html=True)
    st.markdown(metric_card("Tempo", f"{int(f['tempo'])}", "Beats Per Minute", "Speed"), unsafe_allow_html=True)
    st.markdown(metric_card("Stability", f"{int(f['rhythm_stability']*100)}%", "Beat Consistency", "Higher = Robot-like"), unsafe_allow_html=True)
    st.markdown(metric_card("Acoustic", f"{int(f['acousticness']*100)}%", "Organic Sound", "Higher = Non-Electronic"), unsafe_allow_html=True)
    
    st.markdown("</div><br>", unsafe_allow_html=True)

    # 2. Context & Instruments
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='insight-box'>
            <h5 style='color:#6f9eff; margin:0'>Composition</h5>
            <hr style='border-color:rgba(255,255,255,0.1)'>
            <p><strong>🎼 Key:</strong> {res['key']}</p>
            <p><strong>🎹 Genre:</strong> {res['all_genres']}</p>
            <p><strong>🔊 Timbre:</strong> {q['timbre']} ({q['brightness_hz']} Hz)</p>
            <p><strong>🎚️ Dynamics:</strong> {q['compression']}</p>
        </div>""", unsafe_allow_html=True)
    
    with c2:
        if res['instruments']:
            tags = " ".join([f"<span class='recommendation-tag'>{i[0]} ({i[1]}%)</span>" for i in res['instruments']])
        else: tags = "None distinct detected"
        
        st.markdown(f"""
        <div class='insight-box'>
            <h5 style='color:#6f9eff; margin:0'>Instruments</h5>
            <hr style='border-color:rgba(255,255,255,0.1)'>
            <p>{tags}</p>
        </div>""", unsafe_allow_html=True)

    # 3. Lyrics
    st.markdown("### 📝 Lyrics")
    if res['lyrics']:
        st.markdown(f"<div class='insight-box'><em>\"{res['lyrics']}\"</em></div>", unsafe_allow_html=True)
    else:
        st.caption("No lyrics detected.")

    # 4. LLM Summary (Now at the bottom)
    st.markdown("### 🤖 AI Summary")
    if res['llm_summary']:
        st.markdown(f"<div class='insight-box llm-response'>{res['llm_summary']}</div>", unsafe_allow_html=True)
    else:
        st.info("Enter LLM Token in sidebar to see the AI summary.")

# --------------------
# Main UI
# --------------------
st.title("🎵 Audio Captioning")

tab1, tab2 = st.tabs(["🎵 Demo Library", "📤 Upload"])

with tab1:
    cols = st.columns(3)
    for i, (k, v) in enumerate(DEMO_SONGS.items()):
        with cols[i]:
            exists = os.path.exists(v['path'])
            st.markdown(f"**{v['name']}**")
            st.caption(v['desc'])
            if exists:
                if st.button("Analyze", key=k):
                    st.audio(v['path'])
                    with st.spinner("Processing..."):
                        r = run_analysis(v['path'], LLM_TOKEN)
                    display_results(r)
            else:
                st.error("File missing")

with tab2:
    up = st.file_uploader("Upload MP3/WAV", type=['mp3','wav'])
    if up:
        tmp = os.path.join(BASE_DIR, "temp_up.mp3")
        with open(tmp, "wb") as f: f.write(up.getbuffer())
        st.audio(tmp)
        if st.button("Analyze Upload"):
            with st.spinner("Processing..."):
                r = run_analysis(tmp, LLM_TOKEN)
            display_results(r)
        if os.path.exists(tmp): os.remove(tmp)