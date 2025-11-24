import streamlit as st  # pyright: ignore[reportMissingImports]
import pandas as pd
import spotipy  # pyright: ignore[reportMissingImports]
from spotipy.oauth2 import SpotifyClientCredentials  # pyright: ignore[reportMissingImports]
from deepface import DeepFace  # pyright: ignore[reportMissingImports]
import cv2
import keras

import numpy as np
import tempfile
import os

# ==========================================
# 1. CONFIGURATION (API KEYS)
# ==========================================
# PASTE YOUR KEYS HERE FROM THE SPOTIFY DASHBOARD
CLIENT_ID = '70d0343f385a4b96bbaa730ab02f6923'
CLIENT_SECRET = '989ca508074147288a916f19015c2308'

# Connect to Spotify
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# ==========================================
# 2. THE INTELLIGENCE (LOGIC)
# ==========================================
def get_mood_filters(emotion):
    """
    Translates a facial emotion into Spotify Audio Features.
    Ranges are usually 0.0 to 1.0.
    Uses only valid Spotify genre seeds.
    """
    mood_mapping = {
        'happy': {'valence': 0.9, 'energy': 0.8, 'genre': 'pop'},
        'sad': {'valence': 0.2, 'energy': 0.3, 'genre': 'acoustic'},
        'angry': {'valence': 0.3, 'energy': 0.9, 'genre': 'rock'},
        'fear': {'valence': 0.3, 'energy': 0.5, 'genre': 'classical'},
        'surprise': {'valence': 0.8, 'energy': 0.7, 'genre': 'edm'},
        'neutral': {'valence': 0.5, 'energy': 0.5, 'genre': 'chill'},
        'disgust': {'valence': 0.4, 'energy': 0.6, 'genre': 'metal'}
    }
    # Default to 'neutral' if something weird happens
    return mood_mapping.get(emotion, mood_mapping['neutral'])

def get_tracks_by_mood(sp, genre, target_valence, target_energy, limit=5):
    """
    Alternative to recommendations API: Search for tracks by genre.
    Tries to filter by audio features, but falls back to simple search if unavailable.
    """
    try:
        # Step 1: Search for tracks in the genre
        # Try different search strategies
        search_queries = [
            f"genre:{genre}",
            genre,
            f"tag:{genre}",
            f"{genre} music"
        ]
        
        search_results = None
        for query in search_queries:
            try:
                search_results = sp.search(q=query, type='track', limit=limit*10, market='US')
                if search_results['tracks']['items']:
                    break
            except:
                continue
        
        if not search_results or not search_results['tracks']['items']:
            return []
        
        tracks = search_results['tracks']['items']
        
        # Step 2: Try to get audio features (may fail with 403)
        try:
            track_ids = [track['id'] for track in tracks if track.get('id')]
            if track_ids:
                # Spotify audio_features can handle up to 100 IDs at once
                audio_features_list = []
                for i in range(0, len(track_ids), 100):
                    batch = track_ids[i:i+100]
                    try:
                        features_batch = sp.audio_features(batch)
                        if features_batch:
                            audio_features_list.extend(features_batch)
                    except Exception as features_error:
                        # If audio_features fails (403, etc.), skip filtering
                        st.info("Using simple search (audio features unavailable)")
                        return tracks[:limit]
                
                # Create a dictionary mapping track_id to features
                features_dict = {}
                for idx, track_id in enumerate(track_ids):
                    if idx < len(audio_features_list) and audio_features_list[idx]:
                        features_dict[track_id] = audio_features_list[idx]
                
                # Step 3: Score tracks based on how close they match target valence and energy
                scored_tracks = []
                for track in tracks:
                    track_id = track.get('id')
                    if track_id and track_id in features_dict:
                        features = features_dict[track_id]
                        if features.get('valence') is not None and features.get('energy') is not None:
                            # Calculate distance from target (lower is better)
                            valence_diff = abs(features['valence'] - target_valence)
                            energy_diff = abs(features['energy'] - target_energy)
                            score = valence_diff + energy_diff
                            
                            scored_tracks.append({
                                'track': track,
                                'score': score
                            })
                
                # Step 4: Sort by score (best matches first) and return top N
                if scored_tracks:
                    scored_tracks.sort(key=lambda x: x['score'])
                    return [item['track'] for item in scored_tracks[:limit]]
        except Exception as features_error:
            # If audio_features fails, just return top tracks from search
            st.info("Using simple search (audio features unavailable)")
            return tracks[:limit]
        
        # Fallback: return top tracks from search
        return tracks[:limit]
        
    except Exception as e:
        st.warning(f"Error in search method: {str(e)}")
        return []

# ==========================================
# 3. THE INTERFACE (FRONTEND)
# ==========================================
st.set_page_config(
    page_title="MoodMirror", 
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dynamic UI with React Bits-inspired animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    * {
        box-sizing: border-box;
    }
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern gradient background with animated particles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        position: relative;
        overflow-x: hidden;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.2) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(255, 119, 198, 0.2) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(120, 219, 255, 0.15) 0%, transparent 50%);
        animation: gradientShift 15s ease infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Ensure main content is above background */
    .main .block-container {
        position: relative;
        z-index: 1;
    }
    
    /* Make sure Streamlit elements are visible */
    [data-testid="stAppViewContainer"] {
        position: relative;
        z-index: 1;
    }
    
    @keyframes gradientShift {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }
    
    /* Animated header with glassmorphism */
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px) saturate(150%);
        -webkit-backdrop-filter: blur(10px) saturate(150%);
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        position: relative;
        overflow: hidden;
        animation: fadeInDown 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        z-index: 10;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff6b9d 0%, #c44569 25%, #f8b500 50%, #ff6b9d 75%, #c44569 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: gradientFlow 5s ease infinite, float 3s ease-in-out infinite;
        position: relative;
        z-index: 1;
    }
    
    @keyframes gradientFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .subtitle {
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.85);
        font-weight: 400;
        margin-top: 0.5rem;
        animation: fadeIn 1s ease 0.3s both;
        position: relative;
        z-index: 1;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Emotion card with scale and fade animation */
    .emotion-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px) saturate(150%);
        -webkit-backdrop-filter: blur(8px) saturate(150%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: scaleIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
        z-index: 10;
    }
    
    .emotion-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .emotion-card:hover::after {
        left: 100%;
    }
    
    .emotion-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.8) translateY(30px);
        }
        to {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    /* Track cards with stagger animation */
    .track-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px) saturate(150%);
        -webkit-backdrop-filter: blur(8px) saturate(150%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-left: 4px solid #ff6b9d;
        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: slideInLeft 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        position: relative;
        overflow: hidden;
        z-index: 10;
    }
    
    .track-card:nth-child(1) { animation-delay: 0.1s; }
    .track-card:nth-child(2) { animation-delay: 0.2s; }
    .track-card:nth-child(3) { animation-delay: 0.3s; }
    .track-card:nth-child(4) { animation-delay: 0.4s; }
    .track-card:nth-child(5) { animation-delay: 0.5s; }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px) scale(0.9);
        }
        to {
            opacity: 1;
            transform: translateX(0) scale(1);
        }
    }
    
    .track-card:hover {
        transform: translateX(15px) scale(1.02);
        box-shadow: 0 12px 40px rgba(255, 107, 157, 0.4);
        border-left-width: 6px;
        border-left-color: #ff6b9d;
    }
    
    /* Enhanced emotion badges with modern colors */
    .emotion-badge {
        display: inline-block;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.3rem;
        margin: 1rem 0;
        animation: bounceIn 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .emotion-badge::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .emotion-badge:hover::before {
        width: 300px;
        height: 300px;
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3) rotate(-180deg);
        }
        50% {
            transform: scale(1.1) rotate(5deg);
        }
        100% {
            opacity: 1;
            transform: scale(1) rotate(0deg);
        }
    }
    
    /* Modern color palette */
    .happy { 
        background: linear-gradient(135deg, #ff6b9d 0%, #c44569 100%); 
        color: white; 
    }
    .sad { 
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
        color: white; 
    }
    .angry { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
        color: white; 
    }
    .fear { 
        background: linear-gradient(135deg, #a8c0ff 0%, #3f5efb 100%); 
        color: white; 
    }
    .surprise { 
        background: linear-gradient(135deg, #f8b500 0%, #fce043 100%); 
        color: #1a1a1a; 
    }
    .neutral { 
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%); 
        color: #1a1a1a; 
    }
    .disgust { 
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%); 
        color: white; 
    }
    
    /* Genre tag with pulse animation */
    .genre-tag {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        background: rgba(255, 107, 157, 0.2);
        border: 1px solid rgba(255, 107, 157, 0.4);
        border-radius: 25px;
        font-size: 1rem;
        color: #ff6b9d;
        font-weight: 600;
        margin: 0.5rem 0;
        animation: pulseGlow 2s ease-in-out infinite;
        backdrop-filter: blur(10px);
    }
    
    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 0 10px rgba(255, 107, 157, 0.3);
        }
        50% {
            box-shadow: 0 0 20px rgba(255, 107, 157, 0.6);
        }
    }
    
    .artist-name {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    
    /* Enhanced Spotify link */
    .spotify-link {
        display: inline-block;
        padding: 0.7rem 2rem;
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white;
        text-decoration: none;
        border-radius: 30px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        margin-top: 0.5rem;
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .spotify-link::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .spotify-link:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .spotify-link:hover {
        background: linear-gradient(135deg, #1ed760 0%, #1DB954 100%);
        transform: scale(1.1) translateY(-2px);
        box-shadow: 0 6px 25px rgba(29, 185, 84, 0.6);
    }
    
    /* Camera container with glow effect */
    .camera-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px) saturate(150%);
        -webkit-backdrop-filter: blur(8px) saturate(150%);
        padding: 2.5rem;
        border-radius: 24px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeInUp 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        z-index: 10;
        position: relative;
    }
    
    /* Ensure Streamlit widgets are visible */
    [data-testid="stCameraInput"] {
        position: relative;
        z-index: 20;
    }
    
    /* Make sure text is readable */
    .emotion-card p,
    .track-card h3,
    .track-card p {
        color: rgba(255, 255, 255, 0.95) !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Ensure Streamlit native elements are visible */
    .stButton > button {
        position: relative;
        z-index: 20;
        backdrop-filter: none !important;
    }
    
    [data-testid="stCameraInput"] > div {
        position: relative;
        z-index: 20;
        backdrop-filter: none !important;
    }
    
    /* Make sure audio players are visible */
    audio {
        position: relative;
        z-index: 20;
    }
    
    /* Ensure all Streamlit widgets are above background */
    .stApp > div {
        position: relative;
        z-index: 1;
    }
    
    /* Fix any hidden elements */
    .element-container {
        position: relative;
        z-index: 10;
    }
    
    /* Reduce blur on main content area */
    .main .block-container {
        backdrop-filter: none !important;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Loading spinner animation */
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Smooth scroll behavior */
    html {
        scroll-behavior: smooth;
    }
    
    /* Text selection color */
    ::selection {
        background: rgba(255, 107, 157, 0.3);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Header with animated particles
st.markdown("""
    <div class="main-header">
        <h1>🎵 MoodMirror</h1>
        <p class="subtitle">The AI DJ that reads your face</p>
    </div>
    
    <script>
    // Add floating particles animation
    function createParticles() {
        const container = document.querySelector('.stApp');
        if (!container) return;
        
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.style.cssText = `
                position: fixed;
                width: ${Math.random() * 4 + 2}px;
                height: ${Math.random() * 4 + 2}px;
                background: rgba(255, 107, 157, ${Math.random() * 0.5 + 0.2});
                border-radius: 50%;
                pointer-events: none;
                z-index: 1;
                left: ${Math.random() * 100}%;
                top: ${Math.random() * 100}%;
                animation: floatParticle ${Math.random() * 10 + 10}s infinite ease-in-out;
                animation-delay: ${Math.random() * 5}s;
            `;
            container.appendChild(particle);
        }
    }
    
    // Add CSS for particle animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes floatParticle {
            0%, 100% {
                transform: translate(0, 0) scale(1);
                opacity: 0.3;
            }
            25% {
                transform: translate(100px, -100px) scale(1.2);
                opacity: 0.6;
            }
            50% {
                transform: translate(-50px, -200px) scale(0.8);
                opacity: 0.4;
            }
            75% {
                transform: translate(-100px, -50px) scale(1.1);
                opacity: 0.5;
            }
        }
    `;
    document.head.appendChild(style);
    
    // Initialize particles when page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createParticles);
    } else {
        createParticles();
    }
    
    // Add stagger animation to track cards
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateX(0)';
                }, index * 100);
            }
        });
    }, observerOptions);
    
    // Observe track cards when they're added
    setTimeout(() => {
        document.querySelectorAll('.track-card').forEach(card => {
            observer.observe(card);
        });
    }, 1000);
    </script>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: rgba(255, 255, 255, 0.9); margin-bottom: 2rem; animation: fadeIn 1s ease 0.5s both;">
        <p style="font-size: 1.2rem; font-weight: 300;">✨ Take a photo and let AI analyze your emotions to find the perfect soundtrack ✨</p>
    </div>
""", unsafe_allow_html=True)

# THE WEBCAM INPUT with styled container
st.markdown('<div class="camera-container">', unsafe_allow_html=True)
img_file_buffer = st.camera_input("📸 Capture your mood", label_visibility="visible")
st.markdown('</div>', unsafe_allow_html=True)

# Add inline style to ensure camera is visible
st.markdown("""
    <style>
    [data-testid="stCameraInput"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    [data-testid="stCameraInput"] video {
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

if img_file_buffer is not None:
    # A. PRE-PROCESS IMAGE
    # Convert the raw image data to something OpenCV can read
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # B. AI ANALYSIS with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.markdown("""
        <div style="text-align: center; color: white; font-size: 1.1rem; margin: 1rem 0;">
            🔍 Processing your image...
        </div>
    """, unsafe_allow_html=True)
    progress_bar.progress(20)
    
    temp_path = None
    try:
        # Save image to temporary file for DeepFace
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, cv2_img)
            temp_path = tmp_file.name
        
        # DeepFace analyzes the image for emotion
        status_text.markdown("""
            <div style="text-align: center; color: white; font-size: 1.1rem; margin: 1rem 0;">
                🧠 Analyzing facial expressions...
            </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(50)
        
        # enforce_detection=False prevents crash if face is blurry
        analysis = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        
        progress_bar.progress(70)
        
        # Get the dominant emotion (e.g., "happy")
        dominant_emotion = analysis[0]['dominant_emotion']
        confidence = analysis[0]['face_confidence']
        
        # Get emotion breakdown for display
        emotions = analysis[0]['emotion']
        emotion_emoji = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😲',
            'neutral': '😐',
            'disgust': '🤢'
        }
        
        # Display emotion with dynamic styling
        emoji = emotion_emoji.get(dominant_emotion, '🎭')
        st.markdown(f"""
            <div class="emotion-card">
                <div style="text-align: center;">
                    <div class="emotion-badge {dominant_emotion}">
                        {emoji} {dominant_emotion.upper()}
                    </div>
                    <p style="margin-top: 1rem; color: #666;">
                        Confidence: {confidence:.1%} | Analyzing your mood...
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # C. MUSIC RECOMMENDATION
        # Get the audio feature targets
        filters = get_mood_filters(dominant_emotion)
        
        # Use alternative method since recommendations API is not available
        status_text.markdown(f"""
            <div style="text-align: center; color: white; font-size: 1.1rem; margin: 1rem 0;">
                🎵 Searching for perfect {filters['genre']} tracks...
            </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(85)
        
        tracks = get_tracks_by_mood(
            sp, 
            filters['genre'], 
            filters['valence'], 
            filters['energy'], 
            limit=5
        )
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        if not tracks:
            st.error("Could not find tracks. Please try again or check your internet connection.")
            raise Exception("No tracks found")
        
        # D. DISPLAY RESULTS with dynamic styling
        st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0;">
                <h2 style="color: white; font-size: 2rem; margin-bottom: 0.5rem;">
                    🎶 Your Perfect Playlist 🎶
                </h2>
                <span class="genre-tag">{filters['genre'].title()} Music</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Display tracks in a more dynamic way
        for idx, track in enumerate(tracks, 1):
            album_image = track['album']['images'][0]['url'] if track['album']['images'] else None
            track_name = track['name']
            artist_name = track['artists'][0]['name']
            preview_url = track.get('preview_url')
            spotify_url = track['external_urls']['spotify']
            
            # Create dynamic track card
            col1, col2 = st.columns([1.5, 4])
            
            with col1:
                if album_image:
                    st.markdown(f"""
                        <div style="text-align: center; position: relative;">
                            <img src="{album_image}" 
                                 style="width: 100%; 
                                        border-radius: 20px; 
                                        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
                                        transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
                                        cursor: pointer;"
                                 onmouseover="this.style.transform='scale(1.1) rotate(2deg)'; this.style.boxShadow='0 12px 35px rgba(255, 107, 157, 0.5)';"
                                 onmouseout="this.style.transform='scale(1) rotate(0deg)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.4)';">
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="track-card">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <span style="background: linear-gradient(135deg, #ff6b9d 0%, #c44569 100%); 
                                        color: white; 
                                        width: 40px; 
                                        height: 40px; 
                                        border-radius: 50%; 
                                        display: flex; 
                                        align-items: center; 
                                        justify-content: center; 
                                        font-weight: bold; 
                                        margin-right: 1rem;
                                        box-shadow: 0 4px 15px rgba(255, 107, 157, 0.4);
                                        animation: pulse 2s ease-in-out infinite;">
                                {idx}
                            </span>
                            <h3 style="margin: 0; color: rgba(255, 255, 255, 0.95); font-size: 1.6rem; font-weight: 600;">{track_name}</h3>
                        </div>
                        <p class="artist-name">🎤 {artist_name}</p>
                        <p style="color: rgba(255, 255, 255, 0.7); font-size: 1rem; margin: 0.5rem 0;">
                            💿 {track['album']['name']}
                        </p>
                """, unsafe_allow_html=True)
                
                # Audio player
                if preview_url:
                    st.audio(preview_url, format='audio/mp3')
                else:
                    st.markdown('<p style="color: rgba(255, 255, 255, 0.6); font-style: italic;">🎵 No preview available</p>', unsafe_allow_html=True)
                
                # Spotify link
                st.markdown(f"""
                        <a href="{spotify_url}" target="_blank" class="spotify-link">
                            🎧 Listen on Spotify
                        </a>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.markdown("""
            <div style="background: rgba(255, 0, 0, 0.1); padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
                <h3 style="color: #ff6b6b; margin-bottom: 1rem;">⚠️ Oops! Something went wrong</h3>
                <p style="color: white;">I couldn't analyze your face clearly. Try:</p>
                <ul style="color: white; text-align: left; display: inline-block;">
                    <li>Moving closer to the camera</li>
                    <li>Ensuring good lighting</li>
                    <li>Looking directly at the camera</li>
                    <li>Taking another photo</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        # Print the actual error to the console for debugging
        with st.expander("🔍 Technical Details"):
            st.exception(e)
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)