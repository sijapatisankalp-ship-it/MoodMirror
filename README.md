# MoodMirror – Real-Time Emotion Detection App  
An AI-powered emotion recognition system using DeepFace, OpenCV, and Streamlit.

---

## 🧠 Project Overview
Human emotions play a crucial role in communication, mental health, and human–computer interaction.  
**MoodMirror** is a real-time AI application that detects emotions directly from a webcam feed.  
It helps demonstrate how machine learning models can interpret facial expressions and provide instant emotional feedback.

This project was developed as part of the **Kaggle 5-Day AI Intensive Bootcamp – Capstone Project**.

---

# 🎯 Problem Statement
With the rise of digital communication, computers often struggle to understand human emotional states.  
Traditional interfaces cannot automatically detect whether a user is happy, sad, stressed, or angry.

**Goal:**  
Build a real-time system that detects human emotions through facial expressions using computer vision and deep learning.

Applications of such systems include:
- Mental health monitoring  
- Emotion-based user interfaces  
- Smart assistants  
- Personalized content recommendation  
- Human–robot interaction  

---

# 🚀 Features
- 📸 **Live webcam feed**
- 😊 **Real-time emotion recognition**
- 🔍 **DeepFace emotion analysis**
- 📊 **Confidence scores**
- 🎨 **Clean Streamlit UI**
- ⚙️ **Works on CPU (no GPU required)**

---

# 🛠 Tech Stack
- Python  
- Streamlit  
- OpenCV  
- DeepFace  
- TensorFlow / Keras  
- NumPy  
- Pillow  

---

# 📚 How It Works (Technical Workflow)
1. **Streamlit UI loads**, providing a simple interface.  
2. **OpenCV captures frames** from the webcam.  
3. Each frame is passed into **DeepFace**, which performs:
   - Face detection  
   - Alignment  
   - Emotion classification (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust)  
4. The model outputs:
   - **Predicted emotion**  
   - **Confidence scores** for each emotion  
5. Streamlit displays results in real-time.

---

# 🧩 Understanding DeepFace (Model Explanation)
DeepFace is a lightweight facial analysis library built on top of several deep learning models.

Your app uses DeepFace’s **Emotion Model**, which is based on:
- **VGG-Face backbone** (or Facenet/ArcFace depending on configuration)  
- A CNN trained on thousands of aligned face images  
- Outputs probabilities for 7 emotion classes  

**Preprocessing Steps:**
- Face detection using OpenCV Haar Cascade  
- Face alignment  
- Resizing to model input shape  
- Normalization  

**Inference:**
The model outputs a probability distribution:
```
{ happy: 0.81, neutral: 0.12, sad: 0.03, angry: 0.02, surprise: 0.01 }
```

The class with the highest score becomes the predicted emotion.

---

# 📊 Results & Observations
### Example Output (Sample Results)
| Frame | Predicted Emotion | Confidence | Notes |
|-------|-------------------|------------|-------|
| 1     | Happy             | 81%        | Good lighting, face centered |
| 2     | Neutral           | 57%        | Slight head tilt |
| 3     | Angry             | 64%        | Low light environment |
| 4     | Surprise          | 73%        | Eyebrows raised |

### General Findings
- Bright lighting improves accuracy.  
- Side-angled faces reduce confidence.  
- Glasses/face masks can reduce detection quality.  
- Real-time prediction is fast even on CPU.

---

# 🔮 Future Improvements
- Multi-face detection  
- Emotion trend graph over time  
- Emotion-based music or content recommendations  
- Voice emotion recognition  
- No-webcam mode using uploaded images  
- Hybrid model combining audio + video emotions  

---

# 📝 Reflection (What I Learned)
- How deep learning models interpret facial expressions  
- How to integrate DeepFace with Streamlit for real-time processing  
- Performance considerations when processing live video streams  
- How lighting, camera quality, and orientation affect predictions  
- Practical challenges of deploying AI to real users  

---

# 📦 Installation

Clone the project:  
```
git clone <repo>
cd MoodMirror
```

Create and activate virtual environment:

**Mac/Linux**
```
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the app:
```
streamlit run main.py
```

---

# 👨‍💻 Author
**Nirdip Sijapati**  
Capstone Project – Kaggle 5-Day AI Intensive Bootcamp

🎉 **Project Complete**

