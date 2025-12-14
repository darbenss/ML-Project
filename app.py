import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from imblearn.pipeline import Pipeline

# --- 1. CONFIGURATION & IMPORTS ---
# Import custom modules for model compatibility
# We assume preprocessing.py is in the same directory
from preprocessing import feature_engineering, select_numeric_cols

# Monkey patch for joblib to find functions if they were saved in __main__
import __main__
__main__.feature_engineering = feature_engineering
__main__.select_numeric_cols = select_numeric_cols

st.set_page_config(
    page_title="Student Success Simulator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. VISUAL STYLE: DARK GLASSMORPHISM ---
st.markdown("""
<style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        background-attachment: fixed;
    }
    
    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.5); 
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Metrics Styling */
    div[data-testid="metric-container"] {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        background: rgba(255,255,255,0.05);
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        background: rgba(255,255,255,0.08);
    }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, p, span, label {
        color: #ffffff !important;
    }
    
    /* Accent Color Overrides for Sliders/Widgets */
    div.stSlider > div[data-baseweb = "slider"] > div > div > div[role="slider"]{
        background-color: #00f2c3 !important;
    }
    
    /* Hide Default Header/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR: CONTROL CENTER ---
st.sidebar.title("🎛 Control Panel")
st.sidebar.markdown("---")

# Input Data Container
input_data = {}

# Group 1: Academic Profile
with st.sidebar.expander("📚 Academic Profile", expanded=True):
    input_data['study_hours_per_day'] = st.slider("Daily Study Hours", 0, 10, 4)
    input_data['attendance_percentage'] = st.slider("Attendance (%)", 30, 100, 85)
    input_data['Independent_Effort'] = st.slider("Independent Effort (0-10)", 0, 10, 5)
    input_data['internet_quality'] = st.selectbox("Internet Quality", ['Poor', 'Average', 'Good'], index=2)
    input_data['part_time_job'] = st.selectbox("Part Time Job", ['No', 'Yes'])

# Group 2: Lifestyle
with st.sidebar.expander("🧘 Lifestyle", expanded=True):
    input_data['sleep_hours'] = st.slider("Sleep Hours", 3, 10, 7)
    input_data['social_media_hours'] = st.slider("Social Media (Hrs)", 0, 8, 2)
    input_data['netflix_hours'] = st.slider("Netflix (Hrs)", 0, 6, 1)
    input_data['exercise_frequency'] = st.slider("Exercise (Times/Week)", 0, 7, 2)
    input_data['diet_quality'] = st.selectbox("Diet Quality", ['Poor', 'Fair', 'Good'], index=1)

# Group 3: Personal
with st.sidebar.expander("👤 Personal", expanded=False):
    input_data['age'] = st.slider("Age", 17, 25, 20)
    input_data['gender'] = st.selectbox("Gender", ['Male', 'Female'])
    input_data['parental_education_level'] = st.selectbox("Parent Edu Level", ['High School', 'Bachelor', 'Master', 'PhD'], index=1)
    input_data['mental_health_rating'] = st.slider("Mental Health (1-10)", 1, 10, 7)
    input_data['extracurricular_participation'] = st.selectbox("Extracurriculars", ['No', 'Yes'])

# Create DataFrame
input_df = pd.DataFrame([input_data])

# --- 4. MAIN DASHBOARD ---
st.title("🎓 Student Success AI")
st.markdown("### Real-time Performance Prediction")
st.markdown("---")

# --- 5. MODEL LOGIC ---
model_path = 'final_model_CatBoost.pkl'
prediction = None
probs = None
model_loaded = False

try:
    pipeline = joblib.load(model_path)
    raw_prediction = pipeline.predict(input_df)[0]
    probs = pipeline.predict_proba(input_df)[0]
    model_loaded = True
    prediction_idx = int(raw_prediction)
except Exception as e:
    st.error(f"Model could not be loaded. Running in Demo Mode. ({e})")
    # Mock data for demonstration if file is missing
    prediction_idx = 2  # Good
    probs = [0.1, 0.2, 0.5, 0.2] 

# Class Mapping
classes = ['Fail', 'Pass', 'Good', 'Distinction']
pred_label = classes[prediction_idx]

# Calculate Success Probability (Sum of Pass, Good, Distinction)
# Adjust logic depending on what defines "Success" in your context. 
# Here we assume Success = Not Failing.
success_prob = (1.0 - probs[0]) * 100 

# Risk Logic
if success_prob < 60:
    risk_level = "High"
    risk_color = "#ff4b4b" # Red
elif success_prob < 85:
    risk_level = "Moderate"
    risk_color = "#ffa600" # Orange
else:
    risk_level = "Low"
    risk_color = "#00f2c3" # Neon Cyan

# Color coding for prediction label
pred_color = "#00f2c3" if prediction_idx >= 2 else ("#ffa600" if prediction_idx == 1 else "#ff4b4b")

# --- ROW 1: METRICS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<h3 style='text-align: center; color: #aaa !important;'>Predicted Grade</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: {pred_color} !important; margin-top: -20px;'>{pred_label}</h1>", unsafe_allow_html=True)

with col2:
    st.metric("Success Probability", f"{success_prob:.1f}%", delta=f"{success_prob-75:.1f}% vs Avg")

with col3:
    st.markdown(f"<h3 style='text-align: center; color: #aaa !important;'>Risk Factor</h3>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: {risk_color} !important;'>{risk_level}</h2>", unsafe_allow_html=True)

st.markdown("---")

# --- ROW 2: VISUALS ---
viz_col1, viz_col2 = st.columns([1, 1])

# Chart 1: Gauge (Success Probability)
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = success_prob,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Pass Probability", 'font': {'color': 'white', 'size': 20}},
    number = {'font': {'color': 'white'}},
    gauge = {
        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
        'bar': {'color': "rgba(255,255,255,0.2)"},
        'bgcolor': "rgba(0,0,0,0)",
        'borderwidth': 2,
        'bordercolor': "rgba(255,255,255,0.3)",
        'steps': [
            {'range': [0, 50], 'color': 'rgba(255, 75, 75, 0.6)'},
            {'range': [50, 75], 'color': 'rgba(255, 166, 0, 0.6)'},
            {'range': [75, 100], 'color': 'rgba(0, 242, 195, 0.6)'}
        ],
    }
))
fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})

with viz_col1:
    st.plotly_chart(fig_gauge, use_container_width=True)

# Chart 2: Radar (Comparison)
# Normalizing user values for the chart (approximate scales)
# Axes: Study (max 10), Sleep (max 10), Attendance (max 100), Mental (max 10), Effort (max 10)
categories = ['Study', 'Sleep', 'Attendance', 'Mental Health', 'Indep. Effort']
user_values = [
    input_data['study_hours_per_day'],
    input_data['sleep_hours'],
    input_data['attendance_percentage'],
    input_data['mental_health_rating'],
    input_data['Independent_Effort']
]
# Normalized for visualization (scaling attendance down to 0-10 scale equivalent roughly)
user_vis_values = [
    user_values[0], 
    user_values[1], 
    user_values[2] / 10, # Scale 100 -> 10
    user_values[3], 
    user_values[4]
]
benchmark_values = [6, 8, 9.5, 8, 7] # Scaled top 10% (Attendance 95 -> 9.5)

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=user_vis_values,
    theta=categories,
    fill='toself',
    name='You',
    line_color='#00f2c3'
))
fig_radar.add_trace(go.Scatterpolar(
    r=benchmark_values,
    theta=categories,
    fill='toself',
    name='Top 10% Avg',
    line_color='#ff00ff',
    opacity=0.5
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, linecolor='rgba(255,255,255,0.2)', gridcolor='rgba(255,255,255,0.1)'),
        angularaxis=dict(tickfont=dict(color='white'), linecolor='rgba(255,255,255,0.2)'),
        bgcolor='rgba(0,0,0,0)'
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=40, r=40, t=40, b=40),
    template='plotly_dark'
)

with viz_col2:
    st.markdown("<h4 style='text-align: center;'>Benchmark Analysis</h4>", unsafe_allow_html=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# --- ROW 3: AI ADVISOR ---
st.markdown("### 🤖 AI Study Advisor")

tips = []
if input_data['sleep_hours'] < 6:
    tips.append("🌙 **Sleep Optimization:** You are getting less than 6 hours of sleep. Cognitive function drops significantly at this range. Try to get at least +1 hour tonight.")
if input_data['attendance_percentage'] < 75:
    tips.append("🏫 **Attendance Warning:** Your attendance is below 75%. This is a critical factor for distinctions. Prioritize showing up to class.")
if input_data['social_media_hours'] + input_data['netflix_hours'] > 4:
    tips.append("📱 **Distraction Alert:** Total screen entertainment time exceeds 4 hours. Consider the 'Pomodoro' technique to reclaim study time.")
if input_data['study_hours_per_day'] < 2:
    tips.append("📚 **Study Routine:** Current study hours are low. Even consistent 30-minute blocks can improve retention.")
if len(tips) == 0:
    tips.append("🌟 **Great Job!** Your metrics look balanced. Keep maintaining this lifestyle for optimal performance.")

for tip in tips:
    st.info(tip)