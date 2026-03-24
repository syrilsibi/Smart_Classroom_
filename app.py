import streamlit as st
import cv2
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
from mtcnn import MTCNN
from keras_facenet import FaceNet

# CONFIGURATION
ENCODINGS_PATH = r"D:\New folder\SKILLPARK\SMART_CLASSROOOM\encodings.pkl"
REPORT_PATH = r"D:\New folder\SKILLPARK\SMART_CLASSROOOM"

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ----------------------------------------------------
st.set_page_config(page_title="Smart Classroom Attendance", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    /* Modern Slate Blue Theme */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    /* Metric Dashboard Styling */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    div[data-testid="metric-container"] > label {
        color: #94a3b8 !important;
        font-size: 1.1rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        width: 100%;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    
    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e2e8f0;
    }
    
    /* Video Container 'Card' */
    .video-card {
        background-color: #1e293b;
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        text-align: center;
    }
    
    /* Pulse Animation for Video Frame */
    @keyframes pulse-green {
        0% { border-color: #334155; }
        50% { border-color: #22c55e; box-shadow: 0 0 15px #22c55e; }
        100% { border-color: #334155; }
    }
    .pulse-active {
        animation: pulse-green 1s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. MODULAR MODEL LOADING
# ----------------------------------------------------
@st.cache_resource
def load_detector():
    return MTCNN()

@st.cache_resource
def load_embedder():
    return FaceNet()

@st.cache_data
def load_database():
    if not os.path.exists(ENCODINGS_PATH):
        return None, None
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]

# Initialize Models
detector = load_detector()
embedder = load_embedder()
known_encodings, known_names = load_database()

if known_encodings is None:
    st.error("Encoding file not found. Please run train_system.py first.")
    st.stop()

# ----------------------------------------------------
# 3. STATE MANAGEMENT
# ----------------------------------------------------
if 'attendance_list' not in st.session_state:
    st.session_state.attendance_list = []
if 'present_names' not in st.session_state:
    st.session_state.present_names = set()
if 'latest_confidence' not in st.session_state:
    st.session_state.latest_confidence = "--"
if 'system_status' not in st.session_state:
    st.session_state.system_status = "🟢 Active"
if 'last_recognized' not in st.session_state:
    st.session_state.last_recognized = False

# ----------------------------------------------------
# 4. SIDEBAR NAVIGATION & CONTROLS
# ----------------------------------------------------
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio("Go to", ["Live Attendance", "View Records", "System Settings"])

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Controls")
THRESHOLD = st.sidebar.slider("Recognition Threshold", min_value=0.1, max_value=1.5, value=0.70, step=0.05)

if st.sidebar.button("💾 Save Logs to Excel"):
    if st.session_state.attendance_list:
        df = pd.DataFrame(st.session_state.attendance_list)
        fname = f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        df.to_excel(os.path.join(REPORT_PATH, fname), index=False)
        st.toast(f"✅ Excel saved: {fname}")
        st.sidebar.success(f"Saved: {fname}")
    else:
        st.sidebar.warning("No attendance data to save.")

# ----------------------------------------------------
# 5. MAIN CONTENT LAYOUT
# ----------------------------------------------------
st.title("🎓 Smart Classroom: AI Attendance System")

if menu == "Live Attendance":
    
    # Header Metrics Dashboard
    col1, col2, col3, col4 = st.columns(4)
    total_registered = len(set(known_names)) if known_names else 0
    with col1:
        st.metric("Total Registered", f"{total_registered:,}")
    with col2:
        st.metric("Present Today", len(st.session_state.present_names))
    with col3:
        st.metric("Last Confidence", st.session_state.latest_confidence)
    with col4:
        st.metric("System Status", st.session_state.system_status)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Video and Log Layout
    vid_col, log_col = st.columns([2, 1])
    
    with vid_col:
        st.subheader("📹 Live Camera Feed")
        
        # Wrapped container for styling
        st.markdown('<div class="video-card">', unsafe_allow_html=True)
        run_camera = st.checkbox("Toggle Camera On/Off")
        FRAME_WINDOW = st.image([], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with log_col:
        st.subheader("📝 Live Logs")
        log_placeholder = st.empty()
        
        # Function to display the dataframe safely
        def display_recent_logs():
            if st.session_state.attendance_list:
                df_logs = pd.DataFrame(st.session_state.attendance_list)
                # Reverse to show newest at top (optional, but good for logs)
                df_logs = df_logs.iloc[::-1]
            else:
                df_logs = pd.DataFrame(columns=["Name", "Time", "Score"])
            log_placeholder.dataframe(df_logs, use_container_width=True, hide_index=True)
            
        display_recent_logs()

    # Core Camera / Recognition Loop
    if run_camera:
        cap = cv2.VideoCapture(0)
        while run_camera:
            ret, frame = cap.read()
            if not ret: 
                st.error("Could not read from webcam.")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_f, w_f, _ = frame_rgb.shape
            
            # Reset pulse flag for Streamlit update (optional trick if we were passing JS, 
            # but we simulate pulse by frame drawing in Python)
            detected_new_person = False
            
            try:
                faces = detector.detect_faces(frame_rgb)
            except: 
                continue

            for face in faces:
                x, y, w, h = face['box']
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_f, x1 + w), min(h_f, y1 + h)

                if (x2 - x1) > 20:
                    face_crop = cv2.resize(frame_rgb[y1:y2, x1:x2], (160, 160))
                    face_batch = np.expand_dims(face_crop, axis=0)
                    live_enc = embedder.embeddings(face_batch)[0]

                    distances = np.linalg.norm(known_encodings - live_enc, axis=1)
                    idx = np.argmin(distances)
                    score = round(distances[idx], 2)

                    name, color = "Unknown", (255, 0, 0) # Red
                    if score < THRESHOLD:
                        name, color = known_names[idx], (0, 255, 0) # Green
                        
                        # Calculate confidence % (Assuming 1.0 is max threshold, lower is better)
                        conf_pct = max(0, int((1 - (score / 1.5)) * 100))
                        st.session_state.latest_confidence = f"{conf_pct}%"
                        
                        if name not in st.session_state.present_names:
                            st.session_state.present_names.add(name)
                            st.session_state.attendance_list.append({
                                "Name": name, 
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Score": score
                            })
                            # Visual Feedback Toast
                            st.toast(f"✅ Success: {name} recognized!", icon="🎉")
                            detected_new_person = True

                    # Draw Bounding Box and Label
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 3)
                    
                    # Background for text
                    cv2.rectangle(frame_rgb, (x1, y1-35), (x1+(len(name)*15), y1), color, cv2.FILLED)
                    cv2.putText(frame_rgb, f"{name}", (x1+5, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Update video feed natively
            FRAME_WINDOW.image(frame_rgb)
            
            # If someone was newly detected, we refresh the logs and metrics immediately
            if detected_new_person:
                display_recent_logs()
                
        cap.release()

elif menu == "View Records":
    st.subheader("🗃️ Attendance Records & History")
    
    if st.session_state.attendance_list:
        df = pd.DataFrame(st.session_state.attendance_list)
        
        # Searchable logs
        search_query = st.text_input("🔍 Search by Name...")
        if search_query:
            filtered_df = df[df["Name"].str.contains(search_query, case=False)]
        else:
            filtered_df = df
            
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        # CSV Download functionality
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV Record",
            data=csv,
            file_name=f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No attendance records found yet. Start the camera to begin.")

elif menu == "System Settings":
    st.subheader("⚙️ System Configuration")
    
    colA, colB = st.columns(2)
    with colA:
        st.write("**Database Path:**")
        st.code(ENCODINGS_PATH)
        st.write("**Reporting Path:**")
        st.code(REPORT_PATH)
        
    with colB:
        st.write(f"**Total Registered Models Loaded:** {len(known_names) if known_names else 0}")
        st.write(f"**Current Threshold:** {THRESHOLD}")
        
    st.markdown("---")
    st.subheader("🛠️ Maintenance")
    if st.button("🔄 Clear Current Session Data"):
        st.session_state.attendance_list = []
        st.session_state.present_names = set()
        st.session_state.latest_confidence = "--"
        st.success("Attendance session has been cleared.")
        st.rerun()