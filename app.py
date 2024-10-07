import streamlit as st
import cv2
import math
import numpy as np
import mysql.connector
from ultralytics import YOLO
from io import BytesIO
import datetime
import winsound
from PIL import Image
import urllib.parse

connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='admin',
    database='violations'
)

cursor = connection.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS violations2 (
    uid INT AUTO_INCREMENT PRIMARY KEY,
    violation_time DATETIME NOT NULL,
    violation_name VARCHAR(255) NOT NULL,
    violation_image LONGBLOB,
    workshop_name VARCHAR(255) NOT NULL
)
'''
)

model_yolo = YOLO("best.pt")
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 
              'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 
              'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 
              'semi', 'trailer', 'truck and trailer', 'truck', 'van', 
              'vehicle', 'wheel loader']

def insert_violation(violation_time, violation_name, violation_image, workshop_name='def'):
    query = "INSERT INTO violations2 (violation_time, violation_name, violation_image, workshop_name) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (violation_time, violation_name, violation_image, workshop_name))
    connection.commit()

def play_beep():
    frequency = 750 
    duration = 300
    winsound.Beep(frequency, duration)

def run_detection(source):
    cap = cv2.VideoCapture(0 if source == "Local Camera" else BytesIO(source.read()))

    video_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_image = frame.copy()

        results = model_yolo(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > 0.5 and currentClass in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                    violation_time = datetime.datetime.now().isoformat()
                    violation_image = cv2.imencode('.jpg', orig_image)[1].tobytes() 
                    insert_violation(violation_time, currentClass, violation_image)
                    play_beep()

                    myColor = (0, 0, 255)
                    frame = cv2.putText(frame, f"{currentClass}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

    cap.release()

def fetch_logs(limit=5):
    cursor.execute(f"SELECT uid, violation_time, violation_name, workshop_name FROM violations2 ORDER BY violation_time DESC LIMIT {limit}")
    return cursor.fetchall()

def fetch_all_logs():
    cursor.execute("SELECT uid, violation_time, violation_name, workshop_name FROM violations2 ORDER BY violation_time DESC")
    return cursor.fetchall()

def display_violation(uid):
    cursor.execute("SELECT violation_time, violation_name, violation_image, workshop_name FROM violations2 WHERE uid = %s", (uid,))
    result = cursor.fetchone()
    
    if result:
        violation_time, violation_name, violation_image, workshop_name = result
        
        st.write(f"**Violation Time:** {violation_time}")
        st.write(f"**Violation Name:** {violation_name}")
        st.write(f"**Workshop Name:** {workshop_name}")

        image = Image.open(BytesIO(violation_image))
        st.image(image, caption=f"Violation: {violation_name}")
        
    if st.button("Back"):
        # Remove the 'view_log' parameter from the URL to go back to the main screen
        query_params = st.experimental_get_query_params()
        query_params.pop("view_log", None)
        st.experimental_set_query_params(**query_params)
        st.experimental_rerun()

st.title("Real-Time Object Detection & Log Management")

if "run_detection" not in st.session_state:
    st.session_state.run_detection = False

# Handle the video source and detection start/stop actions
st.sidebar.header("Settings")
source = st.sidebar.selectbox("Select video source", ["Local Camera", "Video File"])

if st.sidebar.button("Run Detection"):
    st.session_state.run_detection = True

if st.sidebar.button("Stop Detection"):
    st.session_state.run_detection = False

# Fetch URL query parameters
query_params = st.experimental_get_query_params()
camera_container = st.container()
log_container = st.container()

# If no specific log is being viewed, display the camera and log table
if "view_log" not in query_params:
    with camera_container:
        st.markdown("<div style='height: 70vh; width: 100%;'>", unsafe_allow_html=True)

        if st.session_state.run_detection:
            run_detection(source)

        st.markdown("</div>", unsafe_allow_html=True)

    with log_container:
        st.markdown("<div style='height: 30vh; width: 100%; overflow-y: auto;'>", unsafe_allow_html=True)
        
        st.write("## Logs")
        
        logs = fetch_logs()  # Get only the first 5 logs
        all_logs = fetch_all_logs()  # For counting how many logs there are

        if logs:
            log_table = "<table style='width:100%;'><tr><th>ID</th><th>Time</th><th>Violation</th><th>Workshop</th><th>Action</th></tr>"
            for log in logs:
                uid, violation_time, violation_name, workshop_name = log
                log_table += f"<tr><td>{uid}</td><td>{violation_time}</td><td>{violation_name}</td><td>{workshop_name}</td>"
                log_table += f'<td><a href="/?view_log={uid}" target="_self">View</a></td></tr>'
            
            log_table += "</table>"
            st.write(log_table, unsafe_allow_html=True)
        else:
            st.write("No logs found.")

        # Show a "Show more" option if there are more logs
        if len(all_logs) > 5:
            st.write(f"Showing 5 of {len(all_logs)} logs. Use the scroll bar to see more.")
        st.markdown("</div>", unsafe_allow_html=True)

# If viewing a specific log, show only that log's details
if "view_log" in query_params:
    uid = query_params["view_log"][0]
    display_violation(uid)