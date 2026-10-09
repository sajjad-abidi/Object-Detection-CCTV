import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import datetime
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CCTV Cow Monitor",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SIDEBAR - CCTV SETTINGS
# ============================================
st.sidebar.title("📹 CCTV Settings")
st.sidebar.markdown("---")

# RTSP URL - Your Dahua camera
rtsp_url = st.sidebar.text_input(
    "RTSP URL",
    value="rtsp://admin:L2569DF7@192.168.29.122:554/cam/realmonitor?channel=1&subtype=0"
)

# Stream type selection
stream_type = st.sidebar.selectbox(
    "Stream Quality",
    ["Main Stream (High Quality)", "Sub Stream (Faster)"],
    help="Sub stream is faster but lower quality"
)

# Auto-update URL based on stream type
if "Sub Stream" in stream_type:
    rtsp_url = rtsp_url.replace("subtype=0", "subtype=1")
else:
    rtsp_url = rtsp_url.replace("subtype=1", "subtype=0")

# Detection settings
st.sidebar.markdown("---")
st.sidebar.subheader("Detection")

model_size = st.sidebar.selectbox(
    "Model",
    ["yolov8n-seg.pt (Fastest)", "yolov8s-seg.pt (Balanced)", "yolov8m-seg.pt (Accurate)"]
)

model_map = {
    "yolov8n-seg.pt (Fastest)": "yolov8n-seg.pt",
    "yolov8s-seg.pt (Balanced)": "yolov8s-seg.pt",
    "yolov8m-seg.pt (Accurate)": "yolov8m-seg.pt"
}

confidence = st.sidebar.slider("Confidence", 0.1, 0.9, 0.4, 0.05)
show_timestamp = st.sidebar.checkbox("Show Timestamp", value=True)
show_count = st.sidebar.checkbox("Show Cow Count", value=True)

# Recording settings
st.sidebar.markdown("---")
st.sidebar.subheader("Recording")

enable_recording = st.sidebar.checkbox("Enable Recording", value=False)
record_on_detection = st.sidebar.checkbox("Record Only When Cows Detected", value=False)

# Alert settings
st.sidebar.markdown("---")
st.sidebar.subheader("Alerts")

enable_alerts = st.sidebar.checkbox("Enable Alerts", value=False)
alert_threshold = st.sidebar.number_input("Alert when cows > ", min_value=1, value=3)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model = load_model(model_map[model_size])

# Colors for instances
colors = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
]

# ============================================
# PROCESS FRAME FUNCTION
# ============================================
def process_frame(frame, model, confidence, show_timestamp, show_count):
    """Process a single frame with YOLO segmentation"""
    results = model(frame, conf=confidence, imgsz=640, verbose=False)

    cow_count = 0

    for result in results:
        if result.masks is None:
            continue

        masks = result.masks
        boxes = result.boxes

        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            if class_id != 0:  # COCO class 19 = cow
                continue

            cow_count += 1
            mask_points = masks.xy[i].astype(np.int32)
            color = colors[cow_count % len(colors)]

            # Draw filled mask
            overlay = frame.copy()
            cv2.fillPoly(overlay, [mask_points], color=color)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # Draw mask outline
            cv2.polylines(frame, [mask_points], True, color, 2)

            # Draw label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"Cow {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Draw cow count box
    if show_count:
        cv2.rectangle(frame, (10, 10), (220, 55), (0, 0, 0), -1)
        cv2.putText(frame, f"Cows: {cow_count}", (20, 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw timestamp
    if show_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame, cow_count

# ============================================
# MAIN TITLE
# ============================================
st.title("📹 CCTV Cow Monitoring System")
st.markdown("---")

# ============================================
# CONNECTION STATUS
# ============================================
col_status, col_controls = st.columns([3, 1])

with col_status:
    status_placeholder = st.empty()

with col_controls:
    # Connection test button
    if st.button("🔌 Test Connection", use_container_width=True):
        with st.spinner("Testing connection..."):
            test_cap = cv2.VideoCapture(rtsp_url)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    status_placeholder.success("✅ Connected to CCTV camera!")
                    h, w = frame.shape[:2]
                    st.info(f"Resolution: {w}x{h}")
                else:
                    status_placeholder.error("❌ Connected but cannot read frames")
            else:
                status_placeholder.error("❌ Cannot connect. Check URL and network.")
            test_cap.release()

# ============================================
# MAIN VIDEO DISPLAY
# ============================================
run = st.checkbox("▶ Start Monitoring", value=True)

if run:
    # Open camera
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        st.error("❌ Cannot connect to CCTV camera!")
        st.warning("Please check:")
        st.write("1. Camera is powered on and connected to WiFi")
        st.write("2. RTSP URL is correct")
        st.write("3. Camera and PC are on the same network")
        st.write("4. Username/password are correct")
    else:
        status_placeholder.success("🟢 Connected to CCTV camera")

        # Create display layout
        col_video, col_stats = st.columns([3, 1])

        with col_video:
            st.subheader("Live Feed")
            frame_placeholder = st.empty()

        with col_stats:
            st.subheader("Statistics")
            count_metric = st.metric("Cows Detected", "0")
            fps_metric = st.metric("FPS", "0")
            frame_metric = st.metric("Frames", "0")
            uptime_metric = st.metric("Uptime", "0s")

            # Alert area
            alert_area = st.empty()

            # Snapshot button
            if st.button("📸 Take Snapshot", use_container_width=True):
                st.info("Snapshot will be saved on next frame")

            # Recording status
            record_status = st.empty()

        # Video writer for recording
        video_writer = None
        recording_file = None

        # Stats tracking
        frame_count = 0
        start_time = time.time()
        fps = 0
        fps_counter = 0
        fps_time = time.time()
        last_cow_count = 0
        alert_triggered = False

        # Main monitoring loop
        while run:
            ret, frame = cap.read()

            if not ret:
                # Try to reconnect
                status_placeholder.warning("🟡 Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            # Resize frame
            frame = cv2.resize(frame, (1280, 720))

            # Process frame
            processed_frame, cow_count = process_frame(
                frame, model, confidence, show_timestamp, show_count
            )

            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()

            # Update stats
            frame_count += 1
            uptime = int(time.time() - start_time)
            uptime_str = f"{uptime//3600}h {(uptime%3600)//60}m {uptime%60}s"

            # Display frame
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Update metrics
            count_metric.metric("Cows Detected", cow_count)
            fps_metric.metric("FPS", fps)
            frame_metric.metric("Frames", frame_count)
            uptime_metric.metric("Uptime", uptime_str)

            # Handle recording
            if enable_recording:
                should_record = True
                if record_on_detection and cow_count == 0:
                    should_record = False

                if should_record and video_writer is None:
                    # Start recording
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    recording_file = f"cctv_recording_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(recording_file, fourcc, 15.0, (1280, 720))
                    record_status.info(f"🔴 Recording: {recording_file}")

                if should_record and video_writer is not None:
                    video_writer.write(processed_frame)

                if not should_record and video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    record_status.success(f"✅ Saved: {recording_file}")

            # Handle alerts
            if enable_alerts and cow_count >= alert_threshold and not alert_triggered:
                alert_area.warning(f"🚨 ALERT: {cow_count} cows detected!")
                alert_triggered = True

            if cow_count < alert_threshold:
                alert_triggered = False
                alert_area.empty()

            # Track changes
            if cow_count != last_cow_count:
                last_cow_count = cow_count

            # Control frame rate
            time.sleep(0.03)

        # Cleanup
        if video_writer is not None:
            video_writer.release()

        cap.release()
        status_placeholder.warning("⏹ Monitoring stopped")

        # Show recording file if exists
        if recording_file and os.path.exists(recording_file):
            st.success(f"📁 Recording saved: {recording_file}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        📹 CCTV Cow Monitoring System | Dahua Camera | YOLOv8 Segmentation
    </div>
""", unsafe_allow_html=True)