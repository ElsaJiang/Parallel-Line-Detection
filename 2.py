"""
CURVED PARALLEL LINES DETECTION SYSTEM
Authors: [Your Names Here]
Date: 2025-02-02

PSEUDOCODE:
-----------
1. START camera and setup processing parameters

2. FOR each frame:
   a. Capture frame from camera
   b. Convert to grayscale
   c. Apply strong blur to reduce noise
   d. Use adaptive threshold to find dark regions
   e. Apply morphological operations to connect broken lines
   f. Find all contours (shapes)
   g. Filter contours by size and shape
   h. Select 2 longest contours as parallel lines
   
3. FIT polynomial curves to both lines

4. CALCULATE center line by averaging points

5. DRAW:
   - Red for left line
   - Blue for right line
   - Green for center line

6. DISPLAY in simple GUI
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import sqlite3
import cv2
import numpy as np
import threading
import asyncio
from datetime import datetime

# ============================================================================
# DATABASE
# ============================================================================
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")
conn.commit()

class User(BaseModel):
    username: str
    password: str

# ============================================================================
# FASTAPI
# ============================================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PARAMETERS - ADJUST THESE TO IMPROVE DETECTION
# ============================================================================

controls = {"forward": False, "backward": False, "left": False, "right": False}

# Camera
FRAME_W = 640
FRAME_H = 480

# Image processing - CRITICAL PARAMETERS
BLUR_K = 15              # Stronger blur to reduce noise
THRESH_BLOCKSIZE = 71    # Larger block for adaptive threshold
THRESH_C = 12            # Constant for threshold
MORPH_KERNEL = 9         # Larger morphology kernel to connect lines

# Contour filtering
MIN_CONTOUR_AREA = 200   # Minimum area to be considered
MIN_CONTOUR_LENGTH = 80  # Minimum arc length

# Curve fitting
POLY_DEGREE = 2          # Start with 2 (parabola) - simpler than 3
NUM_CURVE_POINTS = 60    # Number of points to sample
SMOOTHING_WINDOW = 5     # Smoothing window

# Region of Interest - ADJUST BASED ON YOUR CAMERA VIEW
ROI_X = 50
ROI_Y = 180
ROI_W = 540
ROI_H = 240

LINE_WIDTH = 3

# ============================================================================
# GLOBALS
# ============================================================================
latest_frame = None
latest_frame_raw = None
latest_frame_lock = threading.Lock()

LOG_FILE = "user_log.txt"
log_lock = threading.Lock()

def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

# ============================================================================
# CAMERA
# ============================================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_frame(frame):
    """
    Preprocess frame to detect dark tape lines
    Returns binary mask
    """
    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Strong blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
    
    # Adaptive threshold - detects dark regions
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        THRESH_BLOCKSIZE,
        THRESH_C
    )
    
    # Morphological operations to connect broken lines and remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    
    # Close to connect nearby components
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Open to remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return binary

def get_valid_contours(binary_mask):
    """
    Find and filter contours from binary mask
    Returns list of valid contours sorted by length
    """
    # Find all contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    
    for contour in contours:
        # Calculate properties
        area = cv2.contourArea(contour)
        arc_length = cv2.arcLength(contour, False)
        
        # Filter by area and length
        if area >= MIN_CONTOUR_AREA and arc_length >= MIN_CONTOUR_LENGTH:
            # Check if shape is elongated (line-like)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 0.001)
            
            # Lines should have aspect ratio > 2
            if aspect_ratio > 2:
                valid_contours.append((arc_length, contour))
    
    # Sort by arc length (longest first)
    valid_contours.sort(key=lambda x: x[0], reverse=True)
    
    return [cnt for _, cnt in valid_contours]

def fit_polynomial_to_contour(contour, degree=POLY_DEGREE):
    """
    Fit polynomial curve to contour points
    Returns array of curve points or None if fitting fails
    """
    if contour is None or len(contour) < 10:
        return None
    
    # Extract points
    points = contour.reshape(-1, 2).astype(float)
    
    # Sort points by y-coordinate (assume mostly vertical orientation)
    sorted_indices = np.argsort(points[:, 1])
    points = points[sorted_indices]
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Check orientation
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    
    try:
        if y_range > x_range:  # Mostly vertical
            # Fit x = f(y)
            coeffs = np.polyfit(y_coords, x_coords, degree)
            
            # Generate smooth curve
            y_smooth = np.linspace(y_coords.min(), y_coords.max(), NUM_CURVE_POINTS)
            x_smooth = np.polyval(coeffs, y_smooth)
            
            curve_points = np.column_stack([x_smooth, y_smooth])
            
        else:  # Mostly horizontal
            # Fit y = f(x)
            coeffs = np.polyfit(x_coords, y_coords, degree)
            
            # Generate smooth curve
            x_smooth = np.linspace(x_coords.min(), x_coords.max(), NUM_CURVE_POINTS)
            y_smooth = np.polyval(coeffs, x_smooth)
            
            curve_points = np.column_stack([x_smooth, y_smooth])
        
        return curve_points
        
    except:
        return None

def apply_smoothing(points, window=SMOOTHING_WINDOW):
    """Apply moving average smoothing to points"""
    if points is None or len(points) < window:
        return points
    
    smoothed = np.copy(points).astype(float)
    
    # Smooth x and y separately
    for i in range(2):
        smoothed[:, i] = np.convolve(
            points[:, i], 
            np.ones(window) / window, 
            mode='same'
        )
    
    return smoothed

def compute_centerline(left_points, right_points):
    """
    Compute centerline between two curves
    """
    if left_points is None or right_points is None:
        return None
    
    if len(left_points) < 5 or len(right_points) < 5:
        return None
    
    # Make same length
    n = min(len(left_points), len(right_points))
    
    # Resample
    left_resampled = []
    right_resampled = []
    
    for i in range(n):
        left_idx = int(i * len(left_points) / n)
        right_idx = int(i * len(right_points) / n)
        
        left_resampled.append(left_points[left_idx])
        right_resampled.append(right_points[right_idx])
    
    left_resampled = np.array(left_resampled)
    right_resampled = np.array(right_resampled)
    
    # Average
    center = (left_resampled + right_resampled) / 2.0
    
    # Smooth
    center = apply_smoothing(center, SMOOTHING_WINDOW)
    
    return center

def draw_curve_on_image(image, curve_points, color, thickness=LINE_WIDTH):
    """Draw a smooth curve on image"""
    if curve_points is None or len(curve_points) < 2:
        return
    
    # Convert to integer coordinates
    curve_int = curve_points.astype(np.int32)
    
    # Draw as polyline
    cv2.polylines(image, [curve_int], False, color, thickness, cv2.LINE_AA)

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_and_update_frame():
    """Main video processing loop"""
    global latest_frame, latest_frame_raw, cap

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print("Camera started successfully")
    print(f"Looking for lines in ROI: ({ROI_X}, {ROI_Y}, {ROI_W}, {ROI_H})")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize to standard size
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        
        # Save raw frame
        ret_raw, jpeg_raw = cv2.imencode('.jpg', frame)
        if ret_raw:
            with latest_frame_lock:
                latest_frame_raw = jpeg_raw.tobytes()

        # Extract ROI
        roi_x2 = min(ROI_X + ROI_W, FRAME_W)
        roi_y2 = min(ROI_Y + ROI_H, FRAME_H)
        roi = frame[ROI_Y:roi_y2, ROI_X:roi_x2].copy()

        # Preprocess ROI
        binary_mask = preprocess_frame(roi)
        
        # Find contours
        valid_contours = get_valid_contours(binary_mask)
        
        # Process if we have at least 2 contours
        left_curve_roi = None
        right_curve_roi = None
        center_curve_roi = None
        
        if len(valid_contours) >= 2:
            # Take top 2 contours
            contour_1 = valid_contours[0]
            contour_2 = valid_contours[1]
            
            # Determine left vs right based on x position
            def get_mean_x(cnt):
                pts = cnt.reshape(-1, 2)
                return pts[:, 0].mean()
            
            mean_x1 = get_mean_x(contour_1)
            mean_x2 = get_mean_x(contour_2)
            
            if mean_x1 < mean_x2:
                left_contour = contour_1
                right_contour = contour_2
            else:
                left_contour = contour_2
                right_contour = contour_1
            
            # Fit curves
            left_curve_roi = fit_polynomial_to_contour(left_contour, POLY_DEGREE)
            right_curve_roi = fit_polynomial_to_contour(right_contour, POLY_DEGREE)
            
            # Compute centerline
            if left_curve_roi is not None and right_curve_roi is not None:
                center_curve_roi = compute_centerline(left_curve_roi, right_curve_roi)
        
        # Draw on ROI
        if left_curve_roi is not None:
            draw_curve_on_image(roi, left_curve_roi, (0, 0, 255), LINE_WIDTH)
        
        if right_curve_roi is not None:
            draw_curve_on_image(roi, right_curve_roi, (255, 0, 0), LINE_WIDTH)
        
        if center_curve_roi is not None:
            draw_curve_on_image(roi, center_curve_roi, (0, 255, 0), LINE_WIDTH)
        
        # Draw ROI boundary on main frame
        cv2.rectangle(frame, (ROI_X, ROI_Y), (roi_x2, roi_y2), (128, 128, 128), 1)
        
        # Draw curves on main frame (adjust coordinates)
        if left_curve_roi is not None:
            left_curve_main = left_curve_roi + np.array([ROI_X, ROI_Y])
            draw_curve_on_image(frame, left_curve_main, (0, 0, 255), LINE_WIDTH)
        
        if right_curve_roi is not None:
            right_curve_main = right_curve_roi + np.array([ROI_X, ROI_Y])
            draw_curve_on_image(frame, right_curve_main, (255, 0, 0), LINE_WIDTH)
        
        if center_curve_roi is not None:
            center_curve_main = center_curve_roi + np.array([ROI_X, ROI_Y])
            draw_curve_on_image(frame, center_curve_main, (0, 255, 0), LINE_WIDTH)
        
        # Add status text
        num_found = len(valid_contours)
        status = f"Contours found: {num_found}"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        
        # Encode processed frame
        ret2, jpeg = cv2.imencode('.jpg', frame)
        if ret2:
            with latest_frame_lock:
                latest_frame = jpeg.tobytes()

        cv2.waitKey(1)

# Start processing thread
processing_thread = threading.Thread(target=process_and_update_frame, daemon=True)
processing_thread.start()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/video_feed")
async def video_feed():
    async def frame_stream():
        while True:
            data = None
            with latest_frame_lock:
                if latest_frame:
                    data = latest_frame
            if data:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       data + b"\r\n")
            await asyncio.sleep(0.03)
    return StreamingResponse(frame_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed_raw")
async def video_feed_raw():
    async def frame_stream():
        while True:
            data = None
            with latest_frame_lock:
                if latest_frame_raw:
                    data = latest_frame_raw
            if data:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       data + b"\r\n")
            await asyncio.sleep(0.03)
    return StreamingResponse(frame_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/register")
async def register(user: User):
    try:
        cursor.execute("INSERT INTO users (username,password) VALUES (?,?)", 
                      (user.username, user.password))
        conn.commit()
        log_event(f"REGISTER user={user.username}")
        return {"message": "Registration successful"}
    except:
        raise HTTPException(status_code=400, detail="Username exists")

@app.post("/login")
async def login(user: User):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", 
                   (user.username, user.password))
    if cursor.fetchone():
        log_event(f"LOGIN success user={user.username}")
        return {"message": "Login successful"}
    log_event(f"LOGIN failed user={user.username}")
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/status")
async def status():
    return controls

@app.post("/stop")
async def stop():
    for k in controls:
        controls[k] = False
    log_event("STOP")
    return {"message": "Stopped"}

@app.post("/{direction}")
async def move(direction: str):
    if direction not in controls:
        raise HTTPException(status_code=400, detail="Invalid")
    for k in controls:
        controls[k] = False
    controls[direction] = True
    log_event(f"MOVE {direction}")
    return {direction: True}

# ============================================================================
# SIMPLE GUI
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Curved Line Detection</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: Arial; background: #fff; color: #000; }
  
  #login-screen { 
    display: flex; 
    flex-direction: column; 
    align-items: center; 
    justify-content: center; 
    height: 100vh;
  }
  
  #login-screen h1 { font-size: 24px; margin-bottom: 20px; font-weight: normal; }
  #login-screen input { margin: 5px; padding: 8px 15px; font-size: 14px; border: 1px solid #000; background: #fff; width: 200px; }
  #login-screen button { margin: 5px; padding: 8px 20px; font-size: 14px; border: 1px solid #000; background: #fff; color: #000; cursor: pointer; }
  #login-screen button:hover { background: #000; color: #fff; }
  #login-msg { margin-top: 10px; font-size: 14px; }
  
  #app-screen { display: none; height: 100vh; padding: 10px; }
  table { width: 100%; height: 100%; border-collapse: collapse; border: 1px solid #000; }
  td { border: 1px solid #000; text-align: center; vertical-align: middle; padding: 10px; background: #fff; }
  img { display: block; margin: 0 auto; border: 1px solid #000; }
  h3 { font-size: 14px; font-weight: normal; margin-bottom: 10px; }
  
  .control-button { width: 50px; height: 50px; font-size: 20px; border: 1px solid #000; background: #fff; color: #000; cursor: pointer; margin: 2px; }
  .control-button:hover { background: #000; color: #fff; }
  
  #console-log { height: 250px; overflow: auto; border: 1px solid #000; padding: 10px; font-family: monospace; font-size: 12px; background: #fff; color: #000; text-align: left; }
</style>
</head>
<body>

<div id="login-screen">
  <h1>Curved Line Detection</h1>
  <input id="username" placeholder="Username" />
  <input id="password" placeholder="Password" type="password" />
  <div>
    <button onclick="registerUser()">Register</button>
    <button onclick="loginUser()">Login</button>
  </div>
  <p id="login-msg"></p>
</div>

<div id="app-screen">
  <table>
    <tr height="50%">
      <td width="50%">
        <h3>Processed Feed</h3>
        <img src="/video_feed" width="480" height="320" />
      </td>
      <td width="50%">
        <h3>Controls</h3>
        <table style="margin: 0 auto; border: none;">
          <tr>
            <td style="border: none;"></td>
            <td style="border: none;"><button class="control-button" onclick="sendCommand('forward')">&#8593;</button></td>
            <td style="border: none;"></td>
          </tr>
          <tr>
            <td style="border: none;"><button class="control-button" onclick="sendCommand('left')">&#8592;</button></td>
            <td style="border: none;"><button class="control-button" onclick="stopMotor()">&#9632;</button></td>
            <td style="border: none;"><button class="control-button" onclick="sendCommand('right')">&#8594;</button></td>
          </tr>
          <tr>
            <td style="border: none;"></td>
            <td style="border: none;"><button class="control-button" onclick="sendCommand('backward')">&#8595;</button></td>
            <td style="border: none;"></td>
          </tr>
        </table>
      </td>
    </tr>
    <tr height="50%">
      <td width="50%">
        <h3>Raw Feed</h3>
        <img src="/video_feed_raw" width="480" height="320" />
      </td>
      <td width="50%">
        <h3>Log</h3>
        <pre id="console-log"></pre>
      </td>
    </tr>
  </table>
</div>

<script>
const API = "http://127.0.0.1:5000";

async function registerUser() {
  const u = document.getElementById("username").value.trim();
  const p = document.getElementById("password").value.trim();
  if (!u || !p) { document.getElementById("login-msg").innerText = "Enter credentials"; return; }
  try {
    const res = await fetch(API + "/register", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({username: u, password: p}) });
    const data = await res.json();
    document.getElementById("login-msg").innerText = data.message || data.detail;
  } catch (e) { document.getElementById("login-msg").innerText = "Error"; }
}

async function loginUser() {
  const u = document.getElementById("username").value.trim();
  const p = document.getElementById("password").value.trim();
  if (!u || !p) { document.getElementById("login-msg").innerText = "Enter credentials"; return; }
  try {
    const res = await fetch(API + "/login", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({username: u, password: p}) });
    const data = await res.json();
    document.getElementById("login-msg").innerText = data.message || data.detail;
    if (res.ok) {
      document.getElementById("login-screen").style.display = "none";
      document.getElementById("app-screen").style.display = "block";
      log("Login successful");
    }
  } catch (e) { document.getElementById("login-msg").innerText = "Error"; }
}

async function sendCommand(dir) { log("Command: " + dir); try { await fetch(API + "/" + dir, {method: "POST"}); } catch (e) {} }
async function stopMotor() { log("STOP"); try { await fetch(API + "/stop", {method: "POST"}); } catch (e) {} }

function log(msg) {
  const box = document.getElementById("console-log");
  const time = new Date().toLocaleTimeString();
  box.textContent += "[" + time + "] " + msg + "\\n";
  box.scrollTop = box.scrollHeight;
}

document.addEventListener('keydown', function(e) {
  if (document.getElementById("app-screen").style.display === "block") {
    if (e.key === 'ArrowUp' || e.key === 'w') sendCommand('forward');
    if (e.key === 'ArrowDown' || e.key === 's') sendCommand('backward');
    if (e.key === 'ArrowLeft' || e.key === 'a') sendCommand('left');
    if (e.key === 'ArrowRight' || e.key === 'd') sendCommand('right');
    if (e.key === ' ') { stopMotor(); e.preventDefault(); }
  }
});
</script>

</body>
</html>
""")

@app.on_event("shutdown")
def shutdown_event():
    try:
        cap.release()
    except:
        pass

if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("CURVED LINE DETECTION SYSTEM")
    print("="*60)
    print(f"Camera: {FRAME_W}x{FRAME_H}")
    print(f"ROI: ({ROI_X}, {ROI_Y}) size {ROI_W}x{ROI_H}")
    print(f"Blur kernel: {BLUR_K}")
    print(f"Threshold block: {THRESH_BLOCKSIZE}")
    print(f"Morphology kernel: {MORPH_KERNEL}")
    print(f"Polynomial degree: {POLY_DEGREE}")
    print("="*60)
    print("Starting server at http://127.0.0.1:5000")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=5000)
