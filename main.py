"""
CURVED PARALLEL LINES DETECTION SYSTEM
Author: Yi Jiang
Date: 2025-02-02
Version:1.1.0

PSEUDOCODE:
-----------
1. START camera and setup processing parameters

2. FOR each frame:
   a. Capture frame from camera
   b. Convert to grayscale and apply blur
   c. Detect edges using threshold and Canny edge detection
   d. Find contours (shapes) in the image
   e. Filter to get the 2 longest contours (left and right lines)
   
3. FIT curves to the detected lines using polynomial fitting

4. CALCULATE center line by averaging left and right line points

5. DRAW results:
   - Red line for left
   - Blue line for right
   - Green line for center with forward arrow

6. DISPLAY video in web GUI and accept robot control commands
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
# DATABASE SETUP
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
# FASTAPI APPLICATION SETUP
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
# CONFIGURATION PARAMETERS
# ============================================================================

# Robot control state
controls = {"forward": False, "backward": False, "left": False, "right": False}

# Camera resolution
FRAME_W = 640
FRAME_H = 480

# Image processing parameters
BLUR_K = 11              # Gaussian blur kernel size (must be odd)
TH_BLOCK = 55            # Adaptive threshold block size
TH_C = 8                 # Adaptive threshold constant
MORPH_K = 7              # Morphological operation kernel size

# Contour filtering parameters
MIN_ARCLEN = 120.0       # Minimum arc length to be considered a line
MIN_AREA = 150           # Minimum area to filter out noise
MIN_ASPECT_RATIO = 3     # Minimum aspect ratio for line-like shapes

# Curve fitting parameters
POLY_DEGREE = 3          # Polynomial degree: 1=straight, 2=parabola, 3=S-curve
NUM_SAMPLES = 100        # Number of points to sample on fitted curve
SMOOTH_WIN = 9           # Smoothing window size

# Region of interest (crop area)
CROP_X = 120
CROP_Y = 140
CROP_W = 400
CROP_H = 260

# Drawing parameters
LINE_THICK = 3
ARROW_TIP_LENGTH = 0.25

# Canny edge detection thresholds
CANNY_LOW = 50
CANNY_HIGH = 150

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
latest_frame = None           # Processed frame (JPEG bytes)
latest_frame_raw = None       # Raw frame (JPEG bytes)
latest_frame_lock = threading.Lock()

LOG_FILE = "user_log.txt"
log_lock = threading.Lock()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log_event(message):
    """Log events with timestamp to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

# ============================================================================
# CAMERA INITIALIZATION
# ============================================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def preprocess_image(cropped_frame):
    """
    Preprocess image to detect lines without using color information
    
    Input: cropped BGR image
    Output: binary mask with detected lines
    
    Steps:
    1. Convert to grayscale
    2. Apply Gaussian blur
    3. Apply adaptive threshold
    4. Apply Canny edge detection
    5. Combine both methods
    6. Clean with morphological operations
    """
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
    
    # Method 1: Adaptive threshold (detects dark lines on light background)
    mask_adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, TH_BLOCK, TH_C
    )
    
    # Method 2: Canny edge detection (detects edges of tape)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    
    # Dilate edges to make them more continuous
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel_edge, iterations=2)
    
    # Combine both methods
    mask_combined = cv2.bitwise_or(mask_adaptive, edges)
    
    # Morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    mask_final = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask_final

def filter_contours(contours):
    """
    Filter contours to find the two most likely parallel lines
    
    Criteria:
    - Arc length must be above minimum
    - Area must be above minimum
    - Aspect ratio must indicate elongated shape
    """
    valid_contours = []
    
    for cnt in contours:
        arc_len = cv2.arcLength(cnt, closed=False)
        area = cv2.contourArea(cnt)
        
        # Basic filtering
        if arc_len < MIN_ARCLEN or area < MIN_AREA:
            continue
        
        # Check aspect ratio (lines should be elongated)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        
        if aspect_ratio > MIN_ASPECT_RATIO:
            valid_contours.append((arc_len, area, cnt))
    
    # If not enough contours found, lower the standards
    if len(valid_contours) < 2:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 100:
                arc_len = cv2.arcLength(cnt, closed=False)
                valid_contours.append((arc_len, area, cnt))
    
    # Sort by arc length and take top 2
    valid_contours = sorted(valid_contours, key=lambda x: x[0], reverse=True)[:2]
    
    return [cnt[2] for cnt in valid_contours]

def fit_polynomial_curve(contour_pts, degree=POLY_DEGREE):
    """
    Fit a polynomial curve to contour points
    
    Input: contour points array
    Output: (coefficients, parameter_values, fitted_points, mode)
    
    Mode: 'v' for vertical (x as function of y), 'h' for horizontal (y as function of x)
    """
    pts = contour_pts.reshape(-1, 2).astype(float)
    
    if len(pts) < degree + 1:
        return None, None, None, None
    
    # Sort by y-coordinate
    sorted_indices = np.argsort(pts[:, 1])
    pts_sorted = pts[sorted_indices]
    
    xs = pts_sorted[:, 0]
    ys = pts_sorted[:, 1]
    
    # Determine primary direction of line
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    
    try:
        if y_range > x_range * 1.2:  # Mostly vertical
            # Fit x = f(y)
            coeffs = np.polyfit(ys, xs, degree)
            y_smooth = np.linspace(ys.min(), ys.max(), NUM_SAMPLES)
            x_smooth = np.polyval(coeffs, y_smooth)
            fitted_points = np.column_stack([x_smooth, y_smooth])
            return coeffs, y_smooth, fitted_points, 'v'
        else:  # Mostly horizontal
            # Fit y = f(x)
            coeffs = np.polyfit(xs, ys, degree)
            x_smooth = np.linspace(xs.min(), xs.max(), NUM_SAMPLES)
            y_smooth = np.polyval(coeffs, x_smooth)
            fitted_points = np.column_stack([x_smooth, y_smooth])
            return coeffs, x_smooth, fitted_points, 'h'
    except:
        return None, None, None, None

def smooth_points(points, window=SMOOTH_WIN):
    """Apply moving average smoothing to curve points"""
    if len(points) < window:
        return points
    
    smoothed = np.copy(points).astype(float)
    for i in range(2):  # Smooth both x and y coordinates
        smoothed[:, i] = np.convolve(points[:, i], np.ones(window)/window, mode='same')
    
    return smoothed

def calculate_center_line(left_points, right_points):
    """
    Calculate center line between two parallel curves
    
    Method: Point-by-point interpolation
    """
    if left_points is None or right_points is None:
        return None
    
    if len(left_points) < 5 or len(right_points) < 5:
        return None
    
    # Resample both curves to same number of points
    min_len = min(len(left_points), len(right_points))
    
    left_resampled = np.array([
        left_points[int(i * len(left_points) / min_len)] 
        for i in range(min_len)
    ])
    right_resampled = np.array([
        right_points[int(i * len(right_points) / min_len)] 
        for i in range(min_len)
    ])
    
    # Calculate midpoints
    midpoints = (left_resampled + right_resampled) / 2.0
    
    # Smooth the center line
    midpoints_smooth = smooth_points(midpoints, SMOOTH_WIN)
    
    return midpoints_smooth

def draw_curve(img, points, color, thickness=LINE_THICK):
    """Draw a smooth curve on image"""
    if points is None or len(points) < 2:
        return
    
    points_int = points.astype(np.int32)
    cv2.polylines(img, [points_int], False, color, thickness, cv2.LINE_AA)

def draw_direction_arrow(img, points, color=(0, 255, 0), thickness=4):
    """Draw arrow showing forward direction on center line"""
    if points is None or len(points) < 20:
        return
    
    # Use middle section of line for arrow
    mid_idx = len(points) // 2
    start_point = tuple(points[mid_idx].astype(int))
    
    # Look ahead to determine direction
    forward_offset = min(20, len(points) - mid_idx - 1)
    end_idx = mid_idx + forward_offset
    end_point = tuple(points[end_idx].astype(int))
    
    # Draw arrow
    cv2.arrowedLine(img, start_point, end_point, color, 
                    thickness=thickness, tipLength=ARROW_TIP_LENGTH)
    
    # Add text label
    text_pos = (start_point[0] - 40, start_point[1] - 20)
    cv2.putText(img, "FORWARD", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, color, 2, cv2.LINE_AA)

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def process_and_update_frame():
    """
    Main video processing loop
    Runs continuously in background thread
    """
    global latest_frame, latest_frame_raw, cap

    if not cap.isOpened():
        print("ERROR: Camera failed to open")
        return

    print("Camera opened successfully, starting detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        
        # Save raw frame
        ret_raw, jpeg_raw = cv2.imencode('.jpg', frame)
        if ret_raw:
            with latest_frame_lock:
                latest_frame_raw = jpeg_raw.tobytes()

        # Define crop region
        x2 = min(CROP_X + CROP_W, FRAME_W)
        y2 = min(CROP_Y + CROP_H, FRAME_H)
        cropped = frame[CROP_Y:y2, CROP_X:x2].copy()
        h, w = cropped.shape[:2]

        # Step 1: Preprocess image
        mask = preprocess_image(cropped)

        # Step 2: Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 3: Filter contours
        selected_contours = filter_contours(contours)

        # Storage for full-frame coordinates
        left_curve_full = None
        right_curve_full = None
        center_curve_full = None

        if len(selected_contours) >= 2:
            cntA = selected_contours[0]
            cntB = selected_contours[1]

            # Determine which is left and which is right
            def mean_x(cnt):
                pts = cnt.reshape(-1, 2)
                return pts[:, 0].mean()

            if mean_x(cntA) <= mean_x(cntB):
                left_cnt, right_cnt = cntA, cntB
            else:
                left_cnt, right_cnt = cntB, cntA

            # Draw detected contours (for debugging)
            cv2.drawContours(cropped, [left_cnt], -1, (100, 100, 255), 1)
            cv2.drawContours(cropped, [right_cnt], -1, (255, 100, 100), 1)

            # Step 4: Fit polynomial curves
            coeffs_L, param_L, points_L, mode_L = fit_polynomial_curve(left_cnt, POLY_DEGREE)
            coeffs_R, param_R, points_R, mode_R = fit_polynomial_curve(right_cnt, POLY_DEGREE)

            # Draw left line (red)
            if points_L is not None:
                draw_curve(cropped, points_L, (0, 0, 255), LINE_THICK)
                left_curve_full = points_L + np.array([CROP_X, CROP_Y])

            # Draw right line (blue)
            if points_R is not None:
                draw_curve(cropped, points_R, (255, 0, 0), LINE_THICK)
                right_curve_full = points_R + np.array([CROP_X, CROP_Y])

            # Step 5: Calculate center line
            if points_L is not None and points_R is not None:
                center_points = calculate_center_line(points_L, points_R)
                
                if center_points is not None:
                    # Draw center line on cropped frame
                    draw_curve(cropped, center_points, (0, 255, 0), LINE_THICK + 1)
                    draw_direction_arrow(cropped, center_points, (0, 255, 0), thickness=3)
                    
                    # Convert to full frame coordinates
                    center_curve_full = center_points + np.array([CROP_X, CROP_Y])

        # Draw crop region boundary
        cv2.rectangle(frame, (CROP_X, CROP_Y), (x2, y2), (0, 255, 255), 2)
        
        # Draw curves on main frame
        if left_curve_full is not None:
            draw_curve(frame, left_curve_full, (0, 0, 255), LINE_THICK)
        
        if right_curve_full is not None:
            draw_curve(frame, right_curve_full, (255, 0, 0), LINE_THICK)
        
        if center_curve_full is not None:
            draw_curve(frame, center_curve_full, (0, 255, 0), LINE_THICK + 1)
            draw_direction_arrow(frame, center_curve_full, (0, 255, 0), thickness=4)

        # Add status text
        status_text = "Detecting: "
        if left_curve_full is not None and right_curve_full is not None:
            status_text += "Both Lines + Center OK"
            status_color = (0, 255, 0)
        elif left_curve_full is not None or right_curve_full is not None:
            status_text += "Partial Detection"
            status_color = (0, 165, 255)
        else:
            status_text += "No Lines Detected"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, status_color, 2, cv2.LINE_AA)

        # Encode and store processed frame
        ret2, jpeg = cv2.imencode('.jpg', frame)
        if ret2:
            with latest_frame_lock:
                latest_frame = jpeg.tobytes()

        cv2.waitKey(1)

# Start background processing thread
processing_thread = threading.Thread(target=process_and_update_frame, daemon=True)
processing_thread.start()

# ============================================================================
# API ENDPOINTS - VIDEO STREAMING
# ============================================================================

@app.get("/video_feed")
async def video_feed():
    """Stream processed video with line detection"""
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
    """Stream raw video without processing"""
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

# ============================================================================
# API ENDPOINTS - USER AUTHENTICATION
# ============================================================================

@app.post("/register")
async def register(user: User):
    """Register new user"""
    try:
        cursor.execute("INSERT INTO users (username,password) VALUES (?,?)", 
                      (user.username, user.password))
        conn.commit()
        log_event(f"REGISTER | user={user.username}")
        return {"message": "Registration successful"}
    except:
        raise HTTPException(status_code=400, detail="Username already exists or invalid")

@app.post("/login")
async def login(user: User):
    """Verify user credentials"""
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", 
                   (user.username, user.password))
    if cursor.fetchone():
        log_event(f"LOGIN success | user={user.username}")
        return {"message": "Login successful"}

    log_event(f"LOGIN failed | user={user.username}")
    raise HTTPException(status_code=401, detail="Invalid username/password")

# ============================================================================
# API ENDPOINTS - ROBOT CONTROL
# ============================================================================

@app.get("/status")
async def status():
    """Get current robot control state"""
    return controls

@app.post("/stop")
async def stop():
    """Stop all robot movements"""
    for k in controls:
        controls[k] = False
    log_event("CONTROL | stop")
    return {"message": "All movements stopped"}

@app.post("/{direction}")
async def move(direction: str):
    """Set robot movement direction"""
    if direction not in controls:
        raise HTTPException(status_code=400, detail="Invalid direction")
    for k in controls:
        controls[k] = False
    controls[direction] = True
    log_event(f"CONTROL | direction={direction}")
    return {direction: True}

# ============================================================================
# WEB GUI
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve web GUI interface"""
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Curved Line Detection - Robot Control</title>
<style>
  body { 
    margin: 0; 
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }
  
  #login-screen { 
    display: flex; 
    flex-direction: column; 
    align-items: center; 
    justify-content: center; 
    height: 100vh; 
    color: white;
  }
  
  #login-screen h1 {
    font-size: 2.5em;
    margin-bottom: 30px;
  }
  
  #login-screen input {
    margin: 8px;
    padding: 12px 20px;
    font-size: 16px;
    border: none;
    border-radius: 25px;
    width: 250px;
  }
  
  #login-screen button {
    margin: 10px 5px;
    padding: 12px 30px;
    font-size: 16px;
    border: none;
    border-radius: 25px;
    background: white;
    color: #667eea;
    cursor: pointer;
    font-weight: bold;
  }
  
  #login-screen button:hover {
    transform: translateY(-2px);
  }
  
  #app-screen { 
    display: none; 
    height: 100vh; 
    padding: 15px; 
    box-sizing: border-box; 
    background: #f5f5f5;
  }
  
  table { 
    width: 100%; 
    height: 100%; 
    border-collapse: collapse; 
    background: white;
    border-radius: 10px;
    overflow: hidden;
  }
  
  td { 
    border: 2px solid #ddd; 
    text-align: center; 
    vertical-align: middle; 
    padding: 15px;
  }
  
  img { 
    display: block; 
    margin: 0 auto; 
    border-radius: 8px;
  }
  
  .control-button {
    width: 60px;
    height: 60px;
    font-size: 24px;
    border: none;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .control-button:hover {
    transform: scale(1.1);
  }
  
  .control-button:active {
    transform: scale(0.95);
  }
  
  #console-log {
    height: 280px;
    overflow: auto;
    border: 2px solid #ddd;
    padding: 10px;
    font-family: monospace;
    font-size: 13px;
    background: #1e1e1e;
    color: #00ff00;
    border-radius: 8px;
  }
  
  h3 {
    color: #667eea;
    margin-top: 0;
  }
</style>
</head>
<body>

<div id="login-screen">
  <h1>Curved Line Detection System</h1>
  <p style="font-size: 1.2em; margin-bottom: 20px;">Robot Control Interface</p>
  <input id="username" placeholder="Username" />
  <input id="password" placeholder="Password" type="password" />
  <br>
  <div>
    <button onclick="registerUser()">Register</button>
    <button onclick="loginUser()">Login</button>
  </div>
  <p id="login-msg" style="margin-top: 20px; font-size: 1.1em;"></p>
</div>

<div id="app-screen">
  <table>
    <tr height="50%">
      <td width="50%">
        <h3>Processed Feed (Line Detection)</h3>
        <img src="/video_feed" width="520" height="340" alt="Processed Video"/>
      </td>

      <td width="50%">
        <h3>Robot Controls</h3>
        <table style="margin: 0 auto; border: none;">
          <tr align="center">
            <td style="border: none;"></td>
            <td style="border: none;"><button class="control-button" onclick="sendCommand('forward')">&#8593;</button></td>
            <td style="border: none;"></td>
          </tr>
          <tr align="center">
            <td style="border: none;"><button class="control-button" onclick="sendCommand('left')">&#8592;</button></td>
            <td style="border: none;"><button class="control-button" onclick="stopMotor()">&#9632;</button></td>
            <td style="border: none;"><button class="control-button" onclick="sendCommand('right')">&#8594;</button></td>
          </tr>
          <tr align="center">
            <td style="border: none;"></td>
            <td style="border: none;"><button class="control-button" onclick="sendCommand('backward')">&#8595;</button></td>
            <td style="border: none;"></td>
          </tr>
        </table>
        <p style="margin-top: 30px; color: #666;">
          <strong>Detection Legend:</strong><br>
          Red = Left Line<br>
          Blue = Right Line<br>
          Green = Center Line + Direction
        </p>
      </td>
    </tr>

    <tr height="50%">
      <td width="50%">
        <h3>Raw Camera Feed</h3>
        <img src="/video_feed_raw" width="520" height="340" alt="Raw Video"/>
      </td>
      <td width="50%">
        <h3>Console Log</h3>
        <pre id="console-log"></pre>
      </td>
    </tr>
  </table>
</div>

<script>
const API_BASE = "http://127.0.0.1:5000";

async function registerUser() {
  const u = document.getElementById("username").value.trim();
  const p = document.getElementById("password").value.trim();
  
  if (!u || !p) {
    document.getElementById("login-msg").innerText = "Please enter username and password";
    return;
  }
  
  try {
    const res = await fetch(API_BASE + "/register", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({username: u, password: p})
    });
    const data = await res.json();
    document.getElementById("login-msg").innerText = data.message || data.detail;
  } catch (e) {
    document.getElementById("login-msg").innerText = "Network error";
  }
}

async function loginUser() {
  const u = document.getElementById("username").value.trim();
  const p = document.getElementById("password").value.trim();
  
  if (!u || !p) {
    document.getElementById("login-msg").innerText = "Please enter username and password";
    return;
  }
  
  try {
    const res = await fetch(API_BASE + "/login", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({username: u, password: p})
    });
    const data = await res.json();
    document.getElementById("login-msg").innerText = data.message || data.detail;
    if (res.ok) {
      document.getElementById("login-screen").style.display = "none";
      document.getElementById("app-screen").style.display = "block";
      log("Login successful. System initialized.");
      log("Camera feeds are now streaming.");
      log("Curved line detection is active.");
    }
  } catch (e) {
    document.getElementById("login-msg").innerText = "Network error";
  }
}

async function sendCommand(direction) {
  log("Sending command: " + direction.toUpperCase());
  try {
    const res = await fetch(API_BASE + "/" + direction, {method: "POST"});
    const data = await res.json();
    log("Robot moving: " + direction.toUpperCase());
  } catch (e) {
    log("Error sending command: " + direction);
  }
}

async function stopMotor() {
  log("STOP command sent");
  try {
    const res = await fetch(API_BASE + "/stop", {method: "POST"});
    const data = await res.json();
    log("Robot stopped");
  } catch (e) {
    log("Error sending STOP");
  }
}

function log(msg) {
  const box = document.getElementById("console-log");
  const timestamp = new Date().toLocaleTimeString();
  box.textContent += "[" + timestamp + "] " + msg + "\\n";
  box.scrollTop = box.scrollHeight;
}

document.addEventListener('keydown', function(event) {
  if (document.getElementById("app-screen").style.display === "block") {
    switch(event.key) {
      case 'ArrowUp':
      case 'w':
      case 'W':
        sendCommand('forward');
        break;
      case 'ArrowDown':
      case 's':
      case 'S':
        sendCommand('backward');
        break;
      case 'ArrowLeft':
      case 'a':
      case 'A':
        sendCommand('left');
        break;
      case 'ArrowRight':
      case 'd':
      case 'D':
        sendCommand('right');
        break;
      case ' ':
        stopMotor();
        event.preventDefault();
        break;
    }
  }
});
</script>

</body>
</html>
""")

# ============================================================================
# SHUTDOWN HANDLER
# ============================================================================

@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on shutdown"""
    try:
        cap.release()
    except:
        pass

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Curved Line Detection Server...")
    print("Access GUI at: http://127.0.0.1:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
