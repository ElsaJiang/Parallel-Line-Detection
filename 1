"""
CURVED PARALLEL LINES DETECTION SYSTEM

PSEUDOCODE:
-----------
1. START camera and setup processing parameters

2. FOR each frame:
   a. Capture frame from camera
   b. Convert to grayscale and apply blur
   c. Apply adaptive threshold to detect dark lines
   d. Find contours (curved shapes) in the image
   e. Filter to get the 2 longest contours (left and right lines)
   f. Fit polynomial curves to these contours
   
3. CALCULATE center line by averaging left and right curve points

4. DRAW results:
   - Red curve for left line
   - Blue curve for right line
   - Green curve for center line

5. DISPLAY video in simple GUI
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
# FASTAPI APP
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
# CONFIGURATION
# ============================================================================

controls = {"forward": False, "backward": False, "left": False, "right": False}

# Camera
FRAME_W = 640
FRAME_H = 480

# Image processing
BLUR_K = 9
TH_BLOCK = 51
TH_C = 10
MORPH_K = 5

# Contour filtering
MIN_ARCLEN = 100.0
MIN_AREA = 100

# Curve fitting
POLY_DEGREE = 3
NUM_SAMPLES = 80
SMOOTH_WIN = 7

# ROI
CROP_X = 80
CROP_Y = 150
CROP_W = 480
CROP_H = 250

# Drawing
LINE_THICK = 2

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
# DETECTION FUNCTIONS
# ============================================================================

def preprocess_image(frame):
    """Preprocess to detect dark lines on light background"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, TH_BLOCK, TH_C
    )
    
    # Morphology to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return thresh

def fit_curve(contour, degree=POLY_DEGREE):
    """Fit polynomial curve to contour points"""
    pts = contour.reshape(-1, 2).astype(float)
    
    if len(pts) < degree + 1:
        return None, None
    
    # Sort by y
    sorted_idx = np.argsort(pts[:, 1])
    pts_sorted = pts[sorted_idx]
    
    xs = pts_sorted[:, 0]
    ys = pts_sorted[:, 1]
    
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    
    try:
        if y_range > x_range * 1.2:  # Vertical line
            coeffs = np.polyfit(ys, xs, degree)
            y_smooth = np.linspace(ys.min(), ys.max(), NUM_SAMPLES)
            x_smooth = np.polyval(coeffs, y_smooth)
            points = np.column_stack([x_smooth, y_smooth])
            return points, 'v'
        else:  # Horizontal line
            coeffs = np.polyfit(xs, ys, degree)
            x_smooth = np.linspace(xs.min(), xs.max(), NUM_SAMPLES)
            y_smooth = np.polyval(coeffs, x_smooth)
            points = np.column_stack([x_smooth, y_smooth])
            return points, 'h'
    except:
        return None, None

def smooth_curve(points, window=SMOOTH_WIN):
    """Smooth curve points"""
    if points is None or len(points) < window:
        return points
    
    smoothed = np.copy(points).astype(float)
    for i in range(2):
        smoothed[:, i] = np.convolve(points[:, i], np.ones(window)/window, mode='same')
    
    return smoothed

def calculate_center(left_pts, right_pts):
    """Calculate center line between two curves"""
    if left_pts is None or right_pts is None:
        return None
    
    if len(left_pts) < 5 or len(right_pts) < 5:
        return None
    
    # Resample to same length
    min_len = min(len(left_pts), len(right_pts))
    
    left_resample = np.array([
        left_pts[int(i * len(left_pts) / min_len)] 
        for i in range(min_len)
    ])
    right_resample = np.array([
        right_pts[int(i * len(right_pts) / min_len)] 
        for i in range(min_len)
    ])
    
    # Average
    center = (left_resample + right_resample) / 2.0
    
    # Smooth
    center = smooth_curve(center, SMOOTH_WIN)
    
    return center

def draw_curve(img, points, color, thickness=LINE_THICK):
    """Draw curve on image"""
    if points is None or len(points) < 2:
        return
    
    points_int = points.astype(np.int32)
    cv2.polylines(img, [points_int], False, color, thickness, cv2.LINE_AA)

# ============================================================================
# PROCESSING LOOP
# ============================================================================

def process_and_update_frame():
    """Main processing loop"""
    global latest_frame, latest_frame_raw, cap

    if not cap.isOpened():
        print("ERROR: Camera failed to open")
        return

    print("Camera opened successfully")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        
        # Save raw
        ret_raw, jpeg_raw = cv2.imencode('.jpg', frame)
        if ret_raw:
            with latest_frame_lock:
                latest_frame_raw = jpeg_raw.tobytes()

        # Crop ROI
        x2 = min(CROP_X + CROP_W, FRAME_W)
        y2 = min(CROP_Y + CROP_H, FRAME_H)
        cropped = frame[CROP_Y:y2, CROP_X:x2].copy()
        h, w = cropped.shape[:2]

        # Preprocess
        mask = preprocess_image(cropped)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        valid = []
        for cnt in contours:
            arc_len = cv2.arcLength(cnt, closed=False)
            area = cv2.contourArea(cnt)
            if arc_len >= MIN_ARCLEN and area >= MIN_AREA:
                valid.append((arc_len, cnt))
        
        # Get top 2
        if len(valid) < 2:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= 50:
                    valid.append((cv2.arcLength(cnt, False), cnt))
        
        valid = sorted(valid, key=lambda x: x[0], reverse=True)[:2]

        left_curve = None
        right_curve = None
        center_curve = None

        if len(valid) >= 2:
            cnt1 = valid[0][1]
            cnt2 = valid[1][1]

            # Determine left/right
            def mean_x(c):
                p = c.reshape(-1, 2)
                return p[:, 0].mean()

            if mean_x(cnt1) <= mean_x(cnt2):
                left_cnt, right_cnt = cnt1, cnt2
            else:
                left_cnt, right_cnt = cnt2, cnt1

            # Fit curves
            left_pts, _ = fit_curve(left_cnt, POLY_DEGREE)
            right_pts, _ = fit_curve(right_cnt, POLY_DEGREE)

            # Draw on cropped
            if left_pts is not None:
                draw_curve(cropped, left_pts, (0, 0, 255), LINE_THICK)
                left_curve = left_pts + np.array([CROP_X, CROP_Y])

            if right_pts is not None:
                draw_curve(cropped, right_pts, (255, 0, 0), LINE_THICK)
                right_curve = right_pts + np.array([CROP_X, CROP_Y])

            # Calculate center
            if left_pts is not None and right_pts is not None:
                center_pts = calculate_center(left_pts, right_pts)
                
                if center_pts is not None:
                    draw_curve(cropped, center_pts, (0, 255, 0), LINE_THICK)
                    center_curve = center_pts + np.array([CROP_X, CROP_Y])

        # Draw ROI box
        cv2.rectangle(frame, (CROP_X, CROP_Y), (x2, y2), (200, 200, 200), 1)
        
        # Draw on main frame
        if left_curve is not None:
            draw_curve(frame, left_curve, (0, 0, 255), LINE_THICK)
        
        if right_curve is not None:
            draw_curve(frame, right_curve, (255, 0, 0), LINE_THICK)
        
        if center_curve is not None:
            draw_curve(frame, center_curve, (0, 255, 0), LINE_THICK)

        # Encode
        ret2, jpeg = cv2.imencode('.jpg', frame)
        if ret2:
            with latest_frame_lock:
                latest_frame = jpeg.tobytes()

        cv2.waitKey(1)

# Start thread
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
        log_event(f"REGISTER | user={user.username}")
        return {"message": "Registration successful"}
    except:
        raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/login")
async def login(user: User):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", 
                   (user.username, user.password))
    if cursor.fetchone():
        log_event(f"LOGIN success | user={user.username}")
        return {"message": "Login successful"}
    log_event(f"LOGIN failed | user={user.username}")
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/status")
async def status():
    return controls

@app.post("/stop")
async def stop():
    for k in controls:
        controls[k] = False
    log_event("CONTROL | stop")
    return {"message": "Stopped"}

@app.post("/{direction}")
async def move(direction: str):
    if direction not in controls:
        raise HTTPException(status_code=400, detail="Invalid direction")
    for k in controls:
        controls[k] = False
    controls[direction] = True
    log_event(f"CONTROL | direction={direction}")
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
<title>Line Detection</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { 
    font-family: Arial, sans-serif;
    background: #ffffff;
    color: #000000;
  }
  
  #login-screen { 
    display: flex; 
    flex-direction: column; 
    align-items: center; 
    justify-content: center; 
    height: 100vh;
  }
  
  #login-screen h1 {
    font-size: 24px;
    margin-bottom: 20px;
    font-weight: normal;
  }
  
  #login-screen input {
    margin: 5px;
    padding: 8px 15px;
    font-size: 14px;
    border: 1px solid #000000;
    background: #ffffff;
    width: 200px;
  }
  
  #login-screen button {
    margin: 5px;
    padding: 8px 20px;
    font-size: 14px;
    border: 1px solid #000000;
    background: #ffffff;
    color: #000000;
    cursor: pointer;
  }
  
  #login-screen button:hover {
    background: #000000;
    color: #ffffff;
  }
  
  #app-screen { 
    display: none; 
    height: 100vh; 
    padding: 10px;
  }
  
  table { 
    width: 100%; 
    height: 100%; 
    border-collapse: collapse;
    border: 1px solid #000000;
  }
  
  td { 
    border: 1px solid #000000;
    text-align: center; 
    vertical-align: middle; 
    padding: 10px;
    background: #ffffff;
  }
  
  img { 
    display: block; 
    margin: 0 auto;
    border: 1px solid #000000;
  }
  
  h3 {
    font-size: 14px;
    font-weight: normal;
    margin-bottom: 10px;
  }
  
  .control-button {
    width: 50px;
    height: 50px;
    font-size: 20px;
    border: 1px solid #000000;
    background: #ffffff;
    color: #000000;
    cursor: pointer;
    margin: 2px;
  }
  
  .control-button:hover {
    background: #000000;
    color: #ffffff;
  }
  
  #console-log {
    height: 250px;
    overflow: auto;
    border: 1px solid #000000;
    padding: 10px;
    font-family: monospace;
    font-size: 12px;
    background: #ffffff;
    color: #000000;
    text-align: left;
  }
  
  #login-msg {
    margin-top: 10px;
    font-size: 14px;
  }
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
  if (!u || !p) {
    document.getElementById("login-msg").innerText = "Enter username and password";
    return;
  }
  try {
    const res = await fetch(API + "/register", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({username: u, password: p})
    });
    const data = await res.json();
    document.getElementById("login-msg").innerText = data.message || data.detail;
  } catch (e) {
    document.getElementById("login-msg").innerText = "Error";
  }
}

async function loginUser() {
  const u = document.getElementById("username").value.trim();
  const p = document.getElementById("password").value.trim();
  if (!u || !p) {
    document.getElementById("login-msg").innerText = "Enter username and password";
    return;
  }
  try {
    const res = await fetch(API + "/login", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({username: u, password: p})
    });
    const data = await res.json();
    document.getElementById("login-msg").innerText = data.message || data.detail;
    if (res.ok) {
      document.getElementById("login-screen").style.display = "none";
      document.getElementById("app-screen").style.display = "block";
      log("Login successful");
    }
  } catch (e) {
    document.getElementById("login-msg").innerText = "Error";
  }
}

async function sendCommand(dir) {
  log("Command: " + dir);
  try {
    await fetch(API + "/" + dir, {method: "POST"});
  } catch (e) {}
}

async function stopMotor() {
  log("STOP");
  try {
    await fetch(API + "/stop", {method: "POST"});
  } catch (e) {}
}

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
    print("Starting Curved Line Detection Server...")
    print("Access at: http://127.0.0.1:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
