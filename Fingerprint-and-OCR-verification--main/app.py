import os
import json
import re
import cv2
import numpy as np
from scipy.spatial import distance
import fingerprint_feature_extractor
from PIL import Image
import easyocr
from flask import Flask, render_template, request, redirect, url_for, flash, session

app = Flask(__name__)
app.secret_key = "super_secret_ocr_app_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINGERPRINT_DIR = os.path.join(BASE_DIR, 'user_fingerprints')
DATASET_KTP_PATH = os.path.join(BASE_DIR, "dataset_ktp.json")
USERS_JSON_PATH = os.path.join(BASE_DIR, "users.json")
os.makedirs(FINGERPRINT_DIR, exist_ok=True)

ADMIN_FINGERPRINT = os.path.join(BASE_DIR, "Fingerprint 1.jpg")

# Load EasyOCR
reader = easyocr.Reader(['id', 'en'])

# ==============================================================================
# PERMANENT USER DATABASE HELPERS (USERS.JSON)
# ==============================================================================
def load_users():
    if not os.path.exists(USERS_JSON_PATH):
        default_data = {
            "admin": {
                "password": "admin123",
                "fingerprint_path": ADMIN_FINGERPRINT,
                "status": "approved",
                "role": "admin"
            }
        }
        save_users(default_data)
        return default_data
    with open(USERS_JSON_PATH, 'r') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return {}

def save_users(data):
    with open(USERS_JSON_PATH, 'w') as file:
        json.dump(data, file, indent=4)

ATTACK_COUNTER = 0

def load_ktp_dataset():
    if not os.path.exists(DATASET_KTP_PATH):
        return {}
    with open(DATASET_KTP_PATH, 'r') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return {}

def save_ktp_dataset(data):
    with open(DATASET_KTP_PATH, 'w') as file:
        json.dump(data, file, indent=4)

# ==============================================================================
# FINGERPRINT MATCHING LOGIC (OPTIMIZED FOR SPEED)
# ==============================================================================
def get_minutiae_points(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Downscale image if width exceeds 500px for instant processing
    max_width = 500
    h, w = img.shape[:2]
    if w > max_width:
        aspect_ratio = max_width / float(w)
        new_height = int(h * aspect_ratio)
        img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_AREA)

    terminations, bifurcations = fingerprint_feature_extractor.extract_minutiae_features(
        img, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False
    )

    points = []
    for pt in terminations:
        points.append([pt.locX, pt.locY])
    for pt in bifurcations:
        points.append([pt.locX, pt.locY])

    return np.array(points)

def verify_fingerprints(attempt_path, target_path, distance_thresh=15, match_ratio=0.55):
    pts1 = get_minutiae_points(attempt_path)
    pts2 = get_minutiae_points(target_path)

    if pts1 is None or pts2 is None or len(pts1) == 0 or len(pts2) == 0:
        return False, 0.0

    dist_matrix = distance.cdist(pts1, pts2, metric='euclidean')
    min_distances = np.min(dist_matrix, axis=1)
    matched_points = int(np.sum(min_distances < distance_thresh))

    total_pts = min(len(pts1), len(pts2))
    score = (matched_points / total_pts) if total_pts > 0 else 0.0
    return (score >= match_ratio), round(score * 100, 2)

# ==============================================================================
# EASYOCR & KTP GATEKEEPER
# ==============================================================================
def process_card_ocr(img_path, original_filename):
    image = Image.open(img_path).convert("RGB")
    img_np = np.array(image)
    results = reader.readtext(img_np, detail=0)
    
    full_text = " ".join(results).upper()
    nik_match = re.search(r'\b\d{16}\b', full_text)
    is_ktp_keyword = "REPUBLIK INDONESIA" in full_text or "PROVINSI" in full_text
    
    if nik_match or is_ktp_keyword:
        nik_hasil_scan = nik_match.group(0) if nik_match else "UNKNOWN_NIK"
        local_storage = load_ktp_dataset()
        
        if nik_hasil_scan in local_storage:
            return {
                "status": "rejected",
                "message": "🚫 Rejected, only authorized access can see!",
                "detail": "This ID Card data is already registered. Access blocked for data privacy."
            }
        else:
            nama_pendaftar = original_filename.split('.')[0].replace('_', ' ').upper()
            local_storage[nik_hasil_scan] = {
                "name": nama_pendaftar,
                "raw_text": full_text
            }
            save_ktp_dataset(local_storage)
            
            return {
                "status": "registered",
                "message": "🎉 ID Card Successfully Registered!",
                "detail": f"New ID Card (NIK: {nik_hasil_scan}) saved to dataset_ktp.json."
            }
    else:
        return {
            "status": "success",
            "message": "✅ Card Successfully Processed!",
            "results": results if results else ["No text detected in this image."]
        }

# ==============================================================================
# ROUTES
# ==============================================================================

@app.route("/")
def index():
    return render_template("landing.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        file = request.files.get("fingerprint")

        users_db = load_users()

        if username in users_db:
            flash("Username is already registered!", "danger")
            return redirect(url_for("register"))

        if not file or file.filename == "":
            flash("Upload a fingerprint photo for registration!", "warning")
            return redirect(url_for("register"))

        user_fp_path = os.path.join(FINGERPRINT_DIR, f"{username}_fp.jpg")
        
        # Read and resize uploaded fingerprint to 500px max width for instant processing
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        max_w = 500
        if img is not None and img.shape[1] > max_w:
            h, w = img.shape[:2]
            new_h = int(h * (max_w / float(w)))
            img = cv2.resize(img, (max_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(user_fp_path, img)
        else:
            file.seek(0)
            file.save(user_fp_path)

        users_db[username] = {
            "password": password,
            "fingerprint_path": user_fp_path,
            "status": "pending",
            "role": "user"
        }
        save_users(users_db)

        flash("Registration Successful! Awaiting Admin Approval before you can Login.", "info")
        return redirect(url_for("login_user"))

    return render_template("register.html")

@app.route("/login/user", methods=["GET", "POST"])
def login_user():
    global ATTACK_COUNTER
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        file = request.files.get("fingerprint")

        users_db = load_users()

        if username not in users_db or users_db[username]["password"] != password:
            flash("Invalid Username or Password!", "danger")
            return redirect(url_for("login_user"))

        user = users_db[username]

        if user["status"] != "approved":
            flash("Your account is still in PENDING status for Admin approval!", "warning")
            return redirect(url_for("login_user"))

        if not file or file.filename == "":
            flash("Upload your fingerprint photo for verification!", "warning")
            return redirect(url_for("login_user"))

        temp_path = os.path.join(BASE_DIR, "temp_login_attempt.jpg")
        file.save(temp_path)

        is_verified, score_pct = verify_fingerprints(temp_path, user["fingerprint_path"])

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not is_verified:
            ATTACK_COUNTER += 1
            flash(f"Biometric Verification Failed! Match Rate: {score_pct}%", "danger")
            return redirect(url_for("login_user"))

        session["user"] = username
        session["role"] = "user"
        return redirect(url_for("user_ocr"))

    return render_template("login_user.html", attack_count=ATTACK_COUNTER)

@app.route("/login/admin", methods=["GET", "POST"])
def login_admin():
    global ATTACK_COUNTER
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        file = request.files.get("fingerprint")

        users_db = load_users()

        if username != "admin" or password != users_db.get("admin", {}).get("password", "admin123"):
            flash("Invalid Admin Credentials!", "danger")
            return redirect(url_for("login_admin"))

        if not file or file.filename == "":
            flash("Upload Admin Fingerprint!", "warning")
            return redirect(url_for("login_admin"))

        temp_path = os.path.join(BASE_DIR, "temp_admin_attempt.jpg")
        file.save(temp_path)

        is_verified, score_pct = verify_fingerprints(temp_path, ADMIN_FINGERPRINT)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not is_verified:
            ATTACK_COUNTER += 1
            flash(f"Admin Access Denied! Match Rate: {score_pct}%", "danger")
            return redirect(url_for("login_admin"))

        session["user"] = "admin"
        session["role"] = "admin"
        return redirect(url_for("admin_dashboard"))

    return render_template("login_admin.html", attack_count=ATTACK_COUNTER)

@app.route("/admin/dashboard", methods=["GET", "POST"])
def admin_dashboard():
    if session.get("role") != "admin":
        flash("Admin Access Only!", "danger")
        return redirect(url_for("login_admin"))

    users_db = load_users()

    if request.method == "POST":
        action = request.form.get("action")
        selected_users = request.form.getlist("selected_users")

        if action == "approve_selected" and selected_users:
            for u in selected_users:
                if u in users_db:
                    users_db[u]["status"] = "approved"
            save_users(users_db)
            flash(f"Approved {len(selected_users)} user(s)!", "success")

        elif action == "approve_all":
            for u, data in users_db.items():
                if data["status"] == "pending":
                    data["status"] = "approved"
            save_users(users_db)
            flash("All pending users have been approved!", "success")

        elif action.startswith("approve_single_"):
            single_user = action.replace("approve_single_", "")
            if single_user in users_db:
                users_db[single_user]["status"] = "approved"
                save_users(users_db)
                flash(f"User {single_user} approved!", "success")

        return redirect(url_for("admin_dashboard"))

    pending_users = [{"username": u, "status": d["status"]} for u, d in users_db.items() if d["status"] == "pending"]
    approved_users = [{"username": u, "status": d["status"]} for u, d in users_db.items() if d["status"] == "approved" and u != "admin"]

    return render_template("admin_dashboard.html", pending=pending_users, approved=approved_users)

@app.route("/ocr", methods=["GET", "POST"])
def user_ocr():
    if "user" not in session or session.get("role") != "user":
        flash("Please log in as a user first.", "danger")
        return redirect(url_for("login_user"))

    ocr_outcome = None

    if request.method == "POST":
        file = request.files.get("ocr_image")
        if not file or file.filename == "":
            flash("Please select a card image!", "warning")
            return redirect(url_for("user_ocr"))

        temp_ocr_path = os.path.join(BASE_DIR, "temp_ocr_input.jpg")
        file.save(temp_ocr_path)

        ocr_outcome = process_card_ocr(temp_ocr_path, file.filename)

        if os.path.exists(temp_ocr_path):
            os.remove(temp_ocr_path)

    return render_template("ocr.html", username=session["user"], outcome=ocr_outcome)

@app.route("/logout")
def logout():
    session.clear()
    flash("Successfully logged out.", "info")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)