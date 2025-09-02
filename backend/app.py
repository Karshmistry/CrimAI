import os, cv2, pickle, sqlite3, time, numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from insightface.app import FaceAnalysis
import faiss
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# -------------------- Config --------------------
JWT_SECRET = "dev_secret_change_me"   # <== prod me env var se lo
JWT_EXP_MIN = 60 * 24                 # 24 hours
DB_PATH = "crimai.db"

UPLOADS_DIR = "uploads"
GALLERY_DIR = "gallery"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(GALLERY_DIR, exist_ok=True)

# -------------------- App -----------------------
app = Flask(__name__)
CORS(app)

# -------------------- DB Init -------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            filename TEXT,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------- Face + FAISS ----------------
# Load (or create) FAISS index from gallery embeddings
def load_index():
    if os.path.exists("faiss_gallery.pkl"):
        bundle = pickle.load(open("faiss_gallery.pkl", "rb"))
        return bundle["index"], bundle["ids"]
    # if missing, build empty index
    index = faiss.IndexFlatIP(512)
    return index, []

faiss_index, id_list = load_index()

# face model
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640,640))

def normalize(v):
    v = v.astype("float32")
    v /= np.linalg.norm(v) + 1e-9
    return v

def add_face_to_index(img_path, label_filename):
    """Compute embedding for image and add to FAISS + ids; persist to disk."""
    global faiss_index, id_list
    img = cv2.imread(img_path)
    if img is None:
        return False, "Image not readable"
    faces = face_app.get(img[:, :, ::-1])
    if not faces:
        return False, "No face found"
    emb = normalize(faces[0].embedding)
    # If index was empty, recreate with correct dim
    if faiss_index.ntotal == 0 and not isinstance(faiss_index, faiss.IndexIDMap):
        # already 512 dim set; no change needed
        pass
    faiss_index.add(emb[None, :])
    id_list.append(label_filename)
    pickle.dump({"index": faiss_index, "ids": id_list}, open("faiss_gallery.pkl", "wb"))
    return True, "added"

def detect_faces(img):
    faces = face_app.get(img[:, :, ::-1])
    results = []
    for f in faces:
        emb = normalize(f.embedding)
        if faiss_index.ntotal == 0:
            continue
        D, I = faiss_index.search(emb[None, :], 1)
        score, idx = float(D[0, 0]), int(I[0, 0])
        if score >= 0.80 and 0 <= idx < len(id_list):
            results.append({"name": id_list[idx], "score": round(score, 2)})
    return results

# -------------------- Auth Helpers ----------------
def generate_token(user_id, email):
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(minutes=JWT_EXP_MIN)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_token(auth_header):
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1]
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return data
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(fn):
    def wrapper(*args, **kwargs):
        payload = verify_token(request.headers.get("Authorization", ""))
        if not payload:
            return jsonify({"error": "Unauthorized"}), 401
        request.user = payload
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper

# -------------------- Routes: Auth ----------------
@app.route("/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not name or not email or not password:
        return jsonify({"error": "Missing fields"}), 400

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)",
            (name, email, generate_password_hash(password), datetime.utcnow().isoformat())
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "Email already exists"}), 409
    conn.close()

    token = generate_token(user_id, email)
    return jsonify({"message": "Signup success", "token": token})

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE email=?", (email,))
    row = cur.fetchone()
    conn.close()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401
    token = generate_token(row["id"], email)
    return jsonify({"message": "Login success", "token": token})

# -------------------- Routes: Criminal Add --------
@app.route("/criminals/add", methods=["POST"])
@require_auth
def add_criminal():
    # expects multipart/form-data: fields: name, file
    name = request.form.get("name", "").strip()
    file = request.files.get("file")
    if not name or not file:
        return jsonify({"error": "name and file are required"}), 400

    # safe filename like: <name>_<epoch>.jpg
    base = secure_filename(name).replace(" ", "_")
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    filename = f"{base}_{int(time.time())}{ext}"
    save_path = os.path.join(GALLERY_DIR, filename)
    file.save(save_path)

    ok, msg = add_face_to_index(save_path, filename)
    if not ok:
        # delete bad image
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"error": msg}), 400

    return jsonify({"message": "Criminal added", "filename": filename})

# -------------------- Routes: Detect --------------
@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("file")
    if not file:
        return jsonify({"results": [], "error": "file missing"}), 400

    filepath = os.path.join(UPLOADS_DIR, file.filename)
    file.save(filepath)

    results = []
    ext = file.filename.split(".")[-1].lower()
    if ext in ["jpg", "jpeg", "png"]:
        img = cv2.imread(filepath)
        results = detect_faces(img)
    elif ext in ["mp4", "avi", "mov", "mkv"]:
        cap = cv2.VideoCapture(filepath)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 10 == 0:
                res = detect_faces(frame)
                if res:
                    results.extend(res)
                    break
        cap.release()

    # Log detections to DB
    if results:
        conn = get_db()
        cur = conn.cursor()
        for r in results:
            cur.execute(
                "INSERT INTO logs(person_name,confidence,filename,created_at) VALUES(?,?,?,?)",
                (r["name"], r["score"], file.filename, datetime.utcnow().isoformat())
            )
        conn.commit()
        conn.close()

    return jsonify({"results": results})

# -------------------- Routes: Logs (Dashboard) ----
@app.route("/logs", methods=["GET"])
@require_auth
def get_logs():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT person_name, confidence, filename, created_at FROM logs ORDER BY id DESC LIMIT 200")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify({"logs": rows})

# -------------------- Root ------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "CrimAI API running", "routes": ["/auth/signup", "/auth/login", "/criminals/add", "/detect", "/logs"]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
