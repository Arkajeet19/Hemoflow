from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from authlib.integrations.flask_client import OAuth
from functools import wraps
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import re
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.urandom(24)  # Change this for production

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb+srv://dwaipayan:t39BVvL915Pz6fE1@cluster0.94vw2zv.mongodb.net/bloodbank?retryWrites=true&w=majority&appName=Cluster0"
mongo = PyMongo(app)

# Test MongoDB connection on startup
try:
    mongo.db.command('ping')
    print("MongoDB connected successfully!")
except Exception as e:
    print("MongoDB connection failed:", str(e))

# Configure Google OAuth
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='435268235242-gg3jekh1b22uehck4nt211bl3uhlm35j.apps.googleusercontent.com',
    client_secret='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',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'email profile'},
)

# Try XGBoost; fall back to RandomForest if not installed
USE_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    USE_XGB = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------- Config ----------------------
ISSUES_CSV    = os.getenv("ISSUES_CSV",    "fact_issues_slim.csv")
INVENTORY_CSV = os.getenv("INVENTORY_CSV", "fact_inventory_slim.csv")
CALENDAR_CSV  = os.getenv("CALENDAR_CSV",  "dim_calendar_slim.csv")
BANKS_CSV     = os.getenv("BANKS_CSV",     "blood_banks.csv")

MIN_HISTORY_ROWS = 30          # forgiving for slim dataset
N_TAIL_FOR_LAGS  = 30

# Optional alert policy
POLICY = {
    "warning_ratio": 0.10  # WARNING when stock_after < 10% of opening stock for the day
}

# ---------------------- Globals ----------------------
model_registry   = {}  # (H, BG, COMP) -> model
feature_columns  = [
    "lag1","lag7","lag14","roll7","roll14",
    "is_weekend","is_holiday","month","year",
    "is_monsoon","dengue_index","accident_index",
    "closing_stock"
]
last_known_state = {}  # (H, BG, COMP) -> {"tail_frame": df[date,closing_stock,units_issued], "last_stock": float}
CALENDAR_DF      = None
TRAINED_COUNT    = 0
BLOOD_BANKS      = []  # list of dicts: hospital_id,name,lat,lon,address

# ---------------------- Helpers ----------------------
def log(msg): print(msg, flush=True)

def assert_csv_exists(path):
    if not os.path.exists(path):
        log(f"[ERROR] CSV not found: {path}")
        return False
    log(f"[OK] Found CSV: {path}")
    return True

def build_training_frame(issues_df, inv_df, cal_df, hospital_id, blood_group, component):
    df = issues_df[
        (issues_df["hospital_id"] == hospital_id) &
        (issues_df["blood_group"] == blood_group) &
        (issues_df["component"]   == component)
    ].copy()
    if df.empty:
        return None, None, None, "no_issues_rows"

    # Merge calendar
    df = df.merge(cal_df, on="date", how="left")

    # Merge inventory (closing stock for that hospital)
    inv_slice = inv_df[inv_df["hospital_id"] == hospital_id][
        ["date","blood_group","component","closing_stock"]
    ].copy()
    df = df.merge(inv_slice, on=["date","blood_group","component"], how="left")

    # Fill missing stock for training
    if "closing_stock" not in df.columns:
        df["closing_stock"] = 0.0
    df["closing_stock"] = df["closing_stock"].fillna(0.0)

    df = df.sort_values("date")
    # Lags/rolls
    df["lag1"]   = df["units_issued"].shift(1)
    df["lag7"]   = df["units_issued"].shift(7)
    df["lag14"]  = df["units_issued"].shift(14)
    df["roll7"]  = df["units_issued"].shift(1).rolling(7).mean()
    df["roll14"] = df["units_issued"].shift(1).rolling(14).mean()

    pre_drop = len(df)
    df = df.dropna(subset=feature_columns + ["units_issued"])
    post_drop = len(df)
    if post_drop < MIN_HISTORY_ROWS:
        return None, None, None, f"too_few_rows_after_drop (pre={pre_drop}, post={post_drop})"

    X = df[feature_columns].copy()
    y = df["units_issued"].copy()
    meta_tail = df[["date","units_issued","closing_stock"]].copy()
    return meta_tail, X, y, None

def train_all_models():
    """Train per-key models; populate registries; log skips."""
    global CALENDAR_DF, TRAINED_COUNT

    ok1 = assert_csv_exists(ISSUES_CSV)
    ok2 = assert_csv_exists(INVENTORY_CSV)
    ok3 = assert_csv_exists(CALENDAR_CSV)
    if not (ok1 and ok2 and ok3):
        log("[FATAL] One or more CSVs missing. Training aborted.")
        return 0

    issues   = pd.read_csv(ISSUES_CSV, parse_dates=["date"])
    inv      = pd.read_csv(INVENTORY_CSV, parse_dates=["date"])
    calendar = pd.read_csv(CALENDAR_CSV,  parse_dates=["date"])
    CALENDAR_DF = calendar.copy()

    # sanity
    for name, df in [("issues", issues), ("inventory", inv), ("calendar", calendar)]:
        cols = ", ".join(df.columns.astype(str).tolist())
        log(f"[SANITY] {name} rows={len(df)} cols=[{cols}]")

    keys = issues[["hospital_id","blood_group","component"]].drop_duplicates().values.tolist()
    log(f"[INFO] Unique keys to train: {len(keys)}")

    count = 0
    for (hospital_id, blood_group, component) in keys:
        meta_tail, X, y, err = build_training_frame(issues, inv, CALENDAR_DF, hospital_id, blood_group, component)
        if err:
            log(f"[SKIP] {hospital_id}/{blood_group}/{component}: {err}")
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            if USE_XGB:
                model = XGBRegressor(
                    n_estimators=250, learning_rate=0.08, max_depth=6,
                    subsample=0.9, colsample_bytree=0.9, random_state=42
                )
            else:
                model = RandomForestRegressor(n_estimators=250, max_depth=10, random_state=42, n_jobs=-1)

            model.fit(X_train, y_train)

            key = (hospital_id, blood_group, component)
            model_registry[key] = model

            tail = meta_tail.sort_values("date").tail(N_TAIL_FOR_LAGS).copy()
            last_stock = float(tail["closing_stock"].iloc[-1]) if len(tail) else 0.0
            last_known_state[key] = {
                "tail_frame": tail[["date","closing_stock","units_issued"]].copy(),
                "last_stock": last_stock,
            }
            count += 1
            log(f"[OK] Trained: {hospital_id}/{blood_group}/{component} (rows={len(X)})")
        except Exception as e:
            log(f"[ERR] Train failed {hospital_id}/{blood_group}/{component}: {e}")

    TRAINED_COUNT = count
    log(f"[ML] Trained models: {count}")
    if count == 0:
        log("[HINT] If this is 0, check CSV names/paths & columns.")
    return count

def calendar_row_for_date(d):
    row = CALENDAR_DF[CALENDAR_DF["date"] == d]
    if len(row) == 1:
        return row.iloc[0].to_dict()
    return {
        "date": d,
        "is_weekend": 1 if d.weekday() >= 5 else 0,
        "is_holiday": 0,
        "month": d.month,
        "year": d.year,
        "is_monsoon": 1 if d.month in [6,7,8,9] else 0,
        "dengue_index": 4.0,
        "accident_index": 2.5
    }

def forecast_key(
    hospital_id,
    blood_group,
    component,
    horizon_days=14,
    stock_on_hand=None,
    expected_receipts_daily=0.0,
    # simple replenishment policy
    target_stock=40.0,          # try to maintain this level
    lead_time_days=2,           # order arrives after N days
    max_daily_receipt=25.0,     # cap inbound per day
    track_backlog=True          # accumulate unmet demand
):
    """
    Recursive forecasting that:
      - predicts daily demand
      - consumes stock
      - adds (a) user-provided receipts per day and (b) policy-based replenishment after a lead time
      - optionally tracks backlog (unmet demand) when stock runs out
    """
    key = (hospital_id, blood_group, component)
    if key not in model_registry:
        return {"error": f"No trained model for {key}. Check /health and training logs."}

    model = model_registry[key]
    state = last_known_state.get(key)
    if state is None or "tail_frame" not in state:
        return {"error": f"No tail state for {key}. Re-train server."}

    tail = state["tail_frame"].copy()
    current_stock = float(stock_on_hand) if stock_on_hand is not None else float(state["last_stock"])

    last_date = tail["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days)

    # Queue of receipts scheduled by policy: date -> amount
    receipts_queue = {}

    backlog = 0.0  # unmet demand carried forward (if track_backlog=True)
    out = []

    for d in future_dates:
        cal = calendar_row_for_date(d)

        # Lags from updated tail
        lag1  = float(tail["units_issued"].iloc[-1])
        lag7  = float(tail["units_issued"].iloc[-7])  if len(tail) >= 7  else lag1
        lag14 = float(tail["units_issued"].iloc[-14]) if len(tail) >= 14 else lag7
        roll7  = float(tail["units_issued"].iloc[-7:].mean())
        roll14 = float(tail["units_issued"].iloc[-14:].mean())

        # Predict base demand for the day
        feat_row = {
            "lag1": lag1, "lag7": lag7, "lag14": lag14,
            "roll7": roll7, "roll14": roll14,
            "is_weekend": int(cal["is_weekend"]),
            "is_holiday": int(cal.get("is_holiday", 0)),
            "month": int(cal.get("month", d.month)),
            "year": int(cal.get("year", d.year)),
            "is_monsoon": int(cal.get("is_monsoon", 1 if d.month in [6,7,8,9] else 0)),
            "dengue_index": float(cal.get("dengue_index", 4.0)),
            "accident_index": float(cal.get("accident_index", 2.5)),
            "closing_stock": float(current_stock),
        }
        X_row = pd.DataFrame([feat_row])[feature_columns]
        yhat  = float(model.predict(X_row)[0])

        # If tracking backlog, today's total demand we try to fulfill includes backlog
        total_demand = yhat + (backlog if track_backlog else 0.0)

        # Receipts for today = user baseline + scheduled policy receipts (if any), both capped
        user_receipts = float(expected_receipts_daily or 0.0)
        policy_receipts = float(receipts_queue.get(d.date(), 0.0))
        arriving = min(max_daily_receipt, user_receipts + policy_receipts)

        # Opening stock BEFORE consumption (show this in UI)
        stock_open = current_stock

        # Fulfill demand
        fulfilled = min(stock_open + arriving, total_demand)
        gap_units = max(0.0, total_demand - (stock_open + arriving))  # unmet today

        # Update stock after consumption
        stock_after = max(0.0, stock_open + arriving - total_demand)

        # Update backlog
        if track_backlog:
            backlog = gap_units
        else:
            backlog = 0.0

        # Reorder rule: if stock_after < target, schedule a receipt for d + lead_time_days
        if target_stock is not None and lead_time_days is not None:
            arrival_date = (d + pd.Timedelta(days=int(lead_time_days))).date()
            needed = max(0.0, target_stock - stock_after)
            if needed > 0:
                receipts_queue[arrival_date] = receipts_queue.get(arrival_date, 0.0) + needed

        # Alert based on stock_after
        if stock_after <= 0:
            alert = "CRITICAL"
        elif stock_after < max(1.0, stock_open) * POLICY["warning_ratio"]:
            alert = "WARNING"
        else:
            alert = "SAFE"

        out.append({
            "date": d.strftime("%Y-%m-%d"),
            "hospital_id": hospital_id,
            "blood_group": blood_group,
            "component": component,
            "forecast_units": round(yhat, 2),
            "backlog_in": round((total_demand - yhat), 2) if track_backlog else 0.0,
            "stock_open": round(stock_open, 2),
            "arriving_receipts": round(arriving, 2),
            "gap_units": round(gap_units, 2),
            "stock_after": round(stock_after, 2),
            "alert": alert
        })

        # Roll forward lags (use yhat, not total_demand)
        tail = pd.concat([tail, pd.DataFrame([{
            "date": d, "closing_stock": stock_open, "units_issued": yhat
        }])], ignore_index=True)

        current_stock = stock_after  # next day opens with stock_after

    return out

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def load_blood_banks():
    """
    Expects columns: hospital_id,name,lat,lon,address
    Returns list[dict].
    """
    banks = []
    if os.path.exists(BANKS_CSV):
        dfb = pd.read_csv(BANKS_CSV)
        for _,r in dfb.iterrows():
            try:
                banks.append({
                    "hospital_id": str(r["hospital_id"]),
                    "name": str(r["name"]),
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "address": str(r.get("address","")),
                })
            except Exception:
                continue
    else:
        # Fallback examples — replace with your real CSV data
        banks = [
            {"hospital_id":"H001","name":"NRS Medical College","lat":22.5645,"lon":88.3702,"address":"138 AJC Bose Rd, Kolkata"},
            {"hospital_id":"H002","name":"Medical College Kolkata","lat":22.5733,"lon":88.3639,"address":"88 College St, Kolkata"},
        ]
    return banks

# ---------------------- Train on boot ----------------------
BLOOD_BANKS = load_blood_banks()
train_all_models()

# ---------------------- Routes ----------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON or form data:
    {
      "hospital_id": "H001",
      "blood_group": "O+",
      "component": "PRBC",
      "horizon_days": 14,
      "stock_on_hand": 20,               # optional
      "expected_receipts_daily": 2,      # optional
      "target_stock": 40,                # optional policy
      "lead_time_days": 2,               # optional policy
      "max_daily_receipt": 25,           # optional policy
      "track_backlog": true              # optional policy
    }
    """
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        hospital_id = (data.get("hospital_id") or "").strip()
        blood_group = (data.get("blood_group") or "").strip()
        component   = (data.get("component") or "").strip()
        horizon     = int(data.get("horizon_days", 14))

        stock_on_hand = data.get("stock_on_hand", None)
        stock_on_hand = float(stock_on_hand) if stock_on_hand not in (None, "",) else None

        expected_receipts_daily = data.get("expected_receipts_daily", None)
        expected_receipts_daily = float(expected_receipts_daily) if expected_receipts_daily not in (None,"") else 0.0

        # New policy knobs (all optional)
        target_stock = data.get("target_stock", None)
        target_stock = float(target_stock) if target_stock not in (None,"") else 40.0

        lead_time_days = data.get("lead_time_days", None)
        lead_time_days = int(lead_time_days) if lead_time_days not in (None,"") else 2

        max_daily_receipt = data.get("max_daily_receipt", None)
        max_daily_receipt = float(max_daily_receipt) if max_daily_receipt not in (None,"") else 25.0

        track_backlog = data.get("track_backlog", "true")
        track_backlog = str(track_backlog).lower() in ("1","true","yes","y")

        if not (hospital_id and blood_group and component):
            return jsonify({"error": "hospital_id, blood_group, component are required"}), 400

        result = forecast_key(
            hospital_id, blood_group, component,
            horizon_days=horizon,
            stock_on_hand=stock_on_hand,
            expected_receipts_daily=expected_receipts_daily,
            target_stock=target_stock,
            lead_time_days=lead_time_days,
            max_daily_receipt=max_daily_receipt,
            track_backlog=track_backlog
        )
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 404

        return jsonify({"forecasts": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/availability", methods=["GET"])
def availability():
    """
    Query params:
      blood_group (required)
      component   (optional, default PRBC)
      lat, lon    (required for "near me")
      radius_km   (optional, default 12)
      horizon_days (optional, default 1)
      expected_receipts_daily (optional, default 0)
      target_stock, lead_time_days, max_daily_receipt (optional policy knobs)
    Returns a list of nearby banks with predicted availability (units) and alert.
    """
    q = request.args
    blood_group = (q.get("blood_group") or "").strip()
    component   = (q.get("component") or "PRBC").strip()
    try:
        lat = float(q.get("lat")); lon = float(q.get("lon"))
    except Exception:
        return jsonify({"error":"lat and lon are required"}), 400

    radius_km = float(q.get("radius_km", 12) or 12)
    horizon   = int(q.get("horizon_days", 1) or 1)

    expected_receipts_daily = float(q.get("expected_receipts_daily", 0) or 0)
    target_stock      = float(q.get("target_stock", 40) or 40)
    lead_time_days    = int(q.get("lead_time_days", 2) or 2)
    max_daily_receipt = float(q.get("max_daily_receipt", 25) or 25)

    if not blood_group:
        return jsonify({"error":"blood_group is required"}), 400

    results = []
    for bank in BLOOD_BANKS:
        d_km = haversine_km(lat, lon, bank["lat"], bank["lon"])
        if d_km > radius_km:
            continue

        key = (bank["hospital_id"], blood_group, component)
        if key not in model_registry:
            # Skip banks we don't have a trained model for that (BG, component)
            continue

        sim_rows = forecast_key(
            hospital_id=bank["hospital_id"],
            blood_group=blood_group,
            component=component,
            horizon_days=horizon,
            stock_on_hand=None,  # use last closing stock by default
            expected_receipts_daily=expected_receipts_daily,
            target_stock=target_stock,
            lead_time_days=lead_time_days,
            max_daily_receipt=max_daily_receipt,
            track_backlog=True
        )
        if isinstance(sim_rows, dict) and sim_rows.get("error"):
            continue

        row = sim_rows[-1] if sim_rows else None
        if row:
            results.append({
                "hospital_id": bank["hospital_id"],
                "name": bank["name"],
                "lat": bank["lat"],
                "lon": bank["lon"],
                "address": bank.get("address",""),
                "distance_km": round(d_km, 2),
                "forecast_units": row["forecast_units"],
                "stock_open": row["stock_open"],
                "arriving_receipts": row["arriving_receipts"],
                "stock_after": row["stock_after"],
                "gap_units": row["gap_units"],
                "alert": row["alert"]
            })

    # Sort by distance, then by higher stock_after
    results.sort(key=lambda r: (r["distance_km"], -r["stock_after"]))
    return jsonify({"nearby": results})

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "use_xgboost": USE_XGB,
        "models_loaded": len(model_registry),
        "trained_count": TRAINED_COUNT,
        "trained_keys_sample": [
            {"hospital_id": k[0], "blood_group": k[1], "component": k[2]}
            for k in list(model_registry.keys())[:10]
        ]
    })

@app.route("/predict-form")
def predict_form():
    return render_template("predict_form.html")

# ---- Other pages you already had ----
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Email validation
def is_valid_email(email):
    regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    return re.search(regex, email)

@app.route('/check-email', methods=['POST'])
def check_email():
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email required'}), 400
    
    user = mongo.db.users.find_one({'email': email})
    return jsonify({'exists': bool(user)})

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    
    if not all([name, email, password]):
        return jsonify({'error': 'All fields are required'}), 400
    
    if not is_valid_email(email):
        return jsonify({'error': 'Invalid email format'}), 400
    
    if mongo.db.users.find_one({'email': email}):
        return jsonify({'error': 'Email already exists'}), 400
    
    hashed_password = generate_password_hash(password)
    
    user_data = {
        'name': name,
        'email': email,
        'password': hashed_password,
        'auth_method': 'email',
        'profile_pic': None
    }
    
    user_id = mongo.db.users.insert_one(user_data).inserted_id
    session['user_id'] = str(user_id)
    return jsonify({'success': True, 'redirect': url_for('profile')})

# Email/Password Login
@app.route('/login', methods=['POST'])
def email_login():
    email = request.form.get('email')
    password = request.form.get('password')
    
    user = mongo.db.users.find_one({'email': email})
    if not user or not check_password_hash(user.get('password', ''), password):
        return jsonify({'error': 'Invalid email or password'}), 401
    
    session['user_id'] = str(user['_id'])
    return jsonify({'success': True, 'redirect': url_for('profile')})

# Google OAuth routes
@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_authorized', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def google_authorized():
    token = google.authorize_access_token()
    user_info = google.get('userinfo').json()
    
    # Check if user exists
    user = mongo.db.users.find_one({'email': user_info['email']})
    
    if not user:
        # Create new user
        user_data = {
            'name': user_info['name'],
            'email': user_info['email'],
            'auth_method': 'google',
            'google_id': user_info['id'],
            'profile_pic': user_info['picture'].replace('=s96-c', '=s400-c')
        }
        user_id = mongo.db.users.insert_one(user_data).inserted_id
    else:
        user_id = user['_id']
    
    session['user_id'] = str(user_id)
    return redirect(url_for('profile'))

@app.route('/profile')
@login_required
def profile():
    user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
    return render_template('profile.html', user=user)

@app.route('/user')
@login_required
def get_user():
    user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
    return jsonify({
        'displayName': user.get('name', ''),
        'email': user.get('email', ''),
        'photoUrl': user.get('profile_pic', '')
    })

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/aboutus.html')
def aboutus():
    return render_template('aboutus.html')

@app.route('/registration.html')
def registration():
    return render_template('registration.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/whydonateblood.html')
def whydonateblood():
    return render_template('whydonateblood.html')

@app.route('/canyoudonate.html')
def canyoudonate():
    return render_template('canyoudonate.html')

@app.route('/emergency.html')
def emergency():
    return render_template('emergency.html')

@app.route('/admin-login.html')
def admin_login():
    return render_template('admin-login.html')

@app.route('/admin.html')
def admin():
    return render_template('admin.html')

@app.route('/helpdesk.html')
def helpdesk():
    return render_template('helpdesk.html')

@app.route('/prebook.html')
def prebook():
    return render_template('prebook.html')

@app.route('/ambulance.html')
def ambulance():
    return render_template('ambulance.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
