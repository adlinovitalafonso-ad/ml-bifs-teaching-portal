import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
import os
import sqlite3
import json
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    silhouette_score, log_loss
)
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    TFIDF_AVAILABLE = True
except Exception:
    TFIDF_AVAILABLE = False

st.set_page_config(page_title="Machine Learning for BIFS", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------
# PORTAL CONFIGURATION
# ---------------------------------------------------------------------
# For deployment, put these in .streamlit/secrets.toml instead of leaving
# the fallback values in the source file.
def get_secret(name, default):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

ACCESS_CODE = get_secret("ML_PORTAL_ACCESS_CODE", None)
ADMIN_CODE = get_secret("ML_PORTAL_ADMIN_CODE", None)

if not ACCESS_CODE or not ADMIN_CODE:
    st.error("Portal access codes are not configured.")
    st.stop()
ACTIVE_FROM = str(get_secret("ML_PORTAL_ACTIVE_FROM", "2026-01-01T00:00"))
ACTIVE_UNTIL = str(get_secret("ML_PORTAL_ACTIVE_UNTIL", "2027-12-31T23:59"))
PORTAL_TZ = ZoneInfo("Asia/Kolkata")
DB_PATH = Path(__file__).with_name("ml_bifs_analytics.db")
MAX_ACTIVE_USERS = int(get_secret("ML_PORTAL_MAX_USERS", "30"))
SESSION_TIMEOUT_MINUTES = int(get_secret("ML_PORTAL_SESSION_TIMEOUT_MINUTES", "10"))


def parse_portal_dt(value):
    return datetime.fromisoformat(value).replace(tzinfo=PORTAL_TZ)


def portal_is_active():
    now = datetime.now(PORTAL_TZ)
    try:
        start = parse_portal_dt(ACTIVE_FROM)
        end = parse_portal_dt(ACTIVE_UNTIL)
        return start <= now <= end, now, start, end
    except Exception:
        return True, now, None, None


def db_connect():
    return sqlite3.connect(DB_PATH)


def init_analytics_db():
    with db_connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                email TEXT,
                city TEXT,
                timezone TEXT,
                locale TEXT,
                ip_address TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS event_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                email TEXT,
                section TEXT,
                event TEXT,
                details TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                email TEXT,
                rating INTEGER,
                useful_section TEXT,
                comments TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS active_sessions (
                session_id TEXT PRIMARY KEY,
                email TEXT,
                city TEXT,
                login_time TEXT,
                last_seen TEXT,
                logout_time TEXT,
                status TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS session_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                email TEXT,
                city TEXT,
                login_time TEXT,
                logout_time TEXT,
                duration_minutes REAL,
                end_reason TEXT
            )
        """)
        con.commit()


def log_access():
    try:
        timezone = getattr(st.context, "timezone", None)
        locale = getattr(st.context, "locale", None)
        ip_address = getattr(st.context, "ip_address", None)
    except Exception:
        timezone = locale = ip_address = None

    with db_connect() as con:
        con.execute(
            """INSERT INTO access_log
               (timestamp, email, city, timezone, locale, ip_address)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(PORTAL_TZ).isoformat(timespec="seconds"),
                st.session_state.get("user_email", ""),
                st.session_state.get("user_city", ""),
                str(timezone or ""),
                str(locale or ""),
                str(ip_address or "")
            )
        )
        con.commit()


def log_event(section, event, details=None):
    email = st.session_state.get("user_email", "")
    fingerprint = f"{section}|{event}|{json.dumps(details, sort_keys=True, default=str)}"
    key = f"_logged_{abs(hash(fingerprint))}"
    if st.session_state.get(key):
        return
    st.session_state[key] = True

    with db_connect() as con:
        con.execute(
            """INSERT INTO event_log
               (timestamp, email, section, event, details)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now(PORTAL_TZ).isoformat(timespec="seconds"),
                email,
                section,
                event,
                json.dumps(details or {}, default=str)
            )
        )
        con.commit()


def save_feedback(rating, useful_section, comments):
    with db_connect() as con:
        con.execute(
            """INSERT INTO feedback
               (timestamp, email, rating, useful_section, comments)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now(PORTAL_TZ).isoformat(timespec="seconds"),
                st.session_state.get("user_email", ""),
                int(rating),
                useful_section,
                comments
            )
        )
        con.commit()



def _now_iso():
    return datetime.now(PORTAL_TZ).isoformat(timespec="seconds")


def cleanup_expired_sessions():
    now = datetime.now(PORTAL_TZ)
    cutoff = now.timestamp() - (SESSION_TIMEOUT_MINUTES * 60)

    with db_connect() as con:
        rows = con.execute(
            """SELECT session_id, email, city, login_time, last_seen
               FROM active_sessions
               WHERE status = 'active'"""
        ).fetchall()

        expired = []
        for sid, email, city, login_time, last_seen in rows:
            try:
                last_dt = datetime.fromisoformat(last_seen)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=PORTAL_TZ)
                if last_dt.timestamp() < cutoff:
                    expired.append((sid, email, city, login_time, last_seen))
            except Exception:
                pass

        for sid, email, city, login_time, last_seen in expired:
            try:
                login_dt = datetime.fromisoformat(login_time)
                end_dt = datetime.fromisoformat(last_seen)
                if login_dt.tzinfo is None:
                    login_dt = login_dt.replace(tzinfo=PORTAL_TZ)
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=PORTAL_TZ)
                duration = max(0.0, (end_dt - login_dt).total_seconds() / 60)
            except Exception:
                duration = None

            con.execute(
                """INSERT INTO session_history
                   (session_id, email, city, login_time, logout_time, duration_minutes, end_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (sid, email, city, login_time, last_seen, duration, "timeout")
            )
            con.execute(
                "DELETE FROM active_sessions WHERE session_id = ?",
                (sid,)
            )

        con.commit()


def current_active_users():
    cleanup_expired_sessions()
    with db_connect() as con:
        value = con.execute(
            "SELECT COUNT(*) FROM active_sessions WHERE status = 'active'"
        ).fetchone()[0]
    return int(value)


def register_session(email, city):
    cleanup_expired_sessions()
    sid = str(uuid.uuid4())
    now = _now_iso()

    with db_connect() as con:
        con.execute(
            """INSERT INTO active_sessions
               (session_id, email, city, login_time, last_seen, logout_time, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sid, email, city, now, now, None, "active")
        )
        con.commit()

    st.session_state.portal_session_id = sid
    st.session_state.portal_login_time = now
    return sid


def heartbeat():
    sid = st.session_state.get("portal_session_id")
    if not sid:
        return

    with db_connect() as con:
        con.execute(
            """UPDATE active_sessions
               SET last_seen = ?, status = 'active'
               WHERE session_id = ?""",
            (_now_iso(), sid)
        )
        con.commit()


def logout_current_session(reason="logout"):
    sid = st.session_state.get("portal_session_id")
    if not sid:
        return

    now = datetime.now(PORTAL_TZ)
    with db_connect() as con:
        row = con.execute(
            """SELECT email, city, login_time, last_seen
               FROM active_sessions
               WHERE session_id = ?""",
            (sid,)
        ).fetchone()

        if row:
            email, city, login_time, last_seen = row
            try:
                login_dt = datetime.fromisoformat(login_time)
                if login_dt.tzinfo is None:
                    login_dt = login_dt.replace(tzinfo=PORTAL_TZ)
                duration = max(0.0, (now - login_dt).total_seconds() / 60)
            except Exception:
                duration = None

            con.execute(
                """INSERT INTO session_history
                   (session_id, email, city, login_time, logout_time, duration_minutes, end_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    sid, email, city, login_time,
                    now.isoformat(timespec="seconds"),
                    duration, reason
                )
            )
            con.execute(
                "DELETE FROM active_sessions WHERE session_id = ?",
                (sid,)
            )
            con.commit()


def session_elapsed_minutes():
    login_time = st.session_state.get("portal_login_time")
    if not login_time:
        return 0.0
    try:
        start = datetime.fromisoformat(login_time)
        if start.tzinfo is None:
            start = start.replace(tzinfo=PORTAL_TZ)
        return max(
            0.0,
            (datetime.now(PORTAL_TZ) - start).total_seconds() / 60
        )
    except Exception:
        return 0.0


def access_gate():
    active, now, start, end = portal_is_active()
    if not active:
        st.error("This teaching portal is currently outside its active access window.")
        if start and end:
            st.write(
                f"Active window: {start.strftime('%d %b %Y %H:%M')} "
                f"to {end.strftime('%d %b %Y %H:%M')} IST."
            )
        st.stop()

    active_now = current_active_users()

    if st.session_state.get("authenticated", False):
        heartbeat()
        return

    st.title("Machine Learning for BIFS — Teaching Portal")

    available = max(0, MAX_ACTIVE_USERS - active_now)
    if active_now >= MAX_ACTIVE_USERS:
        st.error(
            f"Portal currently full: {active_now} / {MAX_ACTIVE_USERS} users active."
        )
        st.caption(
            f"Sessions inactive for more than {SESSION_TIMEOUT_MINUTES} minutes are released automatically."
        )
        st.stop()
    elif available <= 3:
        st.warning(
            f"Current usage: {active_now} / {MAX_ACTIVE_USERS} users active — {available} place(s) available."
        )
    else:
        st.info(
            f"Current usage: {active_now} / {MAX_ACTIVE_USERS} users active — {available} places available."
        )

    st.caption(
        "Access is restricted for teaching use. Please enter your details and the access code."
    )
    st.caption(
        "Concept and teaching design: Prof. Adlino Afonso | Developed with AI assistance."
    )

    with st.form("portal_login"):
        email = st.text_input("Email ID")
        city = st.text_input("City / location")
        code = st.text_input("Access code", type="password")
        consent = st.checkbox(
            "I consent to my email, stated city, access time, browser timezone/locale, "
            "and portal activity being recorded for teaching analytics."
        )
        submitted = st.form_submit_button("Enter portal")

    if submitted:
        if not email or "@" not in email:
            st.error("Please enter a valid email ID.")
        elif not city.strip():
            st.error("Please enter your city/location.")
        elif not consent:
            st.error("Consent is required because this portal records teaching analytics.")
        elif code != ACCESS_CODE:
            st.error("Incorrect access code.")
        else:
            # Re-check capacity immediately before admitting this user
            if current_active_users() >= MAX_ACTIVE_USERS:
                st.error(
                    f"Portal has just reached maximum capacity ({MAX_ACTIVE_USERS} users). "
                    "Please try again shortly."
                )
                st.stop()

            st.session_state.authenticated = True
            st.session_state.user_email = email.strip()
            st.session_state.user_city = city.strip()
            register_session(email.strip(), city.strip())
            log_access()
            st.rerun()

    st.stop()


init_analytics_db()
access_gate()


st.title("Machine Learning for BIFS — Interactive Laboratory")
st.caption("Interactive demonstrations for regression, classification, clustering, PCA and semi-supervised learning.")

# ---------------------------------------------------------------------
# BUILT-IN DATA
# ---------------------------------------------------------------------
@st.cache_data
def regression_data(seed=42, n=300):
    rng = np.random.default_rng(seed)
    income = rng.uniform(250, 1500, n)
    debt = rng.uniform(0, 500, n)
    score = rng.uniform(500, 850, n)
    employment = rng.uniform(0, 20, n)
    smoker = rng.choice(["No", "Yes"], size=n, p=[0.78, 0.22])
    credit_limit = (
        40 + 0.20 * income - 0.12 * debt + 0.10 * (score - 500)
        + 2.2 * employment + 22 * (smoker == "No").astype(int)
        - 0.00008 * (income - 900) ** 2 + rng.normal(0, 25, n)
    )
    return pd.DataFrame({
        "Annual Income": income,
        "Existing Debt": debt,
        "Credit Score": score,
        "Employment Duration": employment,
        "Smoker": smoker,
        "Recommended Credit Limit": credit_limit
    })

@st.cache_data
def classification_data(seed=42, n=2500, difficulty=0.45):
    class_sep = max(0.45, 1.6 - difficulty)
    X, y = make_classification(
        n_samples=n, n_features=6, n_informative=5, n_redundant=1,
        n_clusters_per_class=2, weights=[0.78, 0.22],
        class_sep=class_sep, flip_y=min(0.08, 0.01 + 0.02 * difficulty),
        random_state=seed
    )
    cols = ["Income", "Debt", "Credit Utilization", "Repayment Delay", "Account Age", "Transaction Activity"]
    df = pd.DataFrame(X, columns=cols)
    df["Channel"] = np.where(df["Transaction Activity"] > 0, "Digital", "Branch")
    df["Default"] = y
    return df

@st.cache_data
def clustering_data(seed=42, n=600):
    X, _ = make_blobs(
        n_samples=n,
        centers=[(-2.5, -1.5), (0.5, 2.0), (3.0, -0.5)],
        cluster_std=[0.8, 0.9, 0.75], random_state=seed
    )
    return pd.DataFrame(X, columns=["Income Score", "Spending Score"])

@st.cache_data
def pca_customer_data(seed=42, n=500):
    rng = np.random.default_rng(seed)
    value = rng.normal(0, 1, n)
    engagement = rng.normal(0, 1, n)
    segment = np.where(value > 0.6, "High Value", np.where(value < -0.6, "Low Value", "Middle"))
    return pd.DataFrame({
        "Income": 80 + 25 * value + rng.normal(0, 8, n),
        "Balance": 50 + 20 * value + rng.normal(0, 8, n),
        "Monthly Spend": 30 + 12 * value + 4 * engagement + rng.normal(0, 5, n),
        "Number of Products": 2 + 0.8 * value + 0.5 * engagement + rng.normal(0, 0.6, n),
        "Transactions": 40 + 12 * engagement + rng.normal(0, 7, n),
        "Digital Usage": 50 + 15 * engagement + rng.normal(0, 8, n),
        "Customer Segment": segment
    })

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()

def categorical_cols(df):
    return [c for c in df.columns if c not in numeric_cols(df)]

def low_cardinality_numeric(df, max_unique=10):
    return [c for c in numeric_cols(df) if df[c].nunique(dropna=True) <= max_unique]

def id_like_columns(df):
    out = []
    for c in df.columns:
        name = c.lower().strip()
        if name in {"id", "rowid", "index", "unnamed: 0"} or name.endswith("_id"):
            out.append(c)
    return out

def upload_or_builtin(key, built_in_df):
    source = st.radio("Data source", ["Built-in demonstration data", "Upload my CSV"], horizontal=True, key=f"{key}_source")
    if source == "Built-in demonstration data":
        return built_in_df.copy(), False
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"{key}_uploader")
    if uploaded is None:
        st.info("Choose a CSV file to continue.")
        st.stop()
    try:
        df, detected_encoding = read_uploaded_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
    st.success(
        f"Loaded {df.shape[0]} rows × {df.shape[1]} columns "
        f"(encoding: {detected_encoding})."
    )
    st.dataframe(df.head(), use_container_width=True)
    return df, True

def binary_target(y_raw):
    vals = list(pd.Series(y_raw.dropna().unique()))
    if len(vals) != 2:
        return None, None
    mapping = {vals[0]: 0, vals[1]: 1}
    return y_raw.map(mapping), mapping

def make_preprocessor(num_features, cat_features, scale_numeric=False):
    transformers = []
    if num_features:
        transformers.append(("num", StandardScaler() if scale_numeric else "passthrough", num_features))
    if cat_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def feature_names_from_preprocessor(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []

def encoding_panel(df, target=None, key="enc"):
    st.subheader("Variable Type / Encoding Check")
    cats = [c for c in categorical_cols(df) if c != target]
    low_num = [c for c in low_cardinality_numeric(df) if c != target and c not in id_like_columns(df)]
    if cats:
        st.info("Categorical columns detected: " + ", ".join(cats) + ". If selected as predictors, the app will one-hot encode them automatically.")
    else:
        st.caption("No text/category predictor columns detected.")
    if low_num:
        st.warning(
            "Low-cardinality numeric columns detected: " + ", ".join(low_num) +
            ". These may be coded categories or ordinal variables. Check their meaning before treating them as continuous."
        )

def metric_dict(y_true, y_pred, y_prob=None):
    d = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        d["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    return d

def metric_table(y_true, y_pred, y_prob=None):
    d = metric_dict(y_true, y_pred, y_prob)
    return pd.DataFrame({"Metric": list(d.keys()), "Score": list(d.values())})

def show_cm(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"]); ax.set_yticklabels(["Actual 0", "Actual 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def threshold_plot(y_true, y_prob):
    thresholds = np.arange(0.10, 0.91, 0.05)
    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        rows.append({
            "Threshold": t,
            "Precision": precision_score(y_true, yp, zero_division=0),
            "Recall": recall_score(y_true, yp, zero_division=0),
            "F1": f1_score(y_true, yp, zero_division=0),
            "Accuracy": accuracy_score(y_true, yp)
        })
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots()
    for c in ["Precision", "Recall", "F1", "Accuracy"]:
        ax.plot(tab["Threshold"], tab[c], label=c)
    ax.set_xlabel("Classification Threshold"); ax.set_ylabel("Metric")
    ax.set_title("Threshold Trade-off"); ax.legend()
    st.pyplot(fig)
    best = tab.loc[tab["F1"].idxmax()]
    st.caption(f"Highest F1 in this sweep occurs near threshold {best['Threshold']:.2f} (F1={best['F1']:.3f}). Business costs may justify another threshold.")

def gradient_learning_plot(model, X_train, y_train, X_test, y_test):
    train_loss, test_loss = [], []
    for p_train, p_test in zip(model.staged_predict_proba(X_train), model.staged_predict_proba(X_test)):
        train_loss.append(log_loss(y_train, p_train)); test_loss.append(log_loss(y_test, p_test))
    fig, ax = plt.subplots()
    ax.plot(range(1, len(train_loss)+1), train_loss, label="Training log loss")
    ax.plot(range(1, len(test_loss)+1), test_loss, label="Test log loss")
    ax.set_xlabel("Boosting Iteration / Number of Trees"); ax.set_ylabel("Log Loss")
    ax.set_title("Gradient Boosting Learning Curve"); ax.legend()
    st.pyplot(fig)
    best_iter = int(np.argmin(test_loss) + 1)
    st.caption(f"Lowest test log loss occurs around iteration {best_iter}. If training loss falls while test loss rises, overfitting is beginning.")

def decision_tree_depth_plot(X_train, y_train, X_test, y_test, prep):
    depths = list(range(1, 13)); train_f1, test_f1 = [], []
    for d in depths:
        pipe = Pipeline([("prep", prep), ("model", DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=42))])
        pipe.fit(X_train, y_train)
        train_f1.append(f1_score(y_train, pipe.predict(X_train), zero_division=0))
        test_f1.append(f1_score(y_test, pipe.predict(X_test), zero_division=0))
    fig, ax = plt.subplots()
    ax.plot(depths, train_f1, label="Training F1"); ax.plot(depths, test_f1, label="Test F1")
    ax.set_xlabel("Maximum Depth"); ax.set_ylabel("F1"); ax.set_title("Decision Tree Depth Tuning")
    ax.legend(); st.pyplot(fig)

def random_forest_tree_plot(X_train, y_train, X_test, y_test, prep, depth):
    counts = [10, 25, 50, 100, 150, 200, 300]
    vals = []
    for n in counts:
        pipe = Pipeline([("prep", prep), ("model", RandomForestClassifier(n_estimators=n, max_depth=depth, class_weight="balanced", random_state=42, n_jobs=-1))])
        pipe.fit(X_train, y_train)
        vals.append(f1_score(y_test, pipe.predict(X_test), zero_division=0))
    fig, ax = plt.subplots()
    ax.plot(counts, vals, marker="o")
    ax.set_xlabel("Number of Trees"); ax.set_ylabel("Test F1"); ax.set_title("Random Forest Tree-count Tuning")
    st.pyplot(fig)

def report_download(title, sections, filename):
    text = title + "\n" + "=" * len(title) + "\n\n" + "\n\n".join(sections)
    st.download_button("Download result + interpretation", data=text.encode("utf-8"), file_name=filename, mime="text/plain")
    return text


def dataset_download(df, filename, label="Download current dataset CSV", key=None):
    """Consistent CSV download control for teaching datasets."""
    if df is None:
        return
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key
    )


def read_uploaded_csv(uploaded_file):
    """Read uploaded CSV files using common text encodings."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    last_error = None
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            return df, encoding
        except UnicodeDecodeError as e:
            last_error = e

    uploaded_file.seek(0)
    if last_error:
        raise last_error
    raise ValueError("Unable to decode the CSV file.")


def classification_interpretation(model_name, metrics, threshold, cm=None, top_features=None, model_notes=None):
    acc = metrics["Accuracy"]
    p = metrics["Precision"]
    r = metrics["Recall"]
    f1 = metrics["F1"]
    auc = metrics.get("ROC-AUC", np.nan)

    parts = []
    parts.append(
        f"At the selected threshold of {threshold:.2f}, {model_name} achieved "
        f"accuracy {acc:.3f}, precision {p:.3f}, recall {r:.3f}, and F1 {f1:.3f}"
        + (f", with ROC-AUC {auc:.3f}." if not np.isnan(auc) else ".")
    )

    if cm is not None:
        tn, fp, fn, tp = cm.ravel()
        parts.append(
            f"The test set contained {tp} true positives and {tn} true negatives, "
            f"while {fp} negative cases were incorrectly flagged and {fn} positive cases were missed."
        )

    if r < 0.50:
        parts.append(
            "Recall is below 0.50, so the model is missing more than half of the positive cases. "
            "This is important if false negatives are costly."
        )
    elif r >= 0.80:
        parts.append(
            "Recall is high, so the model captures most positive cases, although the precision level should be checked for the cost of false alarms."
        )

    if p < 0.50:
        parts.append(
            "Precision is below 0.50, meaning fewer than half of the cases flagged positive are actually positive."
        )
    elif p >= 0.80:
        parts.append(
            "Precision is high, so most cases flagged positive are genuinely positive."
        )

    if not np.isnan(auc):
        if auc >= 0.90:
            auc_text = "excellent discrimination"
        elif auc >= 0.80:
            auc_text = "good discrimination"
        elif auc >= 0.70:
            auc_text = "moderate discrimination"
        elif auc >= 0.60:
            auc_text = "weak discrimination"
        else:
            auc_text = "very limited discrimination"
        parts.append(f"The ROC-AUC indicates {auc_text} across thresholds.")

    if top_features:
        parts.append(
            "The strongest model-level drivers among the selected predictors are "
            + ", ".join(top_features[:3]) + "."
        )

    if model_notes:
        parts.append(model_notes)

    conclusion = (
        "Conclusion: The model should not be selected from accuracy alone. "
        "Its suitability depends on the relative cost of false negatives and false positives."
    )

    if r < 0.50:
        recommendation = (
            "Recommendation: If detecting positive cases is the priority, consider lowering the classification threshold, "
            "improving the predictor set, or comparing a model with better recall while monitoring the increase in false positives."
        )
    elif p < 0.50:
        recommendation = (
            "Recommendation: If false alarms are costly, consider raising the threshold or selecting a model with stronger precision."
        )
    else:
        recommendation = (
            "Recommendation: Validate this result on an untouched sample, compare it with at least one alternative model, "
            "and select the operating threshold using the actual business cost of errors."
        )

    return "\n\n".join(["Interpretation: " + " ".join(parts), conclusion, recommendation])


def classification_region_plot(
    X, y, selected_num, estimator, model_name, key_prefix="classplot"
):
    st.subheader("2D Classification View")

    if len(selected_num) < 2:
        st.caption(
            "Select at least two numeric predictors to display a 2D classification plot."
        )
        return

    c1, c2 = st.columns(2)
    with c1:
        x_feature = st.selectbox(
            "Horizontal-axis predictor",
            selected_num,
            index=0,
            key=f"{key_prefix}_x"
        )
    with c2:
        y_choices = [c for c in selected_num if c != x_feature]
        y_feature = st.selectbox(
            "Vertical-axis predictor",
            y_choices,
            index=0,
            key=f"{key_prefix}_y"
        )

    vis = pd.DataFrame({
        x_feature: pd.to_numeric(X[x_feature], errors="coerce"),
        y_feature: pd.to_numeric(X[y_feature], errors="coerce"),
        "_target": y
    }).dropna()

    if len(vis) > 3000:
        vis = vis.sample(3000, random_state=42)

    Xv = vis[[x_feature, y_feature]]
    yv = vis["_target"].astype(int)

    vis_est = clone(estimator)
    scaler = None

    # -----------------------------------------------------
    # Interactive teaching controls for the 2D refit
    # -----------------------------------------------------
    if model_name == "SVM":
        st.markdown("**Teaching controls for the SVM plot**")

        svm_kernel = vis_est.get_params().get("kernel", "linear")
        current_c = float(vis_est.get_params().get("C", 1.0))

        visual_c = st.slider(
            "Visualization C (margin / penalty trade-off)",
            min_value=0.05,
            max_value=20.0,
            value=float(min(max(current_c, 0.05), 20.0)),
            step=0.05,
            key=f"{key_prefix}_visual_c",
            help=(
                "Smaller C allows a wider margin with more classification errors. "
                "Larger C penalizes errors more strongly and usually gives a tighter margin."
            )
        )
        vis_est.set_params(C=visual_c)

        if svm_kernel == "rbf":
            current_gamma = vis_est.get_params().get("gamma", "scale")
            default_gamma = (
                float(current_gamma)
                if isinstance(current_gamma, (int, float))
                else 0.5
            )
            visual_gamma = st.slider(
                "Visualization gamma (RBF boundary flexibility)",
                min_value=0.01,
                max_value=5.0,
                value=float(min(max(default_gamma, 0.01), 5.0)),
                step=0.01,
                key=f"{key_prefix}_visual_gamma"
            )
            vis_est.set_params(gamma=visual_gamma)

    elif model_name in [
        "Logistic Regression", "Decision Tree",
        "Random Forest", "Gradient Boosting", "XGBoost"
    ]:
        plot_threshold = st.slider(
            "2D plot classification threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            key=f"{key_prefix}_plot_threshold",
            help=(
                "This changes only the displayed 2D classification regions. "
                "The main model metrics above use the threshold selected in the main model controls."
            )
        )
    else:
        plot_threshold = 0.50

    if model_name in ["Logistic Regression", "SVM"]:
        scaler = StandardScaler()
        Xfit = scaler.fit_transform(Xv)
    else:
        Xfit = Xv.values

    vis_est.fit(Xfit, yv)

    x_min, x_max = Xv[x_feature].min(), Xv[x_feature].max()
    y_min, y_max = Xv[y_feature].min(), Xv[y_feature].max()
    x_pad = (x_max - x_min) * 0.08 if x_max > x_min else 1
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 240),
        np.linspace(y_min - y_pad, y_max + y_pad, 240)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_fit = scaler.transform(grid) if scaler is not None else grid

    fig, ax = plt.subplots(figsize=(8, 6))

    if model_name == "SVM":
        # For SVM margins, use the decision function itself.
        # Levels -1, 0, +1 correspond to the two margins and decision boundary.
        z = vis_est.decision_function(grid_fit).reshape(xx.shape)

        ax.contourf(
            xx, yy,
            (z > 0).astype(int),
            levels=[-0.5, 0.5, 1.5],
            alpha=0.16
        )

        contour = ax.contour(
            xx, yy, z,
            levels=[-1, 0, 1],
            linestyles=["--", "-", "--"],
            linewidths=[1.8, 2.2, 1.8]
        )

        # Explicit labels make the plot easier to teach
        fmt = {-1: "Margin -1", 0: "Decision boundary", 1: "Margin +1"}
        try:
            ax.clabel(contour, contour.levels, inline=True, fmt=fmt, fontsize=8)
        except Exception:
            pass

        if hasattr(vis_est, "support_vectors_"):
            sv = vis_est.support_vectors_
            if scaler is not None:
                sv = scaler.inverse_transform(sv)

            ax.scatter(
                sv[:, 0], sv[:, 1],
                s=95,
                facecolors="none",
                edgecolors="black",
                linewidths=1.4,
                label="Support vectors"
            )

        ax.set_title(
            f"SVM Boundary and Margins — C={vis_est.get_params().get('C', 1.0):.2f}"
        )

    else:
        if hasattr(vis_est, "predict_proba"):
            probs = vis_est.predict_proba(grid_fit)[:, 1]
            pred_grid = (probs >= plot_threshold).astype(int).reshape(xx.shape)
        else:
            pred_grid = vis_est.predict(grid_fit).reshape(xx.shape)

        ax.contourf(
            xx, yy,
            pred_grid,
            levels=[-0.5, 0.5, 1.5],
            alpha=0.16
        )
        ax.set_title(
            f"{model_name} Classification Regions"
            + (
                f" — threshold={plot_threshold:.2f}"
                if "plot_threshold" in locals()
                else ""
            )
        )

    ax.scatter(
        Xv[x_feature],
        Xv[y_feature],
        c=yv,
        alpha=0.65,
        edgecolors="none"
    )
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)

    if model_name == "SVM":
        ax.legend()

    st.pyplot(fig)

    if model_name == "SVM":
        st.caption(
            "Teaching visualization: change C above and observe how the fitted 2D SVM boundary, "
            "margins and support vectors change. This plot is refitted using only the two displayed "
            "numeric predictors; the main performance metrics use the full predictor set."
        )
    else:
        st.caption(
            "Teaching visualization: change the displayed threshold to see how the 2D classification "
            "regions change. This plot is refitted using only the two displayed numeric predictors; "
            "the main performance metrics use the full predictor set."
        )


def pca_color_scatter(ax, x, y, color_series, palette_name):
    color_text = color_series.astype(str)
    codes, uniques = pd.factorize(color_text)

    if palette_name == "Default":
        cmap = "viridis"
    else:
        cmap = palette_name

    sc = ax.scatter(x, y, c=codes, cmap=cmap, alpha=0.58)

    if len(uniques) <= 12:
        cmap_obj = plt.get_cmap(cmap)
        denom = max(len(uniques) - 1, 1)
        handles = []
        for i, label in enumerate(uniques):
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=cmap_obj(i / denom),
                    markeredgecolor="none",
                    label=str(label)
                )
            )
        ax.legend(handles=handles, title=color_series.name)




def choose_row_subset(df, key, supervised_target=None):
    """
    Return either the full dataframe or a reproducible row sample.
    If a supervised target is supplied, preserve the target column and
    approximately preserve its class distribution.
    """
    total = len(df)

    use_subset = st.checkbox(
        "Use only a subset of rows",
        value=False,
        key=f"{key}_subset_check"
    )

    if not use_subset or total <= 2:
        return df.copy()

    max_rows = total
    default_rows = min(100, total)
    min_rows = min(20, total)

    n_rows = int(
        st.number_input(
            "Number of rows to use",
            min_value=int(min_rows),
            max_value=int(max_rows),
            value=int(default_rows),
            step=1,
            key=f"{key}_subset_n"
        )
    )

    # Stratified sampling for supervised problems.
    # Sample row INDICES, then select them from the original dataframe.
    # This guarantees that the target column is preserved.
    if (
        supervised_target is not None
        and supervised_target in df.columns
        and df[supervised_target].nunique(dropna=True) > 1
    ):
        try:
            valid_target = df[supervised_target].notna()
            valid_df = df.loc[valid_target]

            if n_rows >= len(valid_df):
                sampled = valid_df.copy()
            else:
                # sklearn's split preserves class proportions and retains
                # the original dataframe columns.
                sampled, _ = train_test_split(
                    valid_df,
                    train_size=n_rows,
                    stratify=valid_df[supervised_target],
                    random_state=42
                )

            return sampled.sample(frac=1, random_state=42).copy()

        except ValueError:
            # If a class is too small for sklearn stratification,
            # fall back to a reproducible random sample without dropping columns.
            return df.sample(
                n=min(n_rows, total),
                random_state=42
            ).copy()

    return df.sample(
        n=min(n_rows, total),
        random_state=42
    ).copy()



def input_new_case(df, numeric_features, categorical_features, key_prefix):
    row = {}
    st.subheader("Try an Unseen / User-entered Case")
    st.caption(
        "Enter predictor values for a new observation. The prediction uses the model trained above."
    )
    with st.form(f"{key_prefix}_new_case"):
        for col in numeric_features:
            series = pd.to_numeric(df[col], errors="coerce")
            default = float(series.median()) if series.notna().any() else 0.0
            row[col] = st.number_input(
                col,
                value=default,
                key=f"{key_prefix}_num_{col}"
            )
        for col in categorical_features:
            options = [str(v) for v in df[col].dropna().astype(str).unique().tolist()]
            if not options:
                options = [""]
            row[col] = st.selectbox(
                col,
                options,
                key=f"{key_prefix}_cat_{col}"
            )
        submitted = st.form_submit_button("Predict new case")
    return submitted, pd.DataFrame([row])


def regression_new_case(pipe_or_model, df, numeric_features, categorical_features, key_prefix, target_name):
    submitted, new_df = input_new_case(
        df, numeric_features, categorical_features, key_prefix
    )
    if submitted:
        try:
            pred = float(pipe_or_model.predict(new_df)[0])
            st.success(f"Predicted {target_name}: {pred:,.3f}")
            return pred, new_df
        except Exception as e:
            st.error(f"Could not generate prediction: {e}")
    return None, None


def classification_new_case(estimator, prep, df, numeric_features, categorical_features, threshold, key_prefix, reverse_mapping=None):
    submitted, new_df = input_new_case(
        df, numeric_features, categorical_features, key_prefix
    )
    if submitted:
        try:
            Xt = prep.transform(new_df)
            prob = float(estimator.predict_proba(Xt)[0, 1])
            pred = int(prob >= threshold)
            display_class = reverse_mapping.get(pred, pred) if reverse_mapping else pred
            st.metric("Predicted probability of positive class", f"{prob:.3f}")
            st.success(
                f"Predicted class at threshold {threshold:.2f}: {display_class}"
            )
            return prob, pred, new_df
        except Exception as e:
            st.error(f"Could not generate prediction: {e}")
    return None, None, None


@st.cache_resource(show_spinner=False)
def get_finbert_pipeline():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )


def vader_sentiment(text_value):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text_value)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, compound, scores


def footer():
    st.markdown("---")
    st.caption(
        "Machine Learning for BIFS — Teaching Portal | "
        "Concept and teaching design: Prof. Adlino Afonso | "
        "Developed with AI assistance | For educational use only."
    )


# ---------------------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------------------
heartbeat()
active_users_sidebar = current_active_users()
st.sidebar.caption(f"Logged in as: {st.session_state.get('user_email','')}")
st.sidebar.caption(
    f"Active users: {active_users_sidebar} / {MAX_ACTIVE_USERS}"
)
st.sidebar.caption(
    f"Current session: {session_elapsed_minutes():.1f} minutes"
)

if st.sidebar.button("Logout"):
    logout_current_session("logout")
    for key in [
        "authenticated", "user_email", "user_city",
        "portal_session_id", "portal_login_time"
    ]:
        st.session_state.pop(key, None)
    st.rerun()

section = st.sidebar.selectbox("Choose learning area", [
    "Home", "Regression", "Classification", "K-Means Clustering",
    "Hierarchical Clustering", "PCA", "Semi-Supervised Learning",
    "Reinforcement Learning", "NLP", "Feedback", "Admin Analytics"
])
log_event(section, "section_opened", {"city": st.session_state.get("user_city", "")})

# ---------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------
if section == "Home":
    st.subheader("About this teaching laboratory")
    st.markdown(
        "**Concept and teaching design: Prof. Adlino Afonso. Developed with AI assistance.**\n\n"
        "This application is intended **only for teaching and classroom experimentation**. "
        "It is not a validated production system and should not be used for real lending, underwriting, fraud, investment, insurance, or customer decisions."
    )
    st.subheader("How students should use it")
    st.markdown(
        "1. Choose a learning area.\n"
        "2. Use the built-in data or upload a CSV.\n"
        "3. Check variable types and encoding requirements.\n"
        "4. Change one setting at a time.\n"
        "5. Observe the plot and metrics.\n"
        "6. Write a business interpretation.\n"
        "7. Export the result and compare it with another model or setting."
    )

# ---------------------------------------------------------------------
# REGRESSION
# ---------------------------------------------------------------------
elif section == "Regression":
    st.header("Regression Laboratory")
    df, custom = upload_or_builtin("reg", regression_data())
    dataset_download(
        df,
        "regression_teaching_dataset.csv",
        "Download regression dataset CSV",
        key="download_reg_dataset"
    )
    df = choose_row_subset(df, "reg")
    nums = numeric_cols(df)
    if len(nums) < 2:
        st.error("Need at least two numeric columns."); st.stop()
    model_type = st.selectbox("Choose regression model", ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression"])
    log_event("Regression", "model_tried", {"model": model_type})
    target = st.selectbox("Choose numeric target", nums, index=(nums.index("Recommended Credit Limit") if "Recommended Credit Limit" in nums else len(nums)-1))
    encoding_panel(df, target, "reg_enc")
    id_cols = id_like_columns(df)
    num_candidates = [c for c in nums if c != target and c not in id_cols]
    cat_candidates = [c for c in categorical_cols(df) if c != target]

    if model_type == "Simple Linear Regression":
        xcol = st.selectbox("Choose numeric predictor", num_candidates)
        clean = df[[xcol, target]].dropna(); X = clean[[xcol]]; y = clean[target]
        model = LinearRegression().fit(X, y)
        pred = model.predict(X)

        fitted_slope = float(model.coef_[0])
        fitted_intercept = float(model.intercept_)

        st.subheader("Interactive Regression Line")
        show_manual = st.checkbox(
            "Try my own slope and intercept",
            value=False,
            key="simple_reg_manual"
        )

        order = np.argsort(clean[xcol].values)
        x_sorted = clean[xcol].values[order]

        fig, ax = plt.subplots()
        ax.scatter(clean[xcol], y, alpha=0.5, label="Observed data")
        ax.plot(
            x_sorted,
            pred[order],
            linewidth=2.2,
            label="Least-squares fitted line"
        )

        if show_manual:
            y_std = float(np.std(y))
            x_std = float(np.std(clean[xcol]))
            slope_span = max(
                abs(fitted_slope) * 3,
                (y_std / x_std) * 3 if x_std > 0 else 1.0,
                1.0
            )
            intercept_span = max(abs(fitted_intercept) * 1.5, y_std * 3, 1.0)

            manual_slope = st.slider(
                "Manual slope",
                min_value=float(fitted_slope - slope_span),
                max_value=float(fitted_slope + slope_span),
                value=float(fitted_slope),
                step=float(max(slope_span / 100, 0.001)),
                key="manual_reg_slope"
            )
            manual_intercept = st.slider(
                "Manual intercept",
                min_value=float(fitted_intercept - intercept_span),
                max_value=float(fitted_intercept + intercept_span),
                value=float(fitted_intercept),
                step=float(max(intercept_span / 100, 0.001)),
                key="manual_reg_intercept"
            )

            manual_pred = manual_intercept + manual_slope * clean[xcol].values
            manual_rmse = mean_squared_error(y, manual_pred) ** 0.5
            fitted_rmse = mean_squared_error(y, pred) ** 0.5

            manual_line = manual_intercept + manual_slope * x_sorted
            ax.plot(
                x_sorted,
                manual_line,
                linestyle="--",
                linewidth=2,
                label="Your line"
            )

            c1, c2 = st.columns(2)
            c1.metric("Least-squares RMSE", f"{fitted_rmse:.3f}")
            c2.metric(
                "Your-line RMSE",
                f"{manual_rmse:.3f}",
                delta=f"{manual_rmse - fitted_rmse:+.3f} vs fitted"
            )

            st.caption(
                "Move the slope and intercept. The least-squares line is the line that minimizes "
                "the sum of squared residuals, so your manually chosen line should normally have "
                "the same or a larger training error."
            )

        ax.set_xlabel(xcol)
        ax.set_ylabel(target)
        ax.set_title("Simple Linear Regression")
        ax.legend()
        st.pyplot(fig)

        r2 = r2_score(y, pred)
        rmse = mean_squared_error(y, pred) ** 0.5
        st.metric("R²", f"{r2:.3f}")
        st.metric("RMSE", f"{rmse:.3f}")
        st.caption(
            f"Fitted equation: {target} = {fitted_intercept:.4f} "
            f"+ ({fitted_slope:.4f} × {xcol})"
        )
        regression_new_case(
            model,
            clean,
            [xcol],
            [],
            "simple_reg_new",
            target
        )
        results = clean.copy(); results["Prediction"] = pred; results["Residual"] = y - pred
        st.download_button("Download predictions CSV", results.to_csv(index=False).encode(), "linear_regression_results.csv", "text/csv")
        interpretation = f"Interpretation: {xcol} alone explains R²={r2:.3f} of the variation in {target}; RMSE={rmse:.3f}.\nConclusion: A single predictor may be useful for illustration but may omit important drivers.\nRecommendation: Compare with multiple regression before drawing business conclusions."
        st.subheader("Conclusion and Recommendation"); st.write(interpretation)
        report_download("Simple Linear Regression Result", [interpretation], "linear_regression_report.txt")

    elif model_type == "Multiple Linear Regression":
        selected_num = st.multiselect("Choose numeric predictors", num_candidates, default=num_candidates[:min(4, len(num_candidates))])
        selected_cat = st.multiselect("Choose categorical predictors (one-hot encoded automatically)", cat_candidates, default=[])
        if len(selected_num) + len(selected_cat) < 2:
            st.warning("Choose at least two predictors."); st.stop()
        cols = selected_num + selected_cat + [target]; clean = df[cols].dropna(); X = clean[selected_num + selected_cat]; y = clean[target]
        prep = make_preprocessor(selected_num, selected_cat, scale_numeric=False)
        pipe = Pipeline([("prep", prep), ("model", LinearRegression())]); pipe.fit(X, y); pred = pipe.predict(X)
        r2 = r2_score(y, pred); rmse = mean_squared_error(y, pred) ** 0.5
        st.metric("R²", f"{r2:.3f}"); st.metric("RMSE", f"{rmse:.3f}")
        names = feature_names_from_preprocessor(pipe.named_steps["prep"]); coefs = pipe.named_steps["model"].coef_
        if len(names) == len(coefs):
            st.subheader("Coefficients after encoding")
            st.dataframe(pd.DataFrame({"Transformed Predictor": names, "Coefficient": coefs}), use_container_width=True)
        regression_new_case(
            pipe,
            clean,
            selected_num,
            selected_cat,
            "multiple_reg_new",
            target
        )
        results = clean.copy(); results["Prediction"] = pred; results["Residual"] = y - pred
        st.download_button("Download predictions CSV", results.to_csv(index=False).encode(), "multiple_regression_results.csv", "text/csv")
        interpretation = f"Interpretation: The selected predictors jointly produce R²={r2:.3f} and RMSE={rmse:.3f}. Categorical predictors were one-hot encoded when selected.\nConclusion: The model combines numeric and categorical information rather than discarding variables such as smoker/region.\nRecommendation: Review coefficient direction, residuals and out-of-sample performance before business use."
        st.subheader("Conclusion and Recommendation"); st.write(interpretation)
        report_download("Multiple Linear Regression Result", [interpretation], "multiple_regression_report.txt")

    else:
        xcol = st.selectbox("Choose numeric predictor", num_candidates); degree = st.slider("Polynomial degree", 1, 6, 2)
        clean = df[[xcol, target]].dropna(); X = clean[[xcol]]; y = clean[target]
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression()); model.fit(X, y)
        grid = np.linspace(clean[xcol].min(), clean[xcol].max(), 300).reshape(-1, 1); curve = model.predict(pd.DataFrame(grid, columns=[xcol])); pred = model.predict(X)
        fig, ax = plt.subplots(); ax.scatter(clean[xcol], y, alpha=0.45); ax.plot(grid[:,0], curve)
        ax.set_xlabel(xcol); ax.set_ylabel(target); ax.set_title(f"Polynomial Regression — Degree {degree}"); st.pyplot(fig)
        r2 = r2_score(y, pred); rmse = mean_squared_error(y, pred) ** 0.5
        st.metric("R²", f"{r2:.3f}"); st.metric("RMSE", f"{rmse:.3f}")
        regression_new_case(
            model,
            clean,
            [xcol],
            [],
            "poly_reg_new",
            target
        )
        results = clean.copy(); results["Prediction"] = pred; results["Residual"] = y - pred
        st.download_button("Download predictions CSV", results.to_csv(index=False).encode(), "polynomial_regression_results.csv", "text/csv")
        interpretation = f"Interpretation: Degree {degree} gives R²={r2:.3f} and RMSE={rmse:.3f}.\nConclusion: Higher degree increases flexibility but may overfit.\nRecommendation: Compare train/test performance across degrees rather than choosing the curve that fits the sample best."
        st.subheader("Conclusion and Recommendation"); st.write(interpretation)
        report_download("Polynomial Regression Result", [interpretation], "polynomial_regression_report.txt")

# ---------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------
elif section == "Classification":
    st.header("Classification Laboratory")
    df, custom = upload_or_builtin("class", classification_data())
    dataset_download(
        df,
        "classification_teaching_dataset.csv",
        "Download classification dataset CSV",
        key="download_class_dataset"
    )
    if custom:
        target_col = st.selectbox("Choose binary target column", df.columns)
        df = choose_row_subset(df, "class", supervised_target=target_col)
        y, mapping = binary_target(df[target_col])
        if y is None:
            st.error("Target must contain exactly two classes."); st.stop()
        st.write("Target encoding:", mapping)
    else:
        target_col = "Default"; y = df[target_col]
    encoding_panel(df, target_col, "class_enc")
    id_cols = id_like_columns(df)
    num_candidates = [c for c in numeric_cols(df) if c != target_col and c not in id_cols]
    cat_candidates = [c for c in categorical_cols(df) if c != target_col]
    selected_num = st.multiselect("Choose numeric predictors", num_candidates, default=num_candidates[:min(6, len(num_candidates))])
    selected_cat = st.multiselect("Choose categorical predictors (one-hot encoded automatically)", cat_candidates, default=[])
    if len(selected_num) + len(selected_cat) < 1:
        st.warning("Choose at least one predictor."); st.stop()
    X = df[selected_num + selected_cat].copy(); valid = X.notna().all(axis=1) & y.notna(); X = X.loc[valid]; y = y.loc[valid].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

    class_counts = y.value_counts().sort_index()
    minority_ratio = class_counts.min() / class_counts.sum()
    st.write("Class distribution:", class_counts.to_dict())
    if minority_ratio < 0.40:
        st.warning(
            f"The target is imbalanced: the minority class represents {minority_ratio:.1%} of observations."
        )

    handle_imbalance = st.checkbox(
        "Handle class imbalance during training",
        value=True,
        help="Recommended when one class is substantially less common than the other."
    )

    model_name = st.selectbox("Choose classification model", ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost", "SVM"])
    log_event("Classification", "model_tried", {"model": model_name})
    threshold = st.slider("Classification threshold", 0.10, 0.90, 0.50, 0.05)
    needs_scaling = model_name in ["Logistic Regression", "SVM"]
    prep = make_preprocessor(selected_num, selected_cat, scale_numeric=needs_scaling)

    if model_name == "Logistic Regression":
        estimator = LogisticRegression(max_iter=2000, class_weight=("balanced" if handle_imbalance else None))
    elif model_name == "Decision Tree":
        depth = st.slider("Maximum depth", 1, 12, 4); estimator = DecisionTreeClassifier(max_depth=depth, class_weight=("balanced" if handle_imbalance else None), random_state=42)
    elif model_name == "Random Forest":
        trees = st.slider("Number of trees", 10, 400, 150, 10); depth = st.slider("Maximum depth", 2, 15, 6)
        estimator = RandomForestClassifier(n_estimators=trees, max_depth=depth, class_weight=("balanced" if handle_imbalance else None), random_state=42, n_jobs=-1)
    elif model_name == "Gradient Boosting":
        trees = st.slider("Number of estimators", 20, 300, 100, 10); lr = st.slider("Learning rate", 0.01, 0.50, 0.10, 0.01); depth = st.slider("Tree depth", 1, 5, 2)
        estimator = GradientBoostingClassifier(n_estimators=trees, learning_rate=lr, max_depth=depth, random_state=42)
    elif model_name == "XGBoost":
        if not XGB_AVAILABLE:
            st.error("Install XGBoost using: python -m pip install xgboost"); st.stop()
        trees = st.slider("Number of estimators", 20, 300, 100, 10); lr = st.slider("Learning rate", 0.01, 0.50, 0.10, 0.01); depth = st.slider("Maximum depth", 1, 8, 3)
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos = (neg_count / pos_count) if (handle_imbalance and pos_count > 0) else 1.0
        estimator = XGBClassifier(
            n_estimators=trees,
            learning_rate=lr,
            max_depth=depth,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos
        )
        if handle_imbalance:
            st.caption(f"XGBoost class-imbalance weight (scale_pos_weight) = {scale_pos:.3f}")
    else:
        kernel = st.selectbox("Kernel", ["linear", "rbf"]); C = st.slider("C", 0.1, 20.0, 1.0, 0.1); gamma = "scale"
        if kernel == "rbf": gamma = st.slider("Gamma", 0.01, 5.0, 0.5, 0.01)
        estimator = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, class_weight=("balanced" if handle_imbalance else None), random_state=42)

    # Fit transformed data explicitly so learning curves can use it.
    X_train_t = prep.fit_transform(X_train); X_test_t = prep.transform(X_test)
    if model_name == "XGBoost":
        estimator.fit(
            X_train_t, y_train,
            eval_set=[(X_train_t, y_train), (X_test_t, y_test)],
            verbose=False
        )
        xgb_results = estimator.evals_result()
    elif model_name == "Gradient Boosting" and handle_imbalance:
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        estimator.fit(X_train_t, y_train, sample_weight=sample_weights)
        st.caption("Gradient Boosting fitted with balanced sample weights.")
    else:
        estimator.fit(X_train_t, y_train)
    y_prob = estimator.predict_proba(X_test_t)[:, 1]; y_pred = (y_prob >= threshold).astype(int)
    metrics = metric_dict(y_test, y_pred, y_prob)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Performance"); st.dataframe(metric_table(y_test, y_pred, y_prob), use_container_width=True)
    with c2:
        show_cm(confusion_matrix(y_test, y_pred))

    transformed_names = feature_names_from_preprocessor(prep)
    if model_name == "Decision Tree":
        st.subheader("Decision Tree")
        fig, ax = plt.subplots(figsize=(15,7)); plot_tree(estimator, feature_names=transformed_names if transformed_names else None, class_names=["Class 0","Class 1"], filled=True, max_depth=min(3, estimator.get_depth()), ax=ax); st.pyplot(fig)
    if model_name in ["Decision Tree", "Random Forest"] and transformed_names:
        st.subheader("Feature Importance")
        imp = pd.DataFrame({"Feature": transformed_names, "Importance": estimator.feature_importances_}).sort_values("Importance", ascending=True)
        fig, ax = plt.subplots(); ax.barh(imp["Feature"], imp["Importance"]); ax.set_title("Feature Importance"); st.pyplot(fig)

    # Visual classification view
    classification_region_plot(
        X, y, selected_num, estimator, model_name, key_prefix="classviz"
    )

    # Keep optimization/learning curves only for boosting models
    if model_name in ["Gradient Boosting", "XGBoost"]:
        st.subheader("Optimization / Learning Plot")

    if model_name == "Gradient Boosting":
        gradient_learning_plot(estimator, X_train_t, y_train, X_test_t, y_test)

    elif model_name == "XGBoost":
        train_loss = xgb_results["validation_0"]["logloss"]
        test_loss = xgb_results["validation_1"]["logloss"]
        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_loss)+1), train_loss, label="Training log loss")
        ax.plot(range(1, len(test_loss)+1), test_loss, label="Test log loss")
        ax.set_xlabel("Boosting Iteration / Number of Trees")
        ax.set_ylabel("Log Loss")
        ax.set_title("XGBoost Learning Curve")
        ax.legend()
        st.pyplot(fig)
        st.caption(
            f"Lowest test log loss occurs around iteration {int(np.argmin(test_loss)+1)}."
        )

    pred_out = X_test.copy()
    pred_out["Actual"] = y_test
    pred_out["Predicted Probability"] = y_prob
    pred_out["Predicted Class"] = y_pred
    st.download_button(
        "Download prediction results CSV",
        pred_out.to_csv(index=False).encode(),
        "classification_results.csv",
        "text/csv"
    )

    cm_now = confusion_matrix(y_test, y_pred)
    top_features = []
    if model_name in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"] and transformed_names:
        if hasattr(estimator, "feature_importances_"):
            imp_now = pd.Series(
                estimator.feature_importances_,
                index=transformed_names
            ).sort_values(ascending=False)
            top_features = imp_now.head(3).index.tolist()
    elif model_name == "Logistic Regression" and transformed_names:
        coef_now = pd.Series(
            np.abs(estimator.coef_[0]),
            index=transformed_names
        ).sort_values(ascending=False)
        top_features = coef_now.head(3).index.tolist()

    notes = "Class imbalance handling was enabled." if handle_imbalance else "Class imbalance handling was not enabled."
    if model_name == "SVM":
        notes += " " + (
            f"The SVM used a {kernel} kernel with C={C:.2f}"
            + (f" and gamma={gamma:.2f}." if kernel == "rbf" and isinstance(gamma, float) else ".")
        )
    elif model_name == "Decision Tree":
        notes += f" The selected tree depth was {depth}."
    elif model_name == "Random Forest":
        notes += f" The forest used {trees} trees with maximum depth {depth}."
    elif model_name in ["Gradient Boosting", "XGBoost"]:
        notes += f" The model used {trees} boosting stages, learning rate {lr:.2f}, and tree depth {depth}."

    interpretation = classification_interpretation(
        model_name, metrics, threshold,
        cm=cm_now,
        top_features=top_features,
        model_notes=notes
    )
    reverse_mapping = None
    if custom:
        reverse_mapping = {v: k for k, v in mapping.items()}

    classification_new_case(
        estimator,
        prep,
        df,
        selected_num,
        selected_cat,
        threshold,
        "classification_new",
        reverse_mapping=reverse_mapping
    )

    st.subheader("Conclusion and Recommendation")
    st.write(interpretation)
    report_download(
        f"{model_name} Classification Result",
        [interpretation],
        "classification_report.txt"
    )

# ---------------------------------------------------------------------
# K-MEANS
# ---------------------------------------------------------------------
elif section == "K-Means Clustering":
    st.header("K-Means Customer Segmentation")
    df, custom = upload_or_builtin("kmeans", clustering_data())
    dataset_download(
        df,
        "kmeans_teaching_dataset.csv",
        "Download K-Means dataset CSV",
        key="download_kmeans_dataset"
    )
    df = choose_row_subset(df, "kmeans")
    encoding_panel(df, None, "km_enc")
    nums = [c for c in numeric_cols(df) if c not in id_like_columns(df)]
    selected = st.multiselect("Choose numeric clustering variables", nums, default=nums[:min(4,len(nums))])
    if len(selected) < 2:
        st.warning("Choose at least two variables."); st.stop()
    work = df[selected].dropna(); k = st.slider("Number of clusters (K)", 2, 8, 3)
    scaler = StandardScaler(); Xs = scaler.fit_transform(work)
    model = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42); labels = model.fit_predict(Xs)
    sil = silhouette_score(Xs, labels); inertia = model.inertia_
    st.metric("Silhouette Score", f"{sil:.3f}"); st.metric("Within-cluster Sum of Squares (Inertia)", f"{inertia:.2f}")
    fig, ax = plt.subplots(); ax.scatter(work[selected[0]], work[selected[1]], c=labels, alpha=0.65)
    centers = scaler.inverse_transform(model.cluster_centers_); ax.scatter(centers[:,0], centers[:,1], marker="X", s=180)
    ax.set_xlabel(selected[0]); ax.set_ylabel(selected[1]); ax.set_title(f"K-Means — K={k}"); st.pyplot(fig)
    st.subheader("Assign a New Case to a Cluster")
    with st.form("kmeans_new_case"):
        new_vals = {}
        for col in selected:
            new_vals[col] = st.number_input(
                col,
                value=float(pd.to_numeric(work[col], errors="coerce").median()),
                key=f"kmeans_new_{col}"
            )
        assign_new = st.form_submit_button("Assign cluster")
    if assign_new:
        new_case = pd.DataFrame([new_vals])
        new_scaled = scaler.transform(new_case[selected])
        new_cluster = int(model.predict(new_scaled)[0])
        st.success(f"The new case is assigned to Cluster {new_cluster}.")
        st.caption("The assignment is based on the nearest fitted K-means centroid.")

    out = work.copy(); out["Cluster"] = labels; st.download_button("Download cluster assignments CSV", out.to_csv(index=False).encode(), "kmeans_clusters.csv", "text/csv")
    quality = "strong" if sil >= 0.5 else ("moderate" if sil >= 0.25 else "weak")
    interpretation = f"Interpretation: K={k} produced silhouette={sil:.3f}, suggesting {quality} separation by a common heuristic.\nConclusion: The statistical result should be combined with business interpretability of the segment profiles.\nRecommendation: Compare nearby K values and profile each cluster before naming or acting on the segments."
    st.subheader("Conclusion and Recommendation"); st.write(interpretation); report_download("K-Means Result", [interpretation], "kmeans_report.txt")

# ---------------------------------------------------------------------
# HIERARCHICAL
# ---------------------------------------------------------------------
elif section == "Hierarchical Clustering":
    st.header("Hierarchical Clustering")
    df, custom = upload_or_builtin("hier", clustering_data(n=150))
    dataset_download(
        df,
        "hierarchical_teaching_dataset.csv",
        "Download hierarchical clustering dataset CSV",
        key="download_hier_dataset"
    )
    df = choose_row_subset(df, "hier")
    encoding_panel(df, None, "hier_enc")
    nums = [c for c in numeric_cols(df) if c not in id_like_columns(df)]
    selected = st.multiselect("Choose numeric clustering variables", nums, default=nums[:min(4,len(nums))])
    if len(selected) < 2:
        st.warning("Choose at least two variables."); st.stop()
    work = df[selected].dropna()
    if len(work) > 300:
        nrows = st.slider("Rows used for dendrogram", 50, 300, 150); work = work.sample(nrows, random_state=42)
    scaler = StandardScaler(); Xs = scaler.fit_transform(work)
    method = st.selectbox("Linkage method", ["single", "complete", "average", "ward"]); k = st.slider("Number of clusters", 2, 8, 3)
    model = AgglomerativeClustering(n_clusters=k, linkage=method); labels = model.fit_predict(Xs)
    c1, c2 = st.columns(2)
    with c1:
        Z = linkage(Xs, method=method); fig, ax = plt.subplots(figsize=(8,5)); dendrogram(Z, no_labels=True, ax=ax); ax.set_xlabel("Observations"); ax.set_ylabel("Linkage Distance"); ax.set_title(f"Dendrogram — {method.title()} Linkage"); st.pyplot(fig)
    with c2:
        fig2, ax2 = plt.subplots(figsize=(7,5)); ax2.scatter(work[selected[0]], work[selected[1]], c=labels, alpha=0.70); ax2.set_xlabel(selected[0]); ax2.set_ylabel(selected[1]); ax2.set_title(f"{k} Clusters"); st.pyplot(fig2)
    st.subheader("New / Unseen Case")
    st.info(
        "Agglomerative hierarchical clustering does not learn a centroid-based prediction rule for unseen observations. "
        "To include a new case correctly, add it to the dataset and refit the hierarchy. "
        "This is an important difference from K-means."
    )

    out = work.copy(); out["Cluster"] = labels; st.download_button("Download cluster assignments CSV", out.to_csv(index=False).encode(), "hierarchical_clusters.csv", "text/csv")
    interpretation = f"Interpretation: {method.title()} linkage with {k} clusters was applied to {len(work)} observations.\nConclusion: Linkage choice changes how cluster-to-cluster distance is defined, so the hierarchy may change even with identical data.\nRecommendation: Compare at least two linkage methods and check whether the resulting segments remain meaningful and stable."
    st.subheader("Conclusion and Recommendation"); st.write(interpretation); report_download("Hierarchical Clustering Result", [interpretation], "hierarchical_report.txt")

# ---------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------
elif section == "PCA":
    st.header("Principal Component Analysis")
    df, custom = upload_or_builtin("pca", pca_customer_data())
    dataset_download(
        df,
        "pca_teaching_dataset.csv",
        "Download PCA dataset CSV",
        key="download_pca_dataset"
    )
    df = choose_row_subset(df, "pca")
    nums = [c for c in numeric_cols(df) if c not in id_like_columns(df)]
    low_num = low_cardinality_numeric(df)
    if low_num:
        st.warning("Potential coded categorical/ordinal numeric columns: " + ", ".join(low_num) + ". PCA treats selected columns as quantitative, so include these only if that interpretation is justified.")
    default_pca = [c for c in nums if df[c].nunique(dropna=True) > 10]
    if len(default_pca) < 2: default_pca = nums[:min(6,len(nums))]
    selected = st.multiselect("Choose numeric variables for PCA", nums, default=default_pca[:min(8,len(default_pca))])
    if len(selected) < 2:
        st.warning("Choose at least two variables."); st.stop()
    color_options = ["(No colour grouping)"] + [c for c in df.columns if c not in selected]
    color_by = st.selectbox("Colour PC1–PC2 plot by", color_options)
    palette = st.selectbox("PCA plot colour palette", ["Default", "viridis", "plasma", "coolwarm", "tab10"])
    work = df[selected].copy(); valid = work.notna().all(axis=1); work = work.loc[valid]
    max_pc = len(selected); n_components = st.slider("Number of principal components", 2, max_pc, min(3,max_pc))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(work)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xs)

    explained = pca.explained_variance_ratio_
    eigenvalues = pca.explained_variance_

    ev = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(n_components)],
        "Variance captured (Eigenvalue)": eigenvalues,
        "Explained variance ratio": explained,
        "Explained variance (%)": explained * 100,
        "Cumulative explained variance (%)": np.cumsum(explained) * 100
    })

    st.subheader("Explained Variance (Eigenvalues)")
    st.dataframe(ev.round(4), use_container_width=True)
    st.caption(
        "The eigenvalue of each principal component is the variance captured by that component. "
        "Explained variance ratio = eigenvalue ÷ total variance."
    )

    fig, ax = plt.subplots()
    ax.bar(ev["Component"], ev["Explained variance (%)"])
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Explained Variance (from Eigenvalues)")
    st.pyplot(fig)

    # pca.components_ are eigenvector coefficients / component weights
    eigenvector_weights = pd.DataFrame(
        pca.components_.T,
        index=selected,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    st.subheader("Component Weights (Eigenvector Coefficients)")
    st.dataframe(eigenvector_weights.round(4), use_container_width=True)
    st.caption(
        "These coefficients define the direction of each principal component and are used to calculate the PC score for each observation."
    )

    # Loadings link eigenvector coefficients with eigenvalues
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(eigenvalues),
        index=selected,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )

    st.subheader("PCA Loadings (Eigenvector Coefficient × √Eigenvalue)")
    st.dataframe(loadings.round(4), use_container_width=True)
    st.caption(
        "A loading combines a variable's component weight (eigenvector coefficient) with the variance captured by that component (eigenvalue). "
        "Large absolute loadings indicate variables strongly associated with that principal component."
    )

    selected_pc = st.selectbox(
        "Choose a principal component to view its loadings",
        [f"PC{i+1}" for i in range(n_components)]
    )

    sorted_loadings = loadings[selected_pc].sort_values()
    fig_load, ax_load = plt.subplots(figsize=(8,5))
    ax_load.barh(sorted_loadings.index, sorted_loadings.values)
    ax_load.axvline(0, linewidth=1)
    ax_load.set_xlabel("Loading")
    ax_load.set_title(f"{selected_pc} Loadings")
    st.pyplot(fig_load)

    st.subheader("How the PCA quantities are linked")
    st.markdown(
        """
        - **Variance captured (Eigenvalue):** how much variance a principal component explains.
        - **Component weight (Eigenvector coefficient):** the coefficient used to define the direction of the principal component.
        - **PCA loading:** component weight (eigenvector coefficient) × √(eigenvalue).
        - **PC score:** the transformed value for each observation after applying the component weights to the standardized variables.
        """
    )

    st.subheader("PC1 vs PC2 Scores")
    fig2, ax2 = plt.subplots()
    if color_by == "(No colour grouping)":
        ax2.scatter(scores[:,0], scores[:,1], alpha=0.55)
    else:
        color_series = df.loc[valid, color_by].copy()
        color_series.name = color_by
        pca_color_scatter(
            ax2, scores[:,0], scores[:,1],
            color_series, palette
        )
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_title("Observations in Principal-Component Space"); st.pyplot(fig2)
    st.subheader("Transform a New Case into Principal-Component Scores")
    with st.form("pca_new_case"):
        pca_new_vals = {}
        for col in selected:
            pca_new_vals[col] = st.number_input(
                col,
                value=float(pd.to_numeric(work[col], errors="coerce").median()),
                key=f"pca_new_{col}"
            )
        transform_new = st.form_submit_button("Calculate PC scores")
    if transform_new:
        new_case = pd.DataFrame([pca_new_vals])
        new_scaled = scaler.transform(new_case[selected])
        new_scores = pca.transform(new_scaled)[0]
        new_score_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(n_components)],
            "Score": new_scores
        })
        st.dataframe(new_score_df, use_container_width=True)
        st.caption(
            "The same standardization and principal-component directions learned from the training data are applied to this new observation."
        )

    score_df = pd.DataFrame(scores, index=work.index, columns=[f"PC{i+1}" for i in range(n_components)])
    st.download_button("Download PCA scores CSV", score_df.to_csv(index=True).encode(), "pca_scores.csv", "text/csv")
    retained = explained[:min(2,len(explained))].sum()
    top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).head(3).index.tolist()
    top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).head(3).index.tolist()
    interpretation = (
        f"Interpretation: PC1 has variance captured (eigenvalue) {eigenvalues[0]:.3f} and explains {explained[0]:.1%} of total variance; "
        f"PC2 has variance captured (eigenvalue) {eigenvalues[1]:.3f} and explains {explained[1]:.1%}. "
        f"Together they retain {retained:.1%}. The strongest contributors to PC1 based on PCA loadings are {', '.join(top_pc1)}, "
        f"while PC2 is driven most strongly by {', '.join(top_pc2)}. "
        + (f"The PC1–PC2 plot is coloured by {color_by}, which is used only for interpretation and not to construct the components. " if color_by != "(No colour grouping)" else "")
        + "\nConclusion: PCA is summarizing the major directions of variation in the selected predictors; it is not optimizing separation of any target class."
        + "\nRecommendation: Use the explained-variance table to decide whether two components are enough, inspect the loading signs and magnitudes before naming a component, and exclude identifiers or coded categories that should not be treated as quantitative measurements."
    )
    st.subheader("Conclusion and Recommendation"); st.write(interpretation); report_download("PCA Result", [interpretation], "pca_report.txt")

# ---------------------------------------------------------------------
# SEMI-SUPERVISED
# ---------------------------------------------------------------------

elif section == "Semi-Supervised Learning":
    st.header("Semi-Supervised Learning")

    method = st.selectbox(
        "Semi-Supervised Method",
        [
            "Wrapper / Self-Training",
            "Clustering-Based"
        ]
    )

    source = st.radio(
        "Data source",
        [
            "Built-in fraud simulation",
            "Upload a fully labelled CSV and hide labels for simulation"
        ],
        horizontal=True
    )

    labelled_percent = st.slider(
        "Percentage initially labelled",
        2, 40, 10, 1
    )

    if source == "Built-in fraud simulation":
        n = st.slider(
            "Number of transactions",
            800, 6000, 3000, 200
        )

        difficulty = st.slider(
            "Problem difficulty / overlap",
            0.0, 1.0, 0.65, 0.05
        )

        df = classification_data(
            n=n,
            difficulty=difficulty
        )
        target_col = "Default"

        dataset_download(
            df,
            "semi_supervised_synthetic_default_dataset.csv",
            "Download semi-supervised synthetic dataset CSV",
            key="download_ssl_builtin_dataset"
        )

        df = choose_row_subset(
            df,
            "ssl_subset_builtin",
            supervised_target=target_col
        )
        y = df[target_col]
    else:
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="ssl_csv"
        )

        if uploaded is None:
            st.info("Choose a CSV file to continue.")
            st.stop()

        df = pd.read_csv(uploaded)

        st.success(
            f"Loaded {df.shape[0]} rows × {df.shape[1]} columns."
        )

        dataset_download(
            df,
            "semi_supervised_uploaded_dataset.csv",
            "Download uploaded semi-supervised dataset CSV",
            key="download_ssl_uploaded_dataset"
        )

        target_col = st.selectbox(
            "Choose binary target",
            df.columns
        )

        df = choose_row_subset(
            df,
            "ssl_subset_upload",
            supervised_target=target_col
        )

        y, mapping = binary_target(df[target_col])

        if y is None:
            st.error("Target must contain exactly two classes.")
            st.stop()

        st.write("Target encoding:", mapping)

    encoding_panel(df, target_col, "ssl_enc")

    all_num = [
        c for c in numeric_cols(df)
        if c != target_col
    ]

    all_cat = [
        c for c in categorical_cols(df)
        if c != target_col
    ]

    selected_num = st.multiselect(
        "Choose numeric predictors",
        all_num,
        default=all_num[:min(6, len(all_num))]
    )

    selected_cat = st.multiselect(
        "Choose categorical predictors (one-hot encoded automatically)",
        all_cat,
        default=[]
    )

    if len(selected_num) + len(selected_cat) < 2:
        st.warning("Choose at least two predictors.")
        st.stop()

    selected_cols = selected_num + selected_cat

    clean = df[selected_cols + [target_col]].dropna().copy()

    if source == "Built-in fraud simulation":
        y = clean[target_col].astype(int)
    else:
        y_raw = clean[target_col]
        y, mapping = binary_target(y_raw)
        y = y.astype(int)

    X = clean[selected_cols].copy()

    # Hold out genuine labelled test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # From remaining training pool, keep only a small portion genuinely labelled
    X_lab, X_unlab, y_lab, y_unlab_true = train_test_split(
        X_temp,
        y_temp,
        train_size=labelled_percent / 100,
        stratify=y_temp,
        random_state=42
    )

    prep = make_preprocessor(
        selected_num,
        selected_cat,
        scale_numeric=True
    )

    X_lab_t = prep.fit_transform(X_lab)
    X_unlab_t = prep.transform(X_unlab)
    X_test_t = prep.transform(X_test)

    # Common PCA basis used only for visualization.
    vis_train_all = np.vstack([X_lab_t, X_unlab_t])
    pca_vis = PCA(n_components=2, random_state=42)
    vis_train_2d = pca_vis.fit_transform(vis_train_all)
    vis_lab_2d = vis_train_2d[:len(X_lab_t)]
    vis_unlab_2d = vis_train_2d[len(X_lab_t):]

    st.subheader("Scatter Plot — Before Semi-Supervised Learning")
    fig_before, ax_before = plt.subplots(figsize=(8, 6))

    y_lab_arr = np.asarray(y_lab)
    lab0 = y_lab_arr == 0
    lab1 = y_lab_arr == 1

    ax_before.scatter(
        vis_unlab_2d[:, 0],
        vis_unlab_2d[:, 1],
        alpha=0.22,
        s=24,
        label="Unlabelled"
    )
    ax_before.scatter(
        vis_lab_2d[lab0, 0],
        vis_lab_2d[lab0, 1],
        s=55,
        marker="o",
        label="Labelled Class 0"
    )
    ax_before.scatter(
        vis_lab_2d[lab1, 0],
        vis_lab_2d[lab1, 1],
        s=55,
        marker="^",
        label="Labelled Class 1"
    )
    ax_before.set_xlabel("Visualization PC1")
    ax_before.set_ylabel("Visualization PC2")
    ax_before.set_title("Before: Small Labelled Set + Large Unlabelled Set")
    ax_before.legend()
    st.pyplot(fig_before)

    st.caption(
        "PCA is used only to display the high-dimensional predictor data in two dimensions. "
        "The semi-supervised method itself uses the full transformed predictor space."
    )

    st.metric("Genuinely labelled training cases", len(X_lab))
    st.metric("Unlabelled cases", len(X_unlab))
    st.metric("Held-out test cases", len(X_test))

    # =====================================================
    # METHOD 1: WRAPPER / SELF-TRAINING
    # =====================================================
    if method == "Wrapper / Self-Training":
        st.markdown(
            "**Stages:** 1) Train supervised model on labelled data → "
            "2) Predict unlabelled cases → "
            "3) Select high-confidence predictions → "
            "4) Add pseudo-labels → "
            "5) Retrain → "
            "6) Repeat if desired → "
            "7) Evaluate on genuinely labelled test data."
        )

        base_model = st.selectbox(
            "Base classifier",
            ["Random Forest", "Logistic Regression"]
        )

        confidence = st.slider(
            "Pseudo-label confidence threshold",
            0.50, 0.99, 0.90, 0.01
        )

        max_rounds = st.slider(
            "Maximum self-training rounds",
            1, 10, 5, 1
        )

        if base_model == "Random Forest":
            def make_ssl_model():
                return RandomForestClassifier(
                    n_estimators=150,
                    max_depth=5,
                    class_weight="balanced",
                    random_state=42
                )
        else:
            def make_ssl_model():
                return LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )

        # Initial supervised model
        initial = make_ssl_model()
        initial.fit(X_lab_t, y_lab)

        p_test_before = initial.predict_proba(X_test_t)[:, 1]
        pred_before = (p_test_before >= 0.50).astype(int)

        before = metric_table(
            y_test,
            pred_before,
            p_test_before
        ).set_index("Metric")["Score"].to_dict()

        # Iterative self-training
        current_X = X_lab_t.copy()
        current_y = np.asarray(y_lab).copy()

        remaining_X = X_unlab_t.copy()
        remaining_indices = np.array(X_unlab.index)

        round_rows = []
        all_pseudo_rows = []

        final_model = initial

        for rnd in range(1, max_rounds + 1):
            if len(remaining_indices) == 0:
                break

            probs = final_model.predict_proba(remaining_X)[:, 1]

            confident_pos = probs >= confidence
            confident_neg = probs <= (1 - confidence)
            mask = confident_pos | confident_neg

            n_new = int(mask.sum())

            if n_new == 0:
                round_rows.append({
                    "Round": rnd,
                    "New pseudo-labels": 0,
                    "Cumulative training size": len(current_y),
                    "Remaining unlabelled": len(remaining_indices),
                    "Stop reason": "No cases met confidence threshold"
                })
                break

            pseudo_y = (probs[mask] >= 0.50).astype(int)

            selected_indices = remaining_indices[mask]

            # Simulation correctness
            true_selected = y_unlab_true.loc[selected_indices].values
            pseudo_correct = float(
                np.mean(true_selected == pseudo_y)
            )

            for idx_val, p, plab, true_lab in zip(
                selected_indices,
                probs[mask],
                pseudo_y,
                true_selected
            ):
                all_pseudo_rows.append({
                    "Round": rnd,
                    "Index": idx_val,
                    "Probability": p,
                    "Pseudo-label": int(plab),
                    "True hidden label": int(true_lab),
                    "Correct": int(plab == true_lab)
                })

            current_X = np.vstack(
                [current_X, remaining_X[mask]]
            )
            current_y = np.concatenate(
                [current_y, pseudo_y]
            )

            remaining_X = remaining_X[~mask]
            remaining_indices = remaining_indices[~mask]

            final_model = make_ssl_model()
            final_model.fit(current_X, current_y)

            p_test_round = final_model.predict_proba(
                X_test_t
            )[:, 1]
            pred_test_round = (
                p_test_round >= 0.50
            ).astype(int)

            round_f1 = f1_score(
                y_test,
                pred_test_round,
                zero_division=0
            )
            round_recall = recall_score(
                y_test,
                pred_test_round,
                zero_division=0
            )

            round_rows.append({
                "Round": rnd,
                "New pseudo-labels": n_new,
                "Pseudo-label correctness": pseudo_correct,
                "Cumulative training size": len(current_y),
                "Remaining unlabelled": len(remaining_indices),
                "Test F1": round_f1,
                "Test Recall": round_recall,
                "Stop reason": ""
            })

        p_test_after = final_model.predict_proba(
            X_test_t
        )[:, 1]
        pred_after = (
            p_test_after >= 0.50
        ).astype(int)

        after = metric_table(
            y_test,
            pred_after,
            p_test_after
        ).set_index("Metric")["Score"].to_dict()

        st.subheader("Before vs After Self-Training")
        comp = pd.DataFrame({
            "Metric": list(before.keys()),
            "Before": list(before.values()),
            "After": [
                after.get(k, np.nan)
                for k in before.keys()
            ]
        })
        st.dataframe(comp, use_container_width=True)

        st.subheader("Round-by-Round Self-Training")
        round_df = pd.DataFrame(round_rows)
        st.dataframe(round_df, use_container_width=True)

        if len(round_df):
            fig, ax = plt.subplots()
            ax.plot(
                round_df["Round"],
                round_df["New pseudo-labels"],
                marker="o",
                label="New pseudo-labels"
            )
            ax.plot(
                round_df["Round"],
                round_df["Remaining unlabelled"],
                marker="o",
                label="Remaining unlabelled"
            )
            ax.set_xlabel("Self-training round")
            ax.set_ylabel("Number of cases")
            ax.set_title("How Unlabelled Cases Change Across Rounds")
            ax.legend()
            st.pyplot(fig)

            if "Test F1" in round_df.columns:
                valid_perf = round_df.dropna(
                    subset=["Test F1"]
                )
                if len(valid_perf):
                    fig2, ax2 = plt.subplots()
                    ax2.plot(
                        valid_perf["Round"],
                        valid_perf["Test F1"],
                        marker="o",
                        label="F1"
                    )
                    ax2.plot(
                        valid_perf["Round"],
                        valid_perf["Test Recall"],
                        marker="o",
                        label="Recall"
                    )
                    ax2.set_xlabel("Self-training round")
                    ax2.set_ylabel("Metric")
                    ax2.set_title(
                        "Test Performance Across Self-Training Rounds"
                    )
                    ax2.legend()
                    st.pyplot(fig2)

        pseudo_df = pd.DataFrame(all_pseudo_rows)

        if len(pseudo_df):
            st.subheader("Pseudo-label Quality — Simulation Only")
            st.metric(
                "Overall pseudo-label correctness",
                f"{pseudo_df['Correct'].mean():.3f}"
            )

            st.download_button(
                "Download pseudo-label details",
                pseudo_df.to_csv(index=False).encode(),
                "wrapper_self_training_pseudolabels.csv",
                "text/csv"
            )

        st.subheader("Scatter Plot — After Wrapper / Self-Training")
        fig_after, ax_after = plt.subplots(figsize=(8, 6))

        final_status = pd.DataFrame({
            "Index": np.array(X_unlab.index),
            "PC1": vis_unlab_2d[:, 0],
            "PC2": vis_unlab_2d[:, 1]
        })

        if len(pseudo_df):
            pseudo_map = (
                pseudo_df.sort_values("Round")
                .drop_duplicates("Index", keep="last")
                .set_index("Index")["Pseudo-label"]
            )
            final_status["Pseudo-label"] = final_status["Index"].map(pseudo_map)
        else:
            final_status["Pseudo-label"] = np.nan

        remaining_set = set(remaining_indices.tolist())
        remaining_mask = final_status["Index"].isin(remaining_set)
        pseudo0 = final_status["Pseudo-label"] == 0
        pseudo1 = final_status["Pseudo-label"] == 1

        ax_after.scatter(
            final_status.loc[remaining_mask, "PC1"],
            final_status.loc[remaining_mask, "PC2"],
            alpha=0.22,
            s=24,
            label="Still unlabelled"
        )
        ax_after.scatter(
            vis_lab_2d[lab0, 0],
            vis_lab_2d[lab0, 1],
            s=55,
            marker="o",
            label="Original labelled Class 0"
        )
        ax_after.scatter(
            vis_lab_2d[lab1, 0],
            vis_lab_2d[lab1, 1],
            s=55,
            marker="^",
            label="Original labelled Class 1"
        )
        ax_after.scatter(
            final_status.loc[pseudo0, "PC1"],
            final_status.loc[pseudo0, "PC2"],
            s=42,
            marker="s",
            label="Pseudo-labelled Class 0"
        )
        ax_after.scatter(
            final_status.loc[pseudo1, "PC1"],
            final_status.loc[pseudo1, "PC2"],
            s=42,
            marker="D",
            label="Pseudo-labelled Class 1"
        )

        ax_after.set_xlabel("Visualization PC1")
        ax_after.set_ylabel("Visualization PC2")
        ax_after.set_title("After: Final Wrapper / Self-Training Status")
        ax_after.legend()
        st.pyplot(fig_after)

        st.subheader("What happened to the remaining unlabelled cases?")
        st.write(
            f"After the final round, **{len(remaining_indices)} cases remain unlabelled**. "
            "These observations never became confident enough to accept as pseudo-labels under the chosen threshold."
        )

        st.subheader("Predict an Unseen Case with the Final Model")
        submitted_ssl, ssl_new_df = input_new_case(
            df,
            selected_num,
            selected_cat,
            "ssl_wrapper_new_case"
        )

        if submitted_ssl:
            try:
                ssl_new_t = prep.transform(ssl_new_df)
                prob = float(
                    final_model.predict_proba(
                        ssl_new_t
                    )[0, 1]
                )
                pred_class = int(prob >= 0.50)

                st.metric(
                    "Predicted probability of positive class",
                    f"{prob:.3f}"
                )
                st.success(
                    f"Predicted class: {pred_class}"
                )
            except Exception as e:
                st.error(
                    f"Could not generate prediction: {e}"
                )

        delta_f1 = (
            after.get("F1", np.nan)
            - before.get("F1", np.nan)
        )

        interpretation = (
            f"Interpretation: Wrapper/self-training began with {len(X_lab)} genuinely labelled cases "
            f"and {len(X_unlab)} unlabelled cases. Across the selected rounds, "
            f"{len(pseudo_df)} cases received pseudo-labels and {len(remaining_indices)} remained unlabelled. "
            f"Test F1 changed from {before.get('F1', np.nan):.3f} "
            f"to {after.get('F1', np.nan):.3f} ({delta_f1:+.3f}). "
            "Conclusion: self-training is useful only when high-confidence pseudo-labels add reliable and informative cases. "
            "Recommendation: inspect both pseudo-label quantity and test performance across rounds rather than assuming that more pseudo-labelled data is always better."
        )

        st.subheader("Conclusion and Recommendation")
        st.write(interpretation)

        log_event(
            "Semi-Supervised Learning",
            "model_tried",
            {
                "method": "Wrapper / Self-Training",
                "base_model": base_model,
                "labelled_percent": labelled_percent,
                "confidence": confidence,
                "rounds": max_rounds,
                "pseudo_labels": len(pseudo_df)
            }
        )

    # =====================================================
    # METHOD 2: CLUSTERING-BASED
    # =====================================================
    else:
        st.markdown(
            "**Stages:** 1) Combine labelled and unlabelled training observations → "
            "2) Cluster observations → "
            "3) Inspect genuinely labelled cases inside each cluster → "
            "4) Give each cluster a majority class when enough labelled evidence exists → "
            "5) Assign those cluster labels to unlabelled cases → "
            "6) Train a supervised model on genuine + inferred labels → "
            "7) Evaluate on the untouched labelled test set."
        )

        k = st.slider(
            "Number of clusters",
            2, 10, 4
        )

        minimum_labelled_per_cluster = st.slider(
            "Minimum genuinely labelled cases needed in a cluster",
            1, 20, 3
        )

        cluster_confidence = st.slider(
            "Minimum labelled majority proportion in a cluster",
            0.50, 1.00, 0.70, 0.05,
            help=(
                "Example: 0.70 means at least 70% of the genuinely labelled cases "
                "inside a cluster must agree on the same class before that cluster "
                "is allowed to pseudo-label its unlabelled members."
            )
        )

        combined_X = np.vstack(
            [X_lab_t, X_unlab_t]
        )

        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=42
        )

        combined_clusters = km.fit_predict(
            combined_X
        )

        lab_clusters = combined_clusters[
            :len(X_lab_t)
        ]
        unlab_clusters = combined_clusters[
            len(X_lab_t):
        ]

        cluster_rows = []
        cluster_to_label = {}

        for cluster_id in range(k):
            mask_lab = lab_clusters == cluster_id
            n_lab_cluster = int(mask_lab.sum())

            if n_lab_cluster == 0:
                cluster_rows.append({
                    "Cluster": cluster_id,
                    "Labelled cases": 0,
                    "Majority class": None,
                    "Majority proportion": None,
                    "Eligible for pseudo-labelling": False
                })
                continue

            labels_here = np.asarray(
                y_lab
            )[mask_lab]

            counts = pd.Series(
                labels_here
            ).value_counts()

            majority_class = int(
                counts.idxmax()
            )

            majority_prop = float(
                counts.max() / counts.sum()
            )

            eligible = (
                n_lab_cluster
                >= minimum_labelled_per_cluster
                and majority_prop
                >= cluster_confidence
            )

            if eligible:
                cluster_to_label[
                    cluster_id
                ] = majority_class

            cluster_rows.append({
                "Cluster": cluster_id,
                "Labelled cases": n_lab_cluster,
                "Majority class": majority_class,
                "Majority proportion": majority_prop,
                "Eligible for pseudo-labelling": eligible
            })

        cluster_summary = pd.DataFrame(
            cluster_rows
        )

        st.subheader("Cluster Evidence from Genuinely Labelled Cases")
        st.dataframe(
            cluster_summary,
            use_container_width=True
        )

        pseudo_mask = np.array([
            c in cluster_to_label
            for c in unlab_clusters
        ])

        pseudo_labels = np.array([
            cluster_to_label[c]
            for c in unlab_clusters[pseudo_mask]
        ], dtype=int)

        selected_unlab_indices = np.array(
            X_unlab.index
        )[pseudo_mask]

        true_hidden = y_unlab_true.loc[
            selected_unlab_indices
        ].values

        pseudo_correctness = (
            float(
                np.mean(
                    pseudo_labels == true_hidden
                )
            )
            if len(pseudo_labels)
            else np.nan
        )

        st.metric(
            "Unlabelled cases assigned a cluster-based pseudo-label",
            len(pseudo_labels)
        )

        st.metric(
            "Still unlabelled",
            len(X_unlab) - len(pseudo_labels)
        )

        if not np.isnan(pseudo_correctness):
            st.metric(
                "Cluster-based pseudo-label correctness — simulation only",
                f"{pseudo_correctness:.3f}"
            )

        # Initial supervised benchmark
        initial_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            class_weight="balanced",
            random_state=42
        )
        initial_model.fit(
            X_lab_t,
            y_lab
        )

        p_before = initial_model.predict_proba(
            X_test_t
        )[:, 1]
        pred_before = (
            p_before >= 0.50
        ).astype(int)

        before = metric_table(
            y_test,
            pred_before,
            p_before
        ).set_index("Metric")["Score"].to_dict()

        if len(pseudo_labels):
            combined_train_X = np.vstack([
                X_lab_t,
                X_unlab_t[pseudo_mask]
            ])

            combined_train_y = np.concatenate([
                np.asarray(y_lab),
                pseudo_labels
            ])

            final_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                class_weight="balanced",
                random_state=42
            )
            final_model.fit(
                combined_train_X,
                combined_train_y
            )
        else:
            final_model = initial_model

        p_after = final_model.predict_proba(
            X_test_t
        )[:, 1]
        pred_after = (
            p_after >= 0.50
        ).astype(int)

        after = metric_table(
            y_test,
            pred_after,
            p_after
        ).set_index("Metric")["Score"].to_dict()

        st.subheader("Before vs After Clustering-Based Semi-Supervision")
        comp = pd.DataFrame({
            "Metric": list(before.keys()),
            "Before": list(before.values()),
            "After": [
                after.get(k_, np.nan)
                for k_ in before.keys()
            ]
        })
        st.dataframe(
            comp,
            use_container_width=True
        )

        # Cluster composition plot
        st.subheader("Cluster Composition")
        plot_df = cluster_summary.copy()

        fig, ax = plt.subplots()
        ax.bar(
            plot_df["Cluster"].astype(str),
            plot_df["Labelled cases"]
        )
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of genuinely labelled cases")
        ax.set_title("Labelled Evidence Available in Each Cluster")
        st.pyplot(fig)

        st.subheader("Scatter Plot — After Clustering-Based Semi-Supervision")
        fig2, ax2 = plt.subplots(figsize=(8, 6))

        cluster_final = pd.DataFrame({
            "Index": np.array(X_unlab.index),
            "PC1": vis_unlab_2d[:, 0],
            "PC2": vis_unlab_2d[:, 1],
            "Cluster": unlab_clusters
        })
        cluster_final["Pseudo-label"] = np.nan

        for cluster_id, class_label in cluster_to_label.items():
            cluster_final.loc[
                cluster_final["Cluster"] == cluster_id,
                "Pseudo-label"
            ] = class_label

        still_unlabelled = cluster_final["Pseudo-label"].isna()
        pseudo0 = cluster_final["Pseudo-label"] == 0
        pseudo1 = cluster_final["Pseudo-label"] == 1

        ax2.scatter(
            cluster_final.loc[still_unlabelled, "PC1"],
            cluster_final.loc[still_unlabelled, "PC2"],
            alpha=0.22,
            s=24,
            label="Still unlabelled"
        )
        ax2.scatter(
            vis_lab_2d[lab0, 0],
            vis_lab_2d[lab0, 1],
            s=55,
            marker="o",
            label="Original labelled Class 0"
        )
        ax2.scatter(
            vis_lab_2d[lab1, 0],
            vis_lab_2d[lab1, 1],
            s=55,
            marker="^",
            label="Original labelled Class 1"
        )
        ax2.scatter(
            cluster_final.loc[pseudo0, "PC1"],
            cluster_final.loc[pseudo0, "PC2"],
            s=42,
            marker="s",
            label="Pseudo-labelled Class 0"
        )
        ax2.scatter(
            cluster_final.loc[pseudo1, "PC1"],
            cluster_final.loc[pseudo1, "PC2"],
            s=42,
            marker="D",
            label="Pseudo-labelled Class 1"
        )

        ax2.set_xlabel("Visualization PC1")
        ax2.set_ylabel("Visualization PC2")
        ax2.set_title("After: Cluster-Based Pseudo-Labelling")
        ax2.legend()
        st.pyplot(fig2)

        st.caption(
            "The same PCA projection used in the initial scatter plot is reused here, "
            "so students can directly compare what changed after clustering-based pseudo-labelling."
        )

        # Detailed cluster assignment export
        cluster_out = X_unlab.copy()
        cluster_out["Cluster"] = unlab_clusters
        cluster_out["Pseudo-label assigned"] = np.nan

        for cluster_id, class_label in cluster_to_label.items():
            cluster_out.loc[
                cluster_out["Cluster"] == cluster_id,
                "Pseudo-label assigned"
            ] = class_label

        cluster_out["True hidden label"] = (
            y_unlab_true
        )

        st.download_button(
            "Download clustering-based pseudo-label results",
            cluster_out.to_csv(
                index=True
            ).encode(),
            "clustering_based_ssl_results.csv",
            "text/csv"
        )

        st.subheader("Predict an Unseen Case with the Final Model")
        submitted_ssl, ssl_new_df = input_new_case(
            df,
            selected_num,
            selected_cat,
            "ssl_cluster_new_case"
        )

        if submitted_ssl:
            try:
                new_t = prep.transform(
                    ssl_new_df
                )
                prob = float(
                    final_model.predict_proba(
                        new_t
                    )[0, 1]
                )
                pred_class = int(
                    prob >= 0.50
                )

                st.metric(
                    "Predicted probability of positive class",
                    f"{prob:.3f}"
                )
                st.success(
                    f"Predicted class: {pred_class}"
                )
            except Exception as e:
                st.error(
                    f"Could not generate prediction: {e}"
                )

        interpretation = (
            f"Interpretation: clustering-based semi-supervision grouped the training observations into {k} clusters. "
            f"{len(pseudo_labels)} unlabelled observations received inferred labels because their clusters contained enough genuinely labelled cases "
            f"with at least {cluster_confidence:.0%} agreement on one class. "
            f"{len(X_unlab) - len(pseudo_labels)} observations remained unlabelled. "
            + (
                f"In this simulation, cluster-based pseudo-label correctness was {pseudo_correctness:.1%}. "
                if not np.isnan(pseudo_correctness)
                else ""
            )
            + f"Test F1 changed from {before.get('F1', np.nan):.3f} to {after.get('F1', np.nan):.3f}. "
            "Conclusion: this method works best when the natural clusters align reasonably well with the target classes. "
            "Recommendation: inspect cluster purity and the number of genuinely labelled cases in each cluster before trusting inferred labels."
        )

        st.subheader("Conclusion and Recommendation")
        st.write(interpretation)

        log_event(
            "Semi-Supervised Learning",
            "model_tried",
            {
                "method": "Clustering-Based",
                "K": k,
                "labelled_percent": labelled_percent,
                "cluster_confidence": cluster_confidence,
                "pseudo_labels": len(pseudo_labels)
            }
        )

# ---------------------------------------------------------------------
# REINFORCEMENT LEARNING
# ---------------------------------------------------------------------
elif section == "Reinforcement Learning":
    st.header("Reinforcement Learning — Multi-Armed Bandit")
    st.write(
        "Business example: a bank chooses among several offers. "
        "The true response rate is initially unknown, so the agent balances exploration and exploitation."
    )

    n_arms = st.slider("Number of offers / actions", 2, 6, 3)
    n_rounds = st.slider("Number of customer interactions", 50, 5000, 500, 50)
    epsilon = st.slider("Exploration rate (epsilon)", 0.00, 0.50, 0.10, 0.01)

    st.caption("For teaching, you can set the hidden true success probability of each action.")
    true_probs = []
    cols = st.columns(n_arms)
    defaults = np.linspace(0.08, 0.20, n_arms)
    for i in range(n_arms):
        with cols[i]:
            true_probs.append(
                st.slider(
                    f"Offer {i+1}",
                    0.01, 0.50,
                    float(round(defaults[i], 2)),
                    0.01,
                    key=f"arm_prob_{i}"
                )
            )

    rng = np.random.default_rng(42)
    counts = np.zeros(n_arms, dtype=int)
    rewards = np.zeros(n_arms, dtype=float)
    cumulative_reward = []
    chosen_actions = []
    interaction_rows = []

    for t in range(n_rounds):
        explore_draw = float(rng.random())

        if explore_draw < epsilon or counts.sum() == 0:
            decision_mode = "Explore"
            action = int(rng.integers(0, n_arms))
        else:
            decision_mode = "Exploit"
            estimates_before = np.divide(
                rewards,
                counts,
                out=np.zeros_like(rewards),
                where=counts > 0
            )
            action = int(np.argmax(estimates_before))

        outcome_draw = float(rng.random())
        reward = 1 if outcome_draw < true_probs[action] else 0

        counts[action] += 1
        rewards[action] += reward
        chosen_actions.append(action)
        cumulative_reward.append(rewards.sum())

        estimates_after = np.divide(
            rewards,
            counts,
            out=np.zeros_like(rewards),
            where=counts > 0
        )

        interaction_rows.append({
            "Interaction": t + 1,
            "Decision mode": decision_mode,
            "Epsilon": epsilon,
            "Action chosen": f"Offer {action + 1}",
            "True success probability of chosen action": float(true_probs[action]),
            "Simulated outcome draw": outcome_draw,
            "Reward": int(reward),
            "Estimated success rate after update": float(estimates_after[action]),
            "Times chosen so far": int(counts[action]),
            "Cumulative reward": float(rewards.sum())
        })

    estimates = np.divide(
        rewards, counts,
        out=np.zeros_like(rewards),
        where=counts > 0
    )

    result = pd.DataFrame({
        "Action": [f"Offer {i+1}" for i in range(n_arms)],
        "Times selected": counts,
        "Observed rewards": rewards.astype(int),
        "Estimated success rate": estimates,
        "True success rate (teaching only)": true_probs
    })
    st.dataframe(result, use_container_width=True)

    interaction_df = pd.DataFrame(interaction_rows)
    with st.expander("View interaction-level learning data"):
        st.dataframe(interaction_df, use_container_width=True)

    dataset_download(
        interaction_df,
        "reinforcement_learning_bandit_interactions.csv",
        "Download RL interaction dataset CSV",
        key="download_rl_interactions"
    )

    fig, ax = plt.subplots()
    ax.plot(range(1, n_rounds + 1), cumulative_reward)
    ax.set_xlabel("Interaction")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Learning Through Repeated Interaction")
    st.pyplot(fig)

    best_est = int(np.argmax(estimates))
    st.success(
        f"After {n_rounds} interactions, the agent currently prefers Offer {best_est+1} "
        f"with estimated success rate {estimates[best_est]:.3f}."
    )
    st.info(
        "Interpretation: epsilon controls exploration. A larger epsilon tries alternatives more often; "
        "a smaller epsilon exploits the currently best-known action more often."
    )
    log_event(
        "Reinforcement Learning",
        "model_tried",
        {"method": "epsilon-greedy bandit", "epsilon": epsilon, "rounds": n_rounds}
    )


# ---------------------------------------------------------------------
# NLP
# ---------------------------------------------------------------------
elif section == "NLP":
    st.header("NLP for BFSI — Sentiment and Financial Text")

    nlp_mode = st.selectbox(
        "Choose NLP activity",
        ["VADER Sentiment", "FinBERT Financial Sentiment", "TF-IDF Keywords"]
    )
    log_event("NLP", "model_tried", {"method": nlp_mode})

    source = st.radio(
        "Text source",
        ["Type / paste text", "Upload CSV"],
        horizontal=True
    )

    texts = []
    row_ids = []

    if source == "Type / paste text":
        user_text = st.text_area(
            "Enter a customer comment, news sentence, financial statement or review",
            height=140,
            value="The bank reported strong profit growth but warned about rising credit losses."
        )
        if user_text.strip():
            texts = [user_text.strip()]
            row_ids = [1]
    else:
        uploaded = st.file_uploader("Upload CSV containing text", type=["csv"], key="nlp_csv")
        if uploaded is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        nlp_df, detected_encoding = read_uploaded_csv(uploaded)
        text_col = st.selectbox("Choose text column", nlp_df.columns.tolist())
        max_rows = st.number_input(
            "Number of text rows to analyse",
            min_value=1,
            max_value=int(len(nlp_df)),
            value=int(min(100, len(nlp_df)))
        )
        work_nlp = nlp_df[[text_col]].dropna().head(int(max_rows))
        texts = work_nlp[text_col].astype(str).tolist()
        row_ids = work_nlp.index.tolist()

    nlp_input_df = pd.DataFrame({
        "Row": row_ids,
        "Text": texts
    })

    dataset_download(
        nlp_input_df,
        "nlp_input_dataset.csv",
        "Download NLP input dataset CSV",
        key="download_nlp_input"
    )

    if nlp_mode == "VADER Sentiment":
        if not VADER_AVAILABLE:
            st.error("Install VADER with: python -m pip install vaderSentiment")
        else:
            rows = []
            for rid, txt in zip(row_ids, texts):
                label, compound, scores = vader_sentiment(txt)
                rows.append({
                    "Row": rid,
                    "Text": txt,
                    "Sentiment": label,
                    "Compound": compound,
                    "Positive": scores["pos"],
                    "Neutral": scores["neu"],
                    "Negative": scores["neg"]
                })
            out_df = pd.DataFrame(rows)
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "Download VADER results",
                out_df.to_csv(index=False).encode(),
                "vader_results.csv",
                "text/csv"
            )
            st.caption(
                "VADER is lexicon/rule based and works well for general sentiment, "
                "but it is not specifically trained on financial language."
            )

    elif nlp_mode == "FinBERT Financial Sentiment":
        st.caption(
            "FinBERT is designed for financial language. The first run may download the model and can be slow."
        )
        try:
            with st.spinner("Loading FinBERT..."):
                finbert = get_finbert_pipeline()
            rows = []
            for rid, txt in zip(row_ids, texts):
                res = finbert(txt[:3000])[0]
                rows.append({
                    "Row": rid,
                    "Text": txt,
                    "Sentiment": res["label"],
                    "Confidence": float(res["score"])
                })
            out_df = pd.DataFrame(rows)
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "Download FinBERT results",
                out_df.to_csv(index=False).encode(),
                "finbert_results.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(
                "FinBERT could not be loaded. Install requirements with: "
                "python -m pip install transformers torch"
            )
            st.caption(str(e))

    else:
        if not texts:
            st.stop()
        vec = TfidfVectorizer(stop_words="english", max_features=30)
        mat = vec.fit_transform(texts)
        scores = np.asarray(mat.mean(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        order = np.argsort(scores)[::-1][:20]
        tfidf_df = pd.DataFrame({
            "Term": terms[order],
            "Mean TF-IDF": scores[order]
        })
        st.dataframe(tfidf_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = tfidf_df.sort_values("Mean TF-IDF")
        ax.barh(plot_df["Term"], plot_df["Mean TF-IDF"])
        ax.set_title("Most Important Terms by TF-IDF")
        st.pyplot(fig)
        st.caption(
            "TF-IDF identifies terms that are important relative to the collection. "
            "It is not a sentiment model by itself."
        )


# ---------------------------------------------------------------------
# FEEDBACK
# ---------------------------------------------------------------------
elif section == "Feedback":
    st.header("Feedback")
    st.write("Your feedback will be used to improve this teaching portal.")

    with st.form("feedback_form"):
        rating = st.slider("Overall usefulness", 1, 5, 4)
        useful = st.selectbox(
            "Most useful section",
            [
                "Regression", "Classification", "K-Means",
                "Hierarchical", "PCA", "Semi-Supervised",
                "Reinforcement Learning", "NLP", "Other"
            ]
        )
        comments = st.text_area(
            "Comments / suggestions",
            placeholder="What helped? What was confusing? What should be added?"
        )
        submit_feedback = st.form_submit_button("Submit feedback")

    if submit_feedback:
        save_feedback(rating, useful, comments)
        st.success("Thank you. Your feedback has been recorded.")


# ---------------------------------------------------------------------
# ADMIN ANALYTICS
# ---------------------------------------------------------------------
elif section == "Admin Analytics":
    st.header("Admin Analytics")

    admin_code = st.text_input("Admin code", type="password")
    if admin_code != ADMIN_CODE:
        st.info("Enter the admin code to view analytics.")
    else:
        cleanup_expired_sessions()

        with db_connect() as con:
            access_df = pd.read_sql_query(
                "SELECT * FROM access_log ORDER BY timestamp DESC", con
            )
            event_df = pd.read_sql_query(
                "SELECT * FROM event_log ORDER BY timestamp DESC", con
            )
            feedback_df = pd.read_sql_query(
                "SELECT * FROM feedback ORDER BY timestamp DESC", con
            )
            active_df = pd.read_sql_query(
                """SELECT session_id, email, city, login_time, last_seen
                   FROM active_sessions
                   WHERE status = 'active'
                   ORDER BY last_seen DESC""",
                con
            )
            history_df = pd.read_sql_query(
                "SELECT * FROM session_history ORDER BY login_time DESC", con
            )

        # Convert timestamps
        for df_, cols in [
            (access_df, ["timestamp"]),
            (event_df, ["timestamp"]),
            (feedback_df, ["timestamp"]),
            (active_df, ["login_time", "last_seen"]),
            (history_df, ["login_time", "logout_time"])
        ]:
            for c in cols:
                if c in df_.columns and len(df_):
                    df_[c] = pd.to_datetime(df_[c], errors="coerce")

        now_ts = pd.Timestamp.now(tz="Asia/Kolkata")

        if len(active_df):
            login_series = active_df["login_time"]
            if getattr(login_series.dt, "tz", None) is None:
                active_df["current_duration_minutes"] = (
                    now_ts.tz_localize(None) - login_series
                ).dt.total_seconds() / 60
            else:
                active_df["current_duration_minutes"] = (
                    now_ts - login_series
                ).dt.total_seconds() / 60

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active users now", len(active_df))
        c2.metric("Maximum allowed", MAX_ACTIVE_USERS)
        c3.metric(
            "Unique users",
            access_df["email"].nunique() if len(access_df) else 0
        )
        c4.metric(
            "Model/activity events",
            len(event_df[event_df["event"] == "model_tried"])
            if len(event_df) else 0
        )

        # -----------------------------
        # Current active users
        # -----------------------------
        st.subheader("Current Active Users")
        if len(active_df):
            active_show = active_df[
                [
                    "email", "city", "login_time",
                    "last_seen", "current_duration_minutes"
                ]
            ].copy()
            active_show["current_duration_minutes"] = (
                active_show["current_duration_minutes"].round(1)
            )
            st.dataframe(active_show, use_container_width=True)
        else:
            st.info("No active users at present.")

        # -----------------------------
        # User frequency / duration summary
        # -----------------------------
        st.subheader("User Frequency and Time Spent")

        if len(history_df):
            user_hist = (
                history_df.groupby(["email", "city"], dropna=False)
                .agg(
                    Sessions=("session_id", "count"),
                    Total_minutes=("duration_minutes", "sum"),
                    Average_session_minutes=("duration_minutes", "mean"),
                    Longest_session_minutes=("duration_minutes", "max"),
                    First_session=("login_time", "min"),
                    Last_session=("login_time", "max")
                )
                .reset_index()
            )

            if len(active_df):
                active_extra = (
                    active_df.groupby(["email", "city"], dropna=False)
                    .agg(
                        Active_minutes=("current_duration_minutes", "sum"),
                        Active_sessions=("session_id", "count")
                    )
                    .reset_index()
                )
                user_hist = user_hist.merge(
                    active_extra,
                    on=["email", "city"],
                    how="outer"
                )
                for col in [
                    "Sessions", "Total_minutes", "Average_session_minutes",
                    "Longest_session_minutes", "Active_minutes", "Active_sessions"
                ]:
                    if col in user_hist.columns:
                        user_hist[col] = user_hist[col].fillna(0)
                user_hist["Total_minutes_including_current"] = (
                    user_hist["Total_minutes"] + user_hist["Active_minutes"]
                )
            else:
                user_hist["Total_minutes_including_current"] = user_hist["Total_minutes"]

            user_hist = user_hist.sort_values(
                "Total_minutes_including_current",
                ascending=False
            )
            st.dataframe(user_hist, use_container_width=True)

            # Plot: total time per user
            top_time = user_hist.head(20).sort_values(
                "Total_minutes_including_current",
                ascending=True
            )
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(
                top_time["email"],
                top_time["Total_minutes_including_current"]
            )
            ax.set_xlabel("Total minutes")
            ax.set_title("Top Users by Total Time Spent")
            st.pyplot(fig)

            # Plot: frequency per user
            top_freq = user_hist.copy()
            if "Active_sessions" in top_freq.columns:
                top_freq["Total_sessions_including_current"] = (
                    top_freq["Sessions"] + top_freq["Active_sessions"]
                )
            else:
                top_freq["Total_sessions_including_current"] = top_freq["Sessions"]

            top_freq = top_freq.sort_values(
                "Total_sessions_including_current",
                ascending=False
            ).head(20).sort_values(
                "Total_sessions_including_current",
                ascending=True
            )

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(
                top_freq["email"],
                top_freq["Total_sessions_including_current"]
            )
            ax.set_xlabel("Number of sessions")
            ax.set_title("Top Users by Access Frequency")
            st.pyplot(fig)

        else:
            st.info(
                "Completed-session duration data will appear after users log out "
                "or sessions expire. Current sessions are shown above."
            )

        # -----------------------------
        # Model / method usage
        # -----------------------------
        st.subheader("Models and Methods Tried")
        model_events = pd.DataFrame()

        if len(event_df):
            model_events = event_df[event_df["event"] == "model_tried"].copy()

        if len(model_events):
            def model_label(row):
                try:
                    d = json.loads(row["details"])
                except Exception:
                    d = {}

                for key in [
                    "model", "method", "base_model",
                    "linkage", "components", "K"
                ]:
                    if key in d:
                        return str(d[key])

                return row["section"]

            model_events["model_or_method"] = model_events.apply(
                model_label,
                axis=1
            )

            model_summary = (
                model_events.groupby(
                    ["section", "model_or_method"],
                    dropna=False
                )
                .agg(
                    Times_tried=("id", "count"),
                    Unique_users=("email", "nunique")
                )
                .reset_index()
                .sort_values("Times_tried", ascending=False)
            )

            st.dataframe(model_summary, use_container_width=True)

            plot_model = model_summary.head(20).sort_values(
                "Times_tried",
                ascending=True
            ).copy()
            plot_model["label"] = (
                plot_model["section"].astype(str)
                + " — "
                + plot_model["model_or_method"].astype(str)
            )

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(plot_model["label"], plot_model["Times_tried"])
            ax.set_xlabel("Times tried")
            ax.set_title("Model / Method Usage")
            st.pyplot(fig)

            # Per-user model usage
            st.subheader("Model Use by User")
            user_model = (
                model_events.groupby(
                    ["email", "section", "model_or_method"],
                    dropna=False
                )
                .size()
                .reset_index(name="Times_tried")
                .sort_values(["email", "Times_tried"], ascending=[True, False])
            )
            st.dataframe(user_model, use_container_width=True)

        else:
            st.info("No model-use events recorded yet.")

        # -----------------------------
        # Usage over time
        # -----------------------------
        st.subheader("Usage Over Time")
        if len(event_df):
            timeline = event_df.copy()
            timeline["date"] = timeline["timestamp"].dt.date
            daily = (
                timeline.groupby("date")
                .size()
                .reset_index(name="Events")
            )

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(daily["date"], daily["Events"], marker="o")
            ax.set_xlabel("Date")
            ax.set_ylabel("Recorded activities")
            ax.set_title("Portal Activity by Day")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # -----------------------------
        # Section frequency
        # -----------------------------
        st.subheader("Sections Accessed")
        if len(event_df):
            section_events = event_df[
                event_df["event"] == "section_opened"
            ]
            if len(section_events):
                section_summary = (
                    section_events.groupby("section")
                    .agg(
                        Opens=("id", "count"),
                        Unique_users=("email", "nunique")
                    )
                    .reset_index()
                    .sort_values("Opens", ascending=False)
                )
                st.dataframe(section_summary, use_container_width=True)

                section_plot = section_summary.sort_values(
                    "Opens",
                    ascending=True
                )
                fig, ax = plt.subplots(figsize=(9, 6))
                ax.barh(section_plot["section"], section_plot["Opens"])
                ax.set_xlabel("Section opens")
                ax.set_title("Most Frequently Accessed Sections")
                st.pyplot(fig)

        # -----------------------------
        # Access log and location
        # -----------------------------
        st.subheader("Access Summary by Stated City / Location")
        if len(access_df):
            city_summary = (
                access_df.groupby("city")
                .size()
                .reset_index(name="Access_records")
                .sort_values("Access_records", ascending=False)
            )
            st.dataframe(city_summary, use_container_width=True)

            city_plot = city_summary.head(15).sort_values(
                "Access_records",
                ascending=True
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(
                city_plot["city"],
                city_plot["Access_records"]
            )
            ax.set_xlabel("Access records")
            ax.set_title("Accesses by Stated City / Location")
            st.pyplot(fig)

        st.subheader("Access Log")
        st.dataframe(access_df, use_container_width=True)

        st.subheader("Session History")
        st.dataframe(history_df, use_container_width=True)

        st.subheader("Feedback")
        st.dataframe(feedback_df, use_container_width=True)

        st.subheader("Download Analytics")
        st.download_button(
            "Download access analytics CSV",
            access_df.to_csv(index=False).encode(),
            "access_analytics.csv",
            "text/csv"
        )
        st.download_button(
            "Download activity analytics CSV",
            event_df.to_csv(index=False).encode(),
            "activity_analytics.csv",
            "text/csv"
        )
        st.download_button(
            "Download session history CSV",
            history_df.to_csv(index=False).encode(),
            "session_history.csv",
            "text/csv"
        )
        st.download_button(
            "Download feedback CSV",
            feedback_df.to_csv(index=False).encode(),
            "feedback.csv",
            "text/csv"
        )

footer()

