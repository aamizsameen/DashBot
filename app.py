import os, json, io, traceback
from typing import Optional
from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# In-memory store keyed by session
_dataframes: dict = {}


def get_df() -> Optional[pd.DataFrame]:
    sid = session.get("sid")
    return _dataframes.get(sid) if sid else None


def compute_analytics(df: pd.DataFrame) -> dict:
    info = {"rows": len(df), "cols": len(df.columns), "columns": []}
    for col in df.columns:
        c = {"name": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isnull().sum()),
             "unique": int(df[col].nunique())}
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            c.update({"min": _safe(desc.get("min")), "max": _safe(desc.get("max")),
                       "mean": _safe(desc.get("mean")), "median": _safe(df[col].median()),
                       "std": _safe(desc.get("std")), "q1": _safe(desc.get("25%")),
                       "q3": _safe(desc.get("75%")),
                       "histogram": _histogram(df[col])})
            c["is_numeric"] = True
        else:
            top = df[col].value_counts().head(10)
            c["top_values"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]
            c["is_numeric"] = False
        info["columns"].append(c)
    # Correlation matrix for numeric cols
    num_cols = df.select_dtypes(include="number")
    if len(num_cols.columns) >= 2:
        corr = num_cols.corr().round(3)
        info["correlation"] = {"labels": list(corr.columns),
                               "matrix": corr.values.tolist()}
    # Sample rows (more for builder mode)
    info["sample"] = json.loads(df.head(50).to_json(orient="records", date_format="iso"))
    info["all_data"] = json.loads(df.to_json(orient="records", date_format="iso")) if len(df) <= 5000 else None
    return info


def _safe(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return round(float(v), 4)


def _histogram(series, bins=20):
    clean = series.dropna()
    if clean.empty:
        return {"edges": [], "counts": []}
    counts, edges = pd.cut(clean, bins=bins, retbins=True)
    hist = counts.value_counts(sort=False)
    return {"edges": [round(float(e), 4) for e in edges],
            "counts": [int(v) for v in hist.values]}


@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = os.urandom(12).hex()
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(error="No file provided"), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify(error="Only CSV files are supported"), 400
    try:
        df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
        sid = session.get("sid", os.urandom(12).hex())
        session["sid"] = sid
        _dataframes[sid] = df
        analytics = compute_analytics(df)
        return jsonify(analytics)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/chat", methods=["POST"])
def chat():
    df = get_df()
    if df is None:
        return jsonify(reply="Please upload a CSV file first so I can help you analyze it.")
    user_msg = request.json.get("message", "")
    if not user_msg.strip():
        return jsonify(reply="Please type a question about your data.")

    # Build context for the LLM
    summary_lines = [f"Dataset: {len(df)} rows, {len(df.columns)} columns.",
                     f"Columns: {', '.join(df.columns.tolist())}",
                     "Dtypes: " + ', '.join(f"{c}({df[c].dtype})" for c in df.columns),
                     "First 3 rows (JSON):"]
    summary_lines.append(df.head(3).to_json(orient="records", date_format="iso"))
    desc = df.describe(include="all").to_string()
    summary_lines.append("Describe:\n" + desc)
    context = "\n".join(summary_lines)

    prompt = (f"You are a data analyst assistant. The user uploaded a CSV dataset.\n"
              f"Here is the dataset context:\n{context}\n\n"
              f"User question: {user_msg}\n\n"
              "Provide a clear, concise, and helpful answer. If the user asks for a chart, "
              "respond with a JSON block wrapped in ```chart ... ``` with keys: type (bar|line|pie|scatter), "
              "labels (list), datasets (list of {{label, data}}), title. Otherwise answer in markdown.")
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return jsonify(reply=resp.text)
    except Exception as e:
        traceback.print_exc()
        return jsonify(reply=f"AI error: {e}")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
