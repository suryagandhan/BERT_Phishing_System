"""
Flask Web Application for BERT-based Phishing Detection
Provides web interface, single-shot and batch API, with fallback mock model.
"""

import os
import sys
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

# ------------------------------------------------------------------ Predictor setup
# Add project root to sys.path to import predict module
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

try:
    from predict import BertPhishingPredictor  # Your real predictor implementation
except Exception as exc:
    logging.warning("Could not import real predictor, using mock instead: %s", exc)

    class BertPhishingPredictor:  # Fallback mock predictor
        def __init__(self):
            self.model_loaded = False

        def _quick(self, text):
            suspicious = any(w in text.lower() for w in ("phishing", "verify", "urgent", "click"))
            prob = 0.85 if suspicious else 0.15
            return {
                "text": text[:120] + "…" if len(text) > 120 else text,
                "prediction": int(suspicious),
                "phishing_probability": prob,
                "confidence": prob,
                "risk_level": "HIGH" if suspicious else "LOW",
                "recommendation": "DEMO MODE – train a real model for full accuracy"
            }

        def analyze_url(self, url):
            return self._quick(url)

        def analyze_email(self, body, *_):
            return self._quick(body)

        def predict_batch(self, texts):
            return [self._quick(text) for text in texts]

        def get_model_info(self):
            return {"status": "Mock predictor", "device": "CPU", "model_path": "N/A"}

# Instantiate a single global predictor instance
predictor = BertPhishingPredictor()

# ------------------------------------------------------------------ Flask app setup

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phishing-app")

# ------------------------------------------------------------------ Routes

@app.route("/", methods=["GET"])
def index():
    """Render main page with input form."""
    return render_template("index.html", model_info=predictor.get_model_info())

@app.route("/analyze", methods=["POST"])
def analyze():
    text_input = request.form.get("input_text", "").strip()
    if not text_input:
        flash("Please enter some text.", "warning")
        return redirect(url_for("index"))

    # Auto detect URL or email
    analysis_type = "url" if text_input.lower().startswith(("http://", "https://", "www.")) else "email"

    start_time = time.perf_counter()
    if analysis_type == "url":
        result = predictor.analyze_url(text_input)
    else:
        email_subject = request.form.get("email_subject")
        result = predictor.analyze_email(text_input, email_subject if email_subject else None)

    # Add probability key for your template
    result["probability"] = result.get("phishing_probability", result.get("confidence", 0))
    result.update(
        analysis_type=analysis_type,
        elapsed=round(time.perf_counter() - start_time, 3),
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    app.logger.info("Sending to template result keys: %s", list(result.keys()))
    app.logger.info("Result content snippet: %s", str(result)[:500])

    return render_template("result.html", result=result)




# -------- JSON API --------

@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON POST endpoint:
    Payload: {"text": "...", "type": "url|email|auto", "subject": "..."}
    Response: JSON with prediction result
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify(error="Missing 'text' parameter."), 400

    analysis_type = data.get("type", "auto")
    if analysis_type == "auto":
        analysis_type = "url" if text.lower().startswith(("http://", "https://", "www.")) else "email"

    if analysis_type not in {"url", "email"}:
        return jsonify(error="Invalid 'type' parameter, must be 'url', 'email', or 'auto'."), 400

    if analysis_type == "url":
        result = predictor.analyze_url(text)
    else:
        subject = data.get("subject")
        result = predictor.analyze_email(text, subject)

    result.update(
        analysis_type=analysis_type,
        timestamp=datetime.utcnow().isoformat(timespec="seconds")
    )
    return jsonify(result)

@app.route("/api/batch", methods=["POST"])
def api_batch_predict():
    """
    JSON POST endpoint for batch predictions:
    Payload: {"texts": ["text1", "text2", ...]}
    Response: {"results":[...], "total": N, "timestamp": "..."}
    """
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    if not isinstance(texts, list):
        return jsonify(error="'texts' must be a list"), 400
    if len(texts) > 100:
        return jsonify(error="Maximum batch size is 100"), 400

    results = predictor.predict_batch(texts)
    return jsonify(
        results=results,
        total=len(texts),
        timestamp=datetime.utcnow().isoformat(timespec="seconds")
    )

@app.route("/api/model-info")
def api_model_info():
    """Return info about the loaded model."""
    info = predictor.get_model_info()
    info["timestamp"] = datetime.utcnow().isoformat(timespec="seconds")
    return jsonify(info)

@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify(
        status="healthy",
        model_loaded=getattr(predictor, "model_loaded", True),
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
    )

# ------------------------------------------------------------------ Error Handlers

@app.errorhandler(404)
def not_found(e):
    flash("Page not found.", "warning")
    return redirect(url_for("index")), 404

@app.errorhandler(500)
def internal_error(e):
    logger.exception(f"Internal server error: {e}")
    flash("Internal server error occurred.", "danger")
    return redirect(url_for("index")), 500

# ------------------------------------------------------------------ Jinja2 Template Filters

@app.template_filter("percentage")
def percentage_filter(value):
    """Convert float (0.0-1.0) to percent string e.g. 0.85 → '85.0%'"""
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "0.0%"

@app.template_filter("risk_color")
def risk_color_filter(level):
    """Map risk level strings to Bootstrap contextual color classes."""
    mapping = {"LOW": "success", "MEDIUM": "warning", "HIGH": "danger"}
    return mapping.get(level.upper(), "secondary")

# ------------------------------------------------------------------ Run Application

if __name__ == "__main__":
    # Create folders if not exist for Flask static/template lookup
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)

    app.run(host="0.0.0.0", port=5000, debug=True)

