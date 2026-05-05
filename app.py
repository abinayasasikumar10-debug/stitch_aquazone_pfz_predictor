"""
AquaZone PFZ Predictor – Flask Web Application
"""

from flask import Flask, render_template, request, jsonify
import os
from predictor.pfz_predictor import predict

app = Flask(__name__)


@app.route("/")
def index():
    """Render the main map-based input page."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_view():
    """
    Handle coordinate submission and render the prediction result.

    Accepts both form POST (from the HTML form) and JSON POST (for API use).
    Also accepts GET with query parameters ?lat=...&lon=... for convenience.
    """
    if request.method == "POST":
        if request.is_json:
            data = request.get_json()
            lat = data.get("latitude")
            lon = data.get("longitude")
        else:
            lat = request.form.get("latitude")
            lon = request.form.get("longitude")
    else:
        lat = request.args.get("lat") or request.args.get("latitude")
        lon = request.args.get("lon") or request.args.get("longitude")

    errors = []
    if lat is None or lat == "":
        errors.append("Latitude is required.")
    if lon is None or lon == "":
        errors.append("Longitude is required.")

    if errors:
        if request.is_json:
            return jsonify({"errors": errors}), 400
        return render_template("index.html", errors=errors)

    try:
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError):
        err = ["Invalid coordinates. Please enter numeric values."]
        if request.is_json:
            return jsonify({"errors": err}), 400
        return render_template("index.html", errors=err)

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        err = ["Coordinates out of range. Latitude must be -90..90, longitude -180..180."]
        if request.is_json:
            return jsonify({"errors": err}), 400
        return render_template("index.html", errors=err)

    result = predict(lat, lon)

    if request.is_json:
        return jsonify(result)

    return render_template(
        "result.html",
        latitude=lat,
        longitude=lon,
        result=result,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for PFZ prediction."""
    data = request.get_json(silent=True) or {}
    lat = data.get("latitude")
    lon = data.get("longitude")

    if lat is None or lon is None:
        return jsonify({"error": "latitude and longitude are required"}), 400

    try:
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError):
        return jsonify({"error": "latitude and longitude must be numeric"}), 400

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return jsonify({"error": "coordinates out of valid range"}), 400

    return jsonify(predict(lat, lon))


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=5000)
