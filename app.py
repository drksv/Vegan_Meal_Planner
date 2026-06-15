import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

user_preferences = {}

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": [
    "https://healthtimeout.in",
    "https://www.healthtimeout.in",
    "https://healthtimeout.infinityfree.me",
    "https://healthtimeout.infinityfree.me/wp/"
]}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://healthtimeout.in"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


# ---------- GROQ CONFIG ----------
api_key = os.getenv("GROQ_API_KEY_MEAL")
client = Groq(api_key=api_key)

MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 550          # SAFE for 4 meals
TEMPERATURE = 0.3


def generate_response(messages):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    return response.choices[0].message.content


@app.route("/plan", methods=["POST", "OPTIONS"])
def meal_plan():

    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    data = request.json or {}
    user_id = data.get("user_id", "default_user")

    prefs = user_preferences.get(
        user_id,
        {
            "age": data.get("age"),
            "gender": "unknown",
            "calories": "1800",
            "cuisine": "Indian"
        }
    )
    user_preferences[user_id] = prefs

    # ---------- SYSTEM PROMPT ----------
    system_message = (
        "You are a vegan nutritionist.\n\n"
        "Create a vegan meal plan for TODAY ONLY with exactly four meals.\n"
        "The meals must be Breakfast, Lunch, Evening Snack, and Dinner.\n\n"

        "Rules:\n"
        "Do NOT provide extra meals.\n"
        "Do NOT number anything.\n"
        "Do NOT use bullet points.\n"
        "Do NOT add explanations or tips.\n\n"

        "Use ONLY these section titles in this exact order:\n"
        "Breakfast\n"
        "Lunch\n"
        "Evening Snack\n"
        "Dinner\n\n"

        "For EACH section provide exactly:\n"
        "One short meal sentence.\n"
        "One macros line with calories, protein, carbs, fats.\n"
        "One ingredients line with max 5 items.\n"
        "One recipe paragraph with max 3 short steps.\n\n"

        "Keep the entire response under 280 words.\n"
        "End cleanly. Do NOT stop mid-sentence."
    )

    # ---------- USER PROMPT ----------
    user_message = (
        f"User details:\n"
        f"Age: {data.get('age')}\n"
        f"Weight: {data.get('weight')} kg\n"
        f"Height: {data.get('height')} cm\n"
        f"Activity: {data.get('activity')}\n"
        f"Goal: {data.get('goal')}"
    )

    try:
        reply = generate_response([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])

        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({
            "error": "Groq failed",
            "details": str(e)
        }), 500


@app.route("/")
def home():
    return "Meal Planner API Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)
