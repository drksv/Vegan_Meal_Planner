import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

user_preferences = {}

app = Flask(__name__)

# Strict CORS config for Render
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://healthtimeout.in",
            "https://www.healthtimeout.in"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Add CORS headers for ALL responses (critical for Render)
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://healthtimeout.in"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ---------------- INIT GROQ CLIENT ----------------
api_key = os.getenv("GROQ_API_KEY_MEAL")
client = Groq(api_key=api_key)

MODEL = "llama-3.1-8b-instant"


# ---------------- FIXED: ENDPOINT NAME MUST MATCH FRONTEND ----------------
@app.route("/plan", methods=["POST", "OPTIONS"])
def meal_plan():

    # Handle CORS preflight
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.json
    message = (
        f"Create a vegan meal plan for a person with: "
        f"age {data.get('age')}, weight {data.get('weight')}, height {data.get('height')}, "
        f"activity level {data.get('activity')}, goal {data.get('goal')}."
    )

    user_id = data.get("user_id", "default_user_123")

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

    system_message = (
        f"You are a certified vegan nutritionist. The user is a {prefs['age']}-year-old "
        f"{prefs['gender']} who prefers {prefs['cuisine']} cuisine and consumes "
        f"{prefs['calories']} calories. Provide a full vegan meal plan with macro split, "
        f"ingredients, timings, and recipes."
    )

    try:
        groq_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ],
            max_tokens=600,
            temperature=0.7
        )

        reply = groq_response.choices[0].message.content

        return jsonify({"response": reply})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": "Groq API failed", "details": str(e)}), 500


@app.route("/")
def home():
    return "Meal Planner API Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=True)
