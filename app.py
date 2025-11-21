import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

# ----------------- IN-MEMORY USER PREFERENCES -----------------
user_preferences = {}

app = Flask(__name__)
CORS(app)

# ----------------- INIT GROQ CLIENT -----------------
api_key = os.getenv("GROQ_API_KEY_MEAL")
if not api_key:
    print("WARNING: GROQ_API_KEY_MEAL is not set.")

client = Groq(api_key=api_key)

MODEL = "llama-3.1-8b-instant"


@app.route("/meal", methods=["POST"])
def meal_plan():
    """Vegan meal planner endpoint"""
    data = request.json
    message = data.get("message")
    user_id = data.get("user_id", "default_user_123")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Load or set default preferences
    prefs = user_preferences.get(
        user_id,
        {
            "age": "unknown",
            "gender": "unknown",
            "calories": "1800",
            "cuisine": "Indian"
        }
    )

    # Save back to memory
    user_preferences[user_id] = prefs

    # SYSTEM MESSAGE — VEGAN MEAL PLANNER
    system_message = (
        f"You are a certified vegan nutrition specialist. The user is a "
        f"{prefs['age']}-year-old {prefs['gender']} who prefers {prefs['cuisine']} cuisine "
        f"and consumes around {prefs['calories']} calories per day. "
        f"Create detailed vegan meal plans including macronutrient breakdown, ingredients, "
        f"portion sizes, and easy preparation steps. Avoid animal products completely. "
        f"Include optional substitutes and health benefits."
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

        # ---------------- FIX: ACCESS MESSAGE CONTENT CORRECTLY ----------------
        reply = groq_response.choices[0].message["content"]

        return jsonify({"response": reply})

    except Exception as e:
        print(f"ERROR processing Groq API request: {e}")
        return jsonify({"error": "Groq API failed", "details": str(e)}), 500


@app.route("/")
def home():
    return "Vegan Meal Planner API Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))  # Different port if running both APIs
    app.run(host="0.0.0.0", port=port, debug=True)
