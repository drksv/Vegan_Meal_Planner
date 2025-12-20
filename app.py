import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

user_preferences = {}

app = Flask(__name__)

# CORS for Render (must match frontend origin)
CORS(app, resources={r"/*": {"origins": [
    "https://healthtimeout.in",
    "https://www.healthtimeout.in"
]}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://healthtimeout.in"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# GROQ
api_key = os.getenv("GROQ_API_KEY_MEAL")
client = Groq(api_key=api_key)
MODEL = "llama-3.1-8b-instant"

@app.route("/plan", methods=["POST", "OPTIONS"])
def meal_plan():

    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    data = request.json

    message = (
        f"Create a vegan meal plan for: "
        f"age {data.get('age')}, weight {data.get('weight')}, "
        f"height {data.get('height')}, activity {data.get('activity')}, goal {data.get('goal')}."
    )

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

    system_message = (
        f"You are a vegan nutritionist advising a {prefs['age']}-year-old "
        f"{prefs['gender']} who eats {prefs['cuisine']} cuisine and "
        f"consumes {prefs['calories']} calories. Provide a full vegan plan "
        f"with macros, ingredients, and recipes."
    )

    try:
        groq_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ],
            max_tokens=1200,
            temperature=0.5
        )

        reply = groq_response.choices[0].message.content

        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": "Groq failed", "details": str(e)}), 500


@app.route("/")
def home():
    return "Meal Planner API Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)

