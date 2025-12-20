import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

user_preferences = {}

app = Flask(__name__)

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


# ---------- GROQ CONFIG ----------
api_key = os.getenv("GROQ_API_KEY_MEAL")
client = Groq(api_key=api_key)
MODEL = "llama-3.1-8b-instant"

MAX_TOKENS = 1200
TEMPERATURE = 0.5
MAX_CONTINUATIONS = 3   # safety limit


def generate_with_continuation(messages):
    """Auto-continue if Groq output is truncated"""
    full_reply = ""
    current_messages = messages.copy()

    for _ in range(MAX_CONTINUATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=current_messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )

        choice = response.choices[0]
        content = choice.message.content
        finish_reason = choice.finish_reason

        full_reply += content

        if finish_reason != "length":
            break  # completed normally

        # Ask model to continue
        current_messages.append(
            {"role": "assistant", "content": content}
        )
        current_messages.append(
            {"role": "user", "content": "Continue exactly from where you stopped."}
        )

    return full_reply


@app.route("/plan", methods=["POST", "OPTIONS"])
def meal_plan():

    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    data = request.json
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
    f"You are a vegan nutritionist.\n\n"
    f"Give EXACTLY ONE vegan meal idea for TODAY ONLY.\n"
    f"Do NOT provide multiple ideas.\n"
    f"Do NOT number anything.\n"
    f"Do NOT use bullet points.\n\n"

    f"Structure the response using ONLY these section titles:\n"
    f"Meal\n"
    f"Macros\n"
    f"Ingredients\n"
    f"Recipe\n\n"

    f"Rules:\n"
    f"- Meal: one short sentence\n"
    f"- Macros: one line with calories, protein, carbs, fats\n"
    f"- Ingredients: maximum 5 items in a single line\n"
    f"- Recipe: maximum 3 short steps in one paragraph\n\n"

    f"Keep total response under 180 words.\n"
    f"End cleanly. Do NOT stop mid-sentence."
)



   user_message = (
    f"User details:\n"
    f"Age: {data.get('age')}\n"
    f"Weight: {data.get('weight')} kg\n"
    f"Height: {data.get('height')} cm\n"
    f"Activity: {data.get('activity')}\n"
    f"Goal: {data.get('goal')}\n"
)

    try:
        reply = generate_with_continuation([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])

        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"error": "Groq failed", "details": str(e)}), 500


@app.route("/")
def home():
    return "Meal Planner API Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)


