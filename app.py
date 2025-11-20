import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

# Use a dedicated key for meal planner
client = Groq(api_key=os.getenv("GROQ_API_KEY_MEAL"))
MODEL = "llama3-70b-8192"   # VALID on Groq


def compute_tdee(weight, height, age, activity, goal, gender):
    weight, height, age = float(weight), float(height), int(age)

    # Basal Metabolic Rate
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Total Daily Energy Expenditure
    tdee = bmr * float(activity)

    # Adjust for goal
    if goal == "loss":
        tdee -= 400
    elif goal == "gain":
        tdee += 350

    return int(max(1200, tdee))  # never below 1200 kcal


@app.route("/plan", methods=["POST"])
def plan():
    try:
        data = request.json

        age = data.get("age")
        weight = data.get("weight")
        height = data.get("height")
        activity = data.get("activity", 1.3)
        goal = data.get("goal", "maintain")
        gender = data.get("gender", "female")

        calories = compute_tdee(weight, height, age, activity, goal, gender)

        system_prompt = (
            "You are a certified nutritionist specializing in vegan Indian diets. "
            "Always format meals cleanly with calories and macros."
        )

        user_prompt = f"""
Create a full-day **100% vegan Indian meal plan** within **{calories} kcal**.
Return exactly 5 meals:

For each meal include:
- Meal Name
- Food items + portions
- Calories per meal
- Macro line EXACTLY like this:
  P: 00 g  C: 00 g  F: 00 g

Do not exceed total calories.
"""

        # GROQ API CALL
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        reply = response.choices[0].message["content"]

        return jsonify({"plan": reply, "calories": calories})

    except Exception as e:
        print("Meal Planner Error:", e)
        return jsonify({"error": "Meal planner failed", "details": str(e)}), 500


@app.route("/", methods=["GET"])
def root():
    return "Meal Planner API Running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
