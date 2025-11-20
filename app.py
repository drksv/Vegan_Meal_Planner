import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY_MEAL"))
MODEL = "mistral-7b-instant"


def compute_tdee(weight, height, age, activity, goal, gender):
    weight, height, age = float(weight), float(height), int(age)

    if gender == "male":
        bmr = 10*weight + 6.25*height - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height - 5*age - 161

    tdee = bmr * float(activity)

    if goal == "loss":
        tdee -= 400
    elif goal == "gain":
        tdee += 350

    return int(max(1200, tdee))


@app.route("/plan", methods=["POST"])
def plan():
    data = request.json

    age = data.get("age")
    weight = data.get("weight")
    height = data.get("height")
    activity = data.get("activity", 1.3)
    goal = data.get("goal", "maintain")
    gender = data.get("gender", "female")

    calories = compute_tdee(weight, height, age, activity, goal, gender)

    prompt = f"""
Create a full-day **vegan Indian meal plan** within {calories} kcal.
Return 5 meals with:
- Meal name
- Food items + portions
- Calories per meal
- Macro line exactly like: P: 00 g C: 00 g F: 00 g
"""

    reply = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7
    ).choices[0].message["content"]

    return jsonify({"plan": reply, "calories": calories})


@app.route("/", methods=["GET"])
def root():
    return "Meal Planner API Running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)




