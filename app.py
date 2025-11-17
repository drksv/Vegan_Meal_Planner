import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment/config
HF_API_KEY = os.getenv("HF_API_KEY") 
HF_MODEL = os.getenv("HF_MODEL")
HF_TIMEOUT = int(os.getenv("HF_TIMEOUT", "60"))  # seconds

if not HF_API_KEY:
    logger.warning("HF_API_KEY not set. The app will fail to call Hugging Face without it.")

HF_ENDPOINT = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}


def call_hf(prompt: str, max_new_tokens: int = 350, temperature: float = 0.6):
    """
    Call Hugging Face text generation endpoint and return text response.
    Handles a few common response types.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }
    try:
        resp = requests.post(HF_ENDPOINT, headers=HEADERS, json=payload, timeout=HF_TIMEOUT)
    except requests.RequestException as e:
        logger.exception("Request to Hugging Face failed")
        return {"error": "Request to Hugging Face failed", "details": str(e)}, 500

    if resp.status_code != 200:
        logger.error("Hugging Face returned status %s -> %s", resp.status_code, resp.text)
        return {"error": "Hugging Face API error", "status_code": resp.status_code, "details": resp.text}, 502

    try:
        parsed = resp.json()
    except ValueError:
        logger.error("Invalid JSON from HF: %s", resp.text)
        return {"error": "Invalid JSON from Hugging Face", "raw": resp.text}, 502

    # Models sometimes return: [{"generated_text": "..."}] or {"generated_text": "..."} or {"error": "..."}
    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict) and "generated_text" in parsed[0]:
        return parsed[0]["generated_text"]
    if isinstance(parsed, dict) and "generated_text" in parsed:
        return parsed["generated_text"]
    if isinstance(parsed, dict) and "error" in parsed:
        logger.error("HF model error: %s", parsed["error"])
        return {"error": "Hugging Face model error", "details": parsed["error"]}, 502

    # Fallback: try first string item if HF returns list of strings
    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], str):
        return parsed[0]

    # Unknown structure — return stringified JSON
    return str(parsed)


def compute_tdee(weight_kg: float, height_cm: float, age: int, activity: float, gender: str = "neutral", goal: str = "maintain"):
    """
    Mifflin-St Jeor BMR + activity to compute TDEE.
    Gender handling: male/female/neutral (neutral uses female baseline minus 0).
    """
    try:
        weight = float(weight_kg)
        height = float(height_cm)
        age = int(age)
        activity = float(activity)
    except Exception:
        raise ValueError("Invalid numeric inputs for weight/height/age/activity.")

    # We'll default to 'neutral' BMR by averaging male/female formula (safe default)
    # male: +5, female: -161 -> neutral ≈ (-78)
    # But to keep things simple and safe, use female constant unless gender == 'male'
    if gender and gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    tdee = bmr * activity

    if goal == "loss":
        tdee -= 400
    elif goal == "gain":
        tdee += 350

    return max(1200, round(tdee))  # floor to 1200 kcal for safety


def build_plan_prompt(calories: int, requirements: str = ""):
    prompt = f"""
You are a certified registered vegan dietitian. Produce a single-day, 100% PLANT-BASED (vegan) meal plan.

REQUIREMENTS:
- Total daily calories: {calories} kcal
- Provide 5 meals: Breakfast, Snack 1, Lunch, Snack 2, Dinner
- For each meal include:
  - A short name/title
  - Bullet list of foods / portions (only simple Indian plant-based foods)
  - Per-meal calories (e.g., "Breakfast - 420 kcal")
  - Macro line formatted EXACTLY like this:
    "P: {{protein}} g C: {{carbs}} g F: {{fat}} g"
- At the end include TOTAL calories and total macros

Extra constraints: {requirements}

Return only the meal plan text.
"""
    return prompt



@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "AI Vegan Meal Planner (Hugging Face) is running."})


@app.route("/plan", methods=["POST"])
def plan():
    """
    POST JSON:
    {
      "age": 30,
      "weight": 70,
      "height": 170,
      "activity": 1.5,
      "goal": "maintain" | "loss" | "gain",
      "gender": "male" | "female" (optional),
      "requirements": "high protein, no soy" (optional)
    }
    """
    data = request.get_json(force=True)
    try:
        weight = data.get("weight")
        height = data.get("height")
        age = data.get("age")
        activity = data.get("activity", 1.3)
        goal = data.get("goal", "maintain")
        gender = data.get("gender", "neutral")
        reqs = data.get("requirements", "")
        # compute calories
        calories = compute_tdee(weight, height, age, activity, gender, goal)
    except Exception as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400

    prompt = build_plan_prompt(calories, reqs)
    logger.info("Calling HF for plan with calories=%s", calories)
    result = call_hf(prompt)

    # If HF returned an error dict, propagate status
    if isinstance(result, tuple) and isinstance(result[0], dict):
        return jsonify(result[0]), result[1]

    return jsonify({"calories": calories, "plan_text": result})


@app.route("/single-meal", methods=["POST"])
def single_meal():
    """
    POST JSON:
    {
      "meal": "Breakfast",
      "target_calories": 400,
      "requirements": "high-protein, Indian ingredients preferred"
    }
    Returns a regenerated single meal (name, items, per-meal calories, macros).
    """
    data = request.get_json(force=True)
    meal = data.get("meal")
    target = data.get("target_calories")
    reqs = data.get("requirements", "")

    if not meal or not target:
        return jsonify({"error": "Please provide 'meal' and 'target_calories'"}), 400

    prompt = f"""
You are a vegan registered dietitian. Regenerate ONLY ONE meal named "{meal}".
Target per-meal calories: {target} kcal.
Include:
 - Meal title
 - Foods/portions
 - Per-meal calories (e.g., "{meal} - {target} kcal")
 - Macro breakdown "P: x g C: y g F: z g"
Use only plant-based ingredients. {reqs}
Return only the meal info (no full-day plan).
"""
    logger.info("Calling HF single-meal for %s calories", target)
    result = call_hf(prompt)

    if isinstance(result, tuple) and isinstance(result[0], dict):
        return jsonify(result[0]), result[1]

    return jsonify({"meal": meal, "meal_text": result})


if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=(os.getenv("FLASK_DEBUG") == "1"))


