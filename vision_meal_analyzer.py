# Import necessary libraries
import os
import requests
from PIL import Image
from torchvision import transforms
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NUTRITIONIX_API_KEY = os.getenv('NUTRITIONIX_API_KEY')
NUTRITIONIX_APP_ID = os.getenv('NUTRITIONIX_APP_ID')
LLAMA_MODEL_PATH = os.getenv('LLAMA_MODEL_PATH')

# Load the Llama Vision Model
def load_vision_model():
    model = torch.load(LLAMA_MODEL_PATH)
    model.eval()
    return model

# Preprocess Input Image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Analyze Meal Image
def analyze_image(model, image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(input_tensor)
    # Example output parsing for ingredients and quantities
    ingredients_with_quantities = {
        "chicken breast": "200g",
        "broccoli": "150g",
        "olive oil": "1 tbsp"
    }
    return ingredients_with_quantities

# Fetch Nutrition Data from Nutritionix
def get_nutrition_data(ingredient, quantity):
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"query": f"{quantity} of {ingredient}"}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Estimate Macronutrients with Vision Model
def estimate_macronutrients_with_vision_model(model, ingredient, quantity):
    prompt = f"""
    Estimate macronutrients for:
    Ingredient: {ingredient}
    Quantity: {quantity}
    Include calories, protein, fat, and carbohydrates.
    """
    # Adjust according to model input requirements
    response = {
        "calories": 250,
        "protein_g": 30,
        "fat_g": 8,
        "carbs_g": 5
    }
    return response

# Calculate Totals for Macronutrients
def calculate_totals(nutrition_data_list):
    totals = {"calories": 0, "protein_g": 0, "fat_g": 0, "carbs_g": 0}
    for data in nutrition_data_list:
        if data:
            totals["calories"] += data["calories"]
            totals["protein_g"] += data["protein_g"]
            totals["fat_g"] += data["fat_g"]
            totals["carbs_g"] += data["carbs_g"]
    return totals

# Assess Healthiness and Alternatives
def assess_healthiness_and_alternatives(model, totals):
    prompt = f"""
    Based on the following macronutrient summary, determine if the meal is healthy:
    {totals}
    If unhealthy, suggest healthier alternatives.
    """
    response = {
        "is_healthy": False,
        "suggestions": ["Use grilled tofu instead of chicken", "Reduce olive oil to 1 tsp"]
    }
    return response

# Generate Chatbot Response
def generate_chatbot_response(ingredients, nutrition_data, totals, health_assessment):
    response = "Here's the analysis of your meal:\n\n"
    for ingredient, data in zip(ingredients.items(), nutrition_data):
        response += summarize_nutrition(ingredient[0], ingredient[1], data) + "\n"
    response += f"\n**Total Macronutrients**:\n- Calories: {totals['calories']} kcal\n"
    response += f"- Protein: {totals['protein_g']}g\n- Fat: {totals['fat_g']}g\n"
    response += f"- Carbs: {totals['carbs_g']}g\n\n"
    if not health_assessment["is_healthy"]:
        response += "This meal is **unhealthy**. Here are some healthier alternatives:\n"
        response += "\n".join(health_assessment["suggestions"]) + "\n"
    else:
        response += "This meal is **healthy**. Great choice!\n"
    return response

# Main Program
def main():
    model = load_vision_model()
    image_path = input("Enter the path to the meal image: ")
    ingredients_with_quantities = analyze_image(model, image_path)
    nutrition_data = []
    for ingredient, quantity in ingredients_with_quantities.items():
        data = get_nutrition_data(ingredient, quantity)
        if not data:
            data = estimate_macronutrients_with_vision_model(model, ingredient, quantity)
        nutrition_data.append(data)
    totals = calculate_totals(nutrition_data)
    health_assessment = assess_healthiness_and_alternatives(model, totals)
    response = generate_chatbot_response(ingredients_with_quantities, nutrition_data, totals, health_assessment)
    print(response)

if __name__ == "__main__":
    main()
