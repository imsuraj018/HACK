import pandas as pd
import numpy as np
import joblib
import json

# Load trained model
model = joblib.load("crop_recommendation.pkl")

# Load input parameters from JSON file
with open("parameters.json", "r", encoding="utf-8") as file:
    manual_data = json.load(file)

# Convert to DataFrame
input_df = pd.DataFrame([manual_data])

# Predict probabilities
probs = model.predict_proba(input_df)[0]
crop_classes = model.classes_

# Create a DataFrame of crop recommendations
crop_scores = pd.DataFrame({
    'Crop': crop_classes,
    'Probability': probs
}).sort_values(by='Probability', ascending=False)

# Load crop info JSON
with open("crop_info.json", "r", encoding="utf-8") as f:
    crop_info = json.load(f)

# Load fertilizer schedule JSON
with open("fertilizer_schedule.json", "r", encoding="utf-8") as f:
    fertilizer_info = json.load(f)

# Display top 5 recommendations with details and fertilizer schedule
top_crops = crop_scores.head(5)
print("Top 5 Crop Recommendations:\n")
print(top_crops)

for index, row in top_crops.iterrows():
    crop_name = row['Crop']
    probability = row['Probability']
    print(f"Crop: {crop_name}\n")
    
    # Crop info
    if crop_name in crop_info:
        info = crop_info[crop_name]
        print("Reason:")
        for lang, text in info['reason'].items():
            print(f"  {lang}: {text}")
        print("Season:")
        for lang, text in info['season'].items():
            print(f"  {lang}: {text}")
        print("Notes:")
        for lang, text in info['notes'].items():
            print(f"  {lang}: {text}")
    
    # Fertilizer info
    if crop_name in fertilizer_info:
        fert = fertilizer_info[crop_name]
        print("\nFertilizer Schedule:")
        print(f"  Nutrients per hectare: {fert['nutrients_kg_per_hectare']}")
        print("  Application Stages:")
        for lang, stages in fert['application_stages'].items():
            print(f"    {lang}:")
            if isinstance(stages, str):  # English guidance as string
                print(f"      {stages}")
            else:
                for stage, details in stages.items():
                    print(f"      Stage: {stage}")
                    if 'विवरण' in details or 'वर्णन' in details:  # Hindi/Marathi
                        description = details.get('विवरण', details.get('वर्णन', ''))
                        print(f"        Description: {description}")
                    if 'खाद' in details or 'खते' in details:
                        fertilizers = details.get('खाद', details.get('खते', []))
                        print(f"        Fertilizers: {fertilizers}")
                    if 'स्प्रे' in details or 'स्प्रे' in details:
                        sprays = details.get('स्प्रे', [])
                        print(f"        Sprays: {sprays}")
    print("\n" + "-"*70 + "\n")
