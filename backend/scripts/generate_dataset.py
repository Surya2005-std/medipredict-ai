"""
Dataset generator for Disease Symptom Prediction.
Generates a comprehensive, realistic symptom-disease dataset with 41 diseases
and 132 symptoms (based on the Kaggle Disease-Symptom dataset structure).
"""

import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────
# SYMPTOM LIST (132 symptoms)
# ──────────────────────────────────────────────
ALL_SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
    'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
    'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_urination', 'fatigue', 'weight_gain', 'anxiety',
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness',
    'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation',
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
    'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
    'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
    'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
    'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
    'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
    'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
    'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
    'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
    'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes',
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
    'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
    'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

# ──────────────────────────────────────────────
# DISEASE → SYMPTOM MAPPING
# ──────────────────────────────────────────────
DISEASE_SYMPTOMS = {
    'Fungal infection': [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches'
    ],
    'Allergy': [
        'continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'
    ],
    'GERD': [
        'stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'cough', 'chest_pain'
    ],
    'Chronic cholestasis': [
        'itching', 'vomiting', 'yellowish_skin', 'dark_urine', 'nausea',
        'loss_of_appetite', 'abdominal_pain'
    ],
    'Drug Reaction': [
        'itching', 'skin_rash', 'stomach_pain', 'burning_micturition', 'spotting_urination'
    ],
    'Peptic ulcer disease': [
        'vomiting', 'indigestion', 'loss_of_appetite', 'abdominal_pain', 'passage_of_gases',
        'internal_itching'
    ],
    'AIDS': [
        'muscle_wasting', 'patches_in_throat', 'high_fever', 'extra_marital_contacts',
        'fatigue', 'weight_loss', 'diarrhoea'
    ],
    'Diabetes': [
        'fatigue', 'weight_loss', 'restlessness', 'lethargy', 'irregular_sugar_level',
        'blurred_and_distorted_vision', 'obesity', 'excessive_hunger', 'increased_appetite',
        'polyuria'
    ],
    'Gastroenteritis': [
        'vomiting', 'sunken_eyes', 'dehydration', 'diarrhoea'
    ],
    'Bronchial Asthma': [
        'fatigue', 'cough', 'high_fever', 'breathlessness', 'family_history', 'mucoid_sputum'
    ],
    'Hypertension': [
        'headache', 'chest_pain', 'dizziness', 'loss_of_balance', 'lack_of_concentration'
    ],
    'Migraine': [
        'acidity', 'indigestion', 'headache', 'blurred_and_distorted_vision', 'excessive_hunger',
        'stiff_neck', 'depression', 'irritability', 'visual_disturbances'
    ],
    'Cervical spondylosis': [
        'back_pain', 'weakness_in_limbs', 'neck_pain', 'dizziness', 'loss_of_balance'
    ],
    'Paralysis (brain hemorrhage)': [
        'vomiting', 'headache', 'weakness_of_one_body_side', 'altered_sensorium'
    ],
    'Jaundice': [
        'itching', 'vomiting', 'fatigue', 'weight_loss', 'high_fever', 'yellowish_skin',
        'dark_urine', 'abdominal_pain'
    ],
    'Malaria': [
        'chills', 'vomiting', 'high_fever', 'sweating', 'headache', 'nausea',
        'diarrhoea', 'muscle_pain'
    ],
    'Chicken pox': [
        'itching', 'skin_rash', 'fatigue', 'lethargy', 'high_fever', 'headache',
        'loss_of_appetite', 'mild_fever', 'swelled_lymph_nodes', 'malaise',
        'red_spots_over_body'
    ],
    'Dengue': [
        'skin_rash', 'chills', 'joint_pain', 'vomiting', 'fatigue', 'high_fever',
        'headache', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
        'back_pain', 'malaise', 'muscle_pain', 'red_spots_over_body'
    ],
    'Typhoid': [
        'chills', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea',
        'constipation', 'abdominal_pain', 'diarrhoea', 'toxic_look_(typhos)', 'belly_pain'
    ],
    'Hepatitis A': [
        'joint_pain', 'vomiting', 'yellowish_skin', 'dark_urine', 'nausea',
        'loss_of_appetite', 'abdominal_pain', 'diarrhoea', 'mild_fever',
        'yellowing_of_eyes', 'muscle_pain'
    ],
    'Hepatitis B': [
        'itching', 'fatigue', 'lethargy', 'yellowish_skin', 'dark_urine',
        'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes',
        'receiving_blood_transfusion', 'receiving_unsterile_injections', 'malaise',
        'fluid_overload.1'
    ],
    'Hepatitis C': [
        'fatigue', 'yellowish_skin', 'nausea', 'loss_of_appetite', 'yellowing_of_eyes',
        'family_history'
    ],
    'Hepatitis D': [
        'joint_pain', 'vomiting', 'fatigue', 'yellowish_skin', 'dark_urine', 'nausea',
        'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes'
    ],
    'Hepatitis E': [
        'joint_pain', 'vomiting', 'fatigue', 'high_fever', 'yellowish_skin', 'dark_urine',
        'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes', 'coma',
        'stomach_bleeding'
    ],
    'Alcoholic hepatitis': [
        'vomiting', 'yellowish_skin', 'abdominal_pain', 'swelling_of_stomach',
        'history_of_alcohol_consumption', 'fluid_overload', 'distention_of_abdomen'
    ],
    'Tuberculosis': [
        'chills', 'vomiting', 'fatigue', 'weight_loss', 'cough', 'high_fever', 'breathlessness',
        'sweating', 'loss_of_appetite', 'mild_fever', 'swelled_lymph_nodes', 'malaise',
        'phlegm', 'blood_in_sputum', 'chest_pain'
    ],
    'Common Cold': [
        'continuous_sneezing', 'chills', 'fatigue', 'cough', 'high_fever', 'headache',
        'swelled_lymph_nodes', 'malaise', 'phlegm', 'throat_irritation', 'redness_of_eyes',
        'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'loss_of_smell',
        'muscle_pain'
    ],
    'Pneumonia': [
        'chills', 'fatigue', 'cough', 'high_fever', 'breathlessness', 'sweating',
        'malaise', 'phlegm', 'chest_pain', 'fast_heart_rate', 'rusty_sputum'
    ],
    'Dimorphic hemorrhoids (piles)': [
        'constipation', 'pain_during_bowel_movements', 'pain_in_anal_region',
        'bloody_stool', 'irritation_in_anus'
    ],
    'Heart attack': [
        'vomiting', 'breathlessness', 'sweating', 'chest_pain'
    ],
    'Varicose veins': [
        'fatigue', 'cramps', 'bruising', 'obesity', 'swollen_legs',
        'swollen_blood_vessels', 'prominent_veins_on_calf'
    ],
    'Hypothyroidism': [
        'fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings', 'lethargy',
        'dizziness', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
        'swollen_extremeties', 'depression', 'irritability', 'abnormal_menstruation'
    ],
    'Hyperthyroidism': [
        'fatigue', 'mood_swings', 'weight_loss', 'restlessness', 'sweating',
        'diarrhoea', 'fast_heart_rate', 'enlarged_thyroid', 'excessive_hunger',
        'muscle_weakness', 'irritability', 'abnormal_menstruation'
    ],
    'Hypoglycemia': [
        'fatigue', 'anxiety', 'cold_hands_and_feets', 'sweating', 'headache',
        'nausea', 'blurred_and_distorted_vision', 'excessive_hunger', 'drying_and_tingling_lips',
        'slurred_speech', 'irritability', 'muscle_weakness', 'palpitations'
    ],
    'Osteoarthritis': [
        'joint_pain', 'neck_pain', 'knee_pain', 'hip_joint_pain', 'swelling_joints',
        'painful_walking'
    ],
    'Arthritis': [
        'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
        'painful_walking'
    ],
    '(Vertigo) Paroxysmal Positional Vertigo': [
        'vomiting', 'headache', 'nausea', 'spinning_movements', 'loss_of_balance',
        'unsteadiness'
    ],
    'Acne': [
        'skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'
    ],
    'Urinary tract infection': [
        'burning_micturition', 'bladder_discomfort', 'foul_smell_of_urine',
        'continuous_feel_of_urine'
    ],
    'Psoriasis': [
        'skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting',
        'small_dents_in_nails', 'inflammatory_nails'
    ],
    'Impetigo': [
        'skin_rash', 'high_fever', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    ],
}

# ──────────────────────────────────────────────
# PRECAUTIONS PER DISEASE
# ──────────────────────────────────────────────
DISEASE_PRECAUTIONS = {
    'Fungal infection': [
        'Keep skin clean and dry', 'Use antifungal powders/creams',
        'Avoid sharing personal items', 'Wear breathable clothing'
    ],
    'Allergy': [
        'Avoid known allergens', 'Use antihistamines', 'Keep windows closed during pollen season',
        'Wear a mask in dusty environments'
    ],
    'GERD': [
        'Avoid fatty/spicy foods', 'Eat smaller meals', 'Don\'t lie down after eating',
        'Raise head of bed'
    ],
    'Chronic cholestasis': [
        'Avoid alcohol', 'Consult a doctor', 'Follow a low-fat diet',
        'Take vitamin supplements'
    ],
    'Drug Reaction': [
        'Stop the suspected drug immediately', 'Consult your doctor',
        'Inform of all medications taken', 'Avoid self-medication'
    ],
    'Peptic ulcer disease': [
        'Avoid spicy food', 'Don\'t take NSAIDs', 'Quit smoking', 'Reduce stress'
    ],
    'AIDS': [
        'Practice safe sex', 'Avoid sharing needles', 'Take antiretroviral therapy',
        'Regular medical checkups'
    ],
    'Diabetes': [
        'Monitor blood sugar regularly', 'Follow a balanced diet', 'Exercise regularly',
        'Take prescribed medications'
    ],
    'Gastroenteritis': [
        'Stay hydrated', 'Avoid solid foods temporarily', 'Wash hands frequently',
        'Use ORS sachets'
    ],
    'Bronchial Asthma': [
        'Use prescribed inhaler', 'Avoid triggers like smoke/dust', 'Monitor peak flow',
        'Keep rescue inhaler accessible'
    ],
    'Hypertension': [
        'Reduce salt intake', 'Exercise regularly', 'Monitor blood pressure daily',
        'Avoid stress'
    ],
    'Migraine': [
        'Avoid trigger foods', 'Sleep in a dark quiet room', 'Stay hydrated',
        'Use prescribed medication'
    ],
    'Cervical spondylosis': [
        'Use a firm pillow', 'Do neck exercises', 'Avoid sudden neck movements',
        'Maintain good posture'
    ],
    'Paralysis (brain hemorrhage)': [
        'Seek emergency medical care immediately', 'Maintain blood pressure',
        'Physiotherapy', 'Avoid stress'
    ],
    'Jaundice': [
        'Drink plenty of water', 'Avoid alcohol', 'Get plenty of rest',
        'Eat high-protein food'
    ],
    'Malaria': [
        'Use mosquito repellent', 'Sleep under nets', 'Take prescribed antimalarials',
        'Eliminate standing water'
    ],
    'Chicken pox': [
        'Stay isolated', 'Use calamine lotion', 'Cut fingernails short',
        'Avoid scratching'
    ],
    'Dengue': [
        'Use mosquito repellent', 'Wear full-sleeve clothes', 'Stay hydrated',
        'Seek early medical care'
    ],
    'Typhoid': [
        'Eat freshly cooked food', 'Drink purified water', 'Wash hands regularly',
        'Get vaccinated'
    ],
    'Hepatitis A': [
        'Maintain hygiene', 'Get vaccinated', 'Avoid contaminated water',
        'Wash hands after toilet use'
    ],
    'Hepatitis B': [
        'Get vaccinated', 'Use protection during sex', 'Avoid sharing needles',
        'Avoid sharing razors'
    ],
    'Hepatitis C': [
        'Avoid sharing needles', 'Use protective equipment', 'Get regular checkups',
        'Consult hepatologist'
    ],
    'Hepatitis D': [
        'Get vaccinated for Hepatitis B', 'Avoid sharing needles',
        'Consult a doctor immediately', 'Avoid alcohol'
    ],
    'Hepatitis E': [
        'Drink purified water', 'Avoid raw/undercooked food', 'Maintain good hygiene',
        'Consult doctor if pregnant'
    ],
    'Alcoholic hepatitis': [
        'Stop alcohol consumption', 'Consult a hepatologist', 'Follow prescribed diet',
        'Take supplements'
    ],
    'Tuberculosis': [
        'Complete the full TB treatment', 'Cover mouth when coughing', 'Improve ventilation',
        'Avoid close contact with others'
    ],
    'Common Cold': [
        'Rest adequately', 'Stay hydrated', 'Use decongestants', 'Avoid cold environments'
    ],
    'Pneumonia': [
        'Get vaccinated', 'Seek immediate medical care', 'Complete antibiotics course',
        'Rest and stay warm'
    ],
    'Dimorphic hemorrhoids (piles)': [
        'Increase fiber intake', 'Drink lots of water', 'Avoid straining',
        'Use sitz baths'
    ],
    'Heart attack': [
        'Call emergency services immediately', 'Chew aspirin if not allergic',
        'Lie down calmly', 'Avoid strenuous activity'
    ],
    'Varicose veins': [
        'Elevate legs when resting', 'Wear compression stockings', 'Exercise regularly',
        'Avoid prolonged standing'
    ],
    'Hypothyroidism': [
        'Take levothyroxine as prescribed', 'Regular thyroid function tests',
        'Exercise regularly', 'Eat iodine-rich foods'
    ],
    'Hyperthyroidism': [
        'Take anti-thyroid medication', 'Regular thyroid tests', 'Avoid iodine-rich foods',
        'Stress management'
    ],
    'Hypoglycemia': [
        'Consume fast-acting carbohydrates', 'Eat small frequent meals',
        'Monitor blood sugar', 'Carry glucose tablets'
    ],
    'Osteoarthritis': [
        'Use heat/cold therapy', 'Maintain healthy weight', 'Do low-impact exercises',
        'Use assistive devices'
    ],
    'Arthritis': [
        'Follow anti-inflammatory diet', 'Physiotherapy', 'Use prescribed medications',
        'Protect joints during activities'
    ],
    '(Vertigo) Paroxysmal Positional Vertigo': [
        'Do Epley maneuver', 'Avoid sudden head movements', 'Sit up slowly',
        'Consult ENT specialist'
    ],
    'Acne': [
        'Keep skin clean', 'Avoid touching face', 'Use non-comedogenic products',
        'Consult a dermatologist'
    ],
    'Urinary tract infection': [
        'Drink plenty of water', 'Urinate after intercourse', 'Avoid holding urine',
        'Take prescribed antibiotics'
    ],
    'Psoriasis': [
        'Moisturize regularly', 'Use prescribed topical treatments',
        'Avoid triggers like stress', 'Get light therapy'
    ],
    'Impetigo': [
        'Keep sores covered', 'Wash hands frequently', 'Avoid touching sores',
        'Take prescribed antibiotics'
    ],
}

# ──────────────────────────────────────────────
# SYMPTOM SEVERITY WEIGHTS (1–7)
# ──────────────────────────────────────────────
SYMPTOM_SEVERITY = {
    'itching': 1, 'skin_rash': 3, 'nodal_skin_eruptions': 4, 'continuous_sneezing': 4,
    'shivering': 5, 'chills': 3, 'joint_pain': 3, 'stomach_pain': 5, 'acidity': 3,
    'ulcers_on_tongue': 4, 'muscle_wasting': 3, 'vomiting': 5, 'burning_micturition': 6,
    'spotting_urination': 6, 'fatigue': 4, 'weight_gain': 3, 'anxiety': 4,
    'cold_hands_and_feets': 5, 'mood_swings': 3, 'weight_loss': 3, 'restlessness': 5,
    'lethargy': 2, 'patches_in_throat': 6, 'irregular_sugar_level': 5, 'cough': 4,
    'high_fever': 7, 'sunken_eyes': 3, 'breathlessness': 4, 'sweating': 3,
    'dehydration': 4, 'indigestion': 5, 'headache': 3, 'yellowish_skin': 3,
    'dark_urine': 4, 'nausea': 4, 'loss_of_appetite': 4, 'pain_behind_the_eyes': 4,
    'back_pain': 3, 'constipation': 4, 'abdominal_pain': 5, 'diarrhoea': 6,
    'mild_fever': 1, 'yellow_urine': 1, 'yellowing_of_eyes': 4, 'acute_liver_failure': 6,
    'fluid_overload': 4, 'swelling_of_stomach': 7, 'swelled_lymph_nodes': 6,
    'malaise': 1, 'blurred_and_distorted_vision': 5, 'phlegm': 5, 'throat_irritation': 4,
    'redness_of_eyes': 4, 'sinus_pressure': 4, 'runny_nose': 5, 'congestion': 4,
    'chest_pain': 6, 'weakness_in_limbs': 6, 'fast_heart_rate': 3,
    'pain_during_bowel_movements': 5, 'pain_in_anal_region': 6, 'bloody_stool': 5,
    'irritation_in_anus': 6, 'neck_pain': 5, 'dizziness': 4, 'cramps': 4,
    'bruising': 4, 'obesity': 4, 'swollen_legs': 5, 'swollen_blood_vessels': 5,
    'puffy_face_and_eyes': 5, 'enlarged_thyroid': 6, 'brittle_nails': 5,
    'swollen_extremeties': 5, 'excessive_hunger': 4, 'extra_marital_contacts': 5,
    'drying_and_tingling_lips': 4, 'slurred_speech': 4, 'knee_pain': 3,
    'hip_joint_pain': 5, 'muscle_weakness': 3, 'stiff_neck': 4, 'swelling_joints': 5,
    'movement_stiffness': 5, 'spinning_movements': 6, 'loss_of_balance': 4,
    'unsteadiness': 5, 'weakness_of_one_body_side': 6, 'loss_of_smell': 3,
    'bladder_discomfort': 4, 'foul_smell_of_urine': 5, 'continuous_feel_of_urine': 6,
    'passage_of_gases': 5, 'internal_itching': 4, 'toxic_look_(typhos)': 5,
    'depression': 3, 'irritability': 2, 'muscle_pain': 2, 'altered_sensorium': 5,
    'red_spots_over_body': 4, 'belly_pain': 4, 'abnormal_menstruation': 6,
    'dischromic_patches': 6, 'watering_from_eyes': 4, 'increased_appetite': 5,
    'polyuria': 4, 'family_history': 5, 'mucoid_sputum': 4, 'rusty_sputum': 4,
    'lack_of_concentration': 3, 'visual_disturbances': 4,
    'receiving_blood_transfusion': 5, 'receiving_unsterile_injections': 6,
    'coma': 7, 'stomach_bleeding': 6, 'distention_of_abdomen': 4,
    'history_of_alcohol_consumption': 5, 'fluid_overload.1': 4,
    'blood_in_sputum': 5, 'prominent_veins_on_calf': 6, 'palpitations': 4,
    'painful_walking': 5, 'pus_filled_pimples': 4, 'blackheads': 4,
    'scurring': 4, 'skin_peeling': 3, 'silver_like_dusting': 4,
    'small_dents_in_nails': 5, 'inflammatory_nails': 5, 'blister': 4,
    'red_sore_around_nose': 5, 'yellow_crust_ooze': 4,
}


def generate_dataset(samples_per_disease=180, noise_level=0.05):
    """
    Generate a synthetic but realistic disease-symptom dataset.
    Each disease gets `samples_per_disease` training samples.
    `noise_level` controls the proportion of randomly flipped symptom bits.
    """
    np.random.seed(42)
    rows = []

    for disease, core_symptoms in DISEASE_SYMPTOMS.items():
        for _ in range(samples_per_disease):
            row = {s: 0 for s in ALL_SYMPTOMS}
            # Add core symptoms (always present)
            for sym in core_symptoms:
                if sym in row:
                    row[sym] = 1
            # Random noise: flip some bits
            for sym in ALL_SYMPTOMS:
                if np.random.random() < noise_level:
                    row[sym] = 1 - row[sym]
            row['Disease'] = disease
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_datasets(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Main dataset
    df = generate_dataset(samples_per_disease=120)
    df.to_csv(os.path.join(output_dir, 'disease_symptoms.csv'), index=False)
    print(f"✅ Main dataset saved: {len(df)} rows × {len(df.columns)} cols")

    # Severity mapping
    sev_df = pd.DataFrame(
        list(SYMPTOM_SEVERITY.items()), columns=['Symptom', 'Severity']
    )
    sev_df.to_csv(os.path.join(output_dir, 'symptom_severity.csv'), index=False)
    print(f"✅ Severity mapping saved: {len(sev_df)} symptoms")

    # Precautions
    prec_rows = []
    for disease, precs in DISEASE_PRECAUTIONS.items():
        prec_rows.append({
            'Disease': disease,
            'Precaution_1': precs[0] if len(precs) > 0 else '',
            'Precaution_2': precs[1] if len(precs) > 1 else '',
            'Precaution_3': precs[2] if len(precs) > 2 else '',
            'Precaution_4': precs[3] if len(precs) > 3 else '',
        })
    prec_df = pd.DataFrame(prec_rows)
    prec_df.to_csv(os.path.join(output_dir, 'disease_precautions.csv'), index=False)
    print(f"✅ Precautions saved: {len(prec_df)} diseases")

    return df


if __name__ == '__main__':
    save_datasets('../data')
