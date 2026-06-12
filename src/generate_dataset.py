"""
Generates a realistic synthetic IELTS dataset with 5000 rows.
Run: python src/generate_dataset.py
Output: data/raw/dataset.csv
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_STUDENTS = 5000

np.random.seed(RANDOM_SEED)

# --- Base latent ability (hidden variable driving most scores) ---
latent_ability = np.random.normal(loc=0.5, scale=0.15, size=N_STUDENTS)
latent_ability = np.clip(latent_ability, 0, 1)

def noisy(base, scale=0.08, low=0, high=1):
    return np.clip(base + np.random.normal(0, scale, N_STUDENTS), low, high)

# --- Four IELTS skill scores (1–9 band scale) ---
reading   = np.round(noisy(latent_ability, 0.10) * 8 + 1, 1)
writing   = np.round(noisy(latent_ability, 0.12) * 8 + 1, 1)
listening = np.round(noisy(latent_ability, 0.09) * 8 + 1, 1)
speaking  = np.round(noisy(latent_ability, 0.11) * 8 + 1, 1)

# Clip to valid IELTS band range
for arr in [reading, writing, listening, speaking]:
    np.clip(arr, 1.0, 9.0, out=arr)

# --- Behavioural features ---
attendance   = np.round(noisy(latent_ability, 0.12, 40, 100) * 100, 1)
attendance   = np.clip(attendance, 40.0, 100.0)

practice_hrs = np.round(noisy(latent_ability, 0.15, 0, 1) * 10, 1)
practice_hrs = np.clip(practice_hrs, 0.5, 10.0)

vocabulary   = np.round(noisy(latent_ability, 0.10) * 100, 1)
vocabulary   = np.clip(vocabulary, 20.0, 100.0)

grammar      = np.round(noisy(latent_ability, 0.10) * 100, 1)
grammar      = np.clip(grammar, 20.0, 100.0)

mock_score   = np.round(noisy(latent_ability, 0.09) * 8 + 1, 1)
mock_score   = np.clip(mock_score, 1.0, 9.0)

# --- Target: Final IELTS Band (weighted combination + small noise) ---
final_band = (
    0.25 * reading
    + 0.20 * writing
    + 0.25 * listening
    + 0.15 * speaking
    + 0.10 * mock_score
    + 0.05 * (vocabulary / 100 * 8 + 1)
)
final_band = np.round(final_band + np.random.normal(0, 0.15, N_STUDENTS), 1)
final_band = np.clip(final_band, 1.0, 9.0)

# --- Assemble DataFrame ---
df = pd.DataFrame({
    "Reading_Score":        reading,
    "Writing_Score":        writing,
    "Listening_Score":      listening,
    "Speaking_Score":       speaking,
    "Attendance_Percentage": attendance,
    "Practice_Hours":       practice_hrs,
    "Vocabulary_Score":     vocabulary,
    "Grammar_Score":        grammar,
    "Mock_Test_Score":      mock_score,
    "Final_IELTS_Band":     final_band,
})

output_path = "data/raw/dataset.csv"
df.to_csv(output_path, index=False)
print(f"Dataset saved to {output_path}")
print(df.describe().round(2))
