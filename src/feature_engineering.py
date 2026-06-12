import pandas as pd


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Average of the four IELTS skill scores
    df["Avg_Skill_Score"] = df[
        ["Reading_Score", "Writing_Score", "Listening_Score", "Speaking_Score"]
    ].mean(axis=1)
    # Combined language ability proxy
    df["Language_Ability"] = (df["Vocabulary_Score"] + df["Grammar_Score"]) / 2
    # Engagement index: how much the student puts in
    df["Engagement_Index"] = (df["Attendance_Percentage"] / 100) * df["Practice_Hours"]
    return df


def get_feature_names(df: pd.DataFrame, target_col: str = "Final_IELTS_Band") -> list:
    return [c for c in df.columns if c != target_col]
