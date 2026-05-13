"""
EV Energy Consumption Training Script
- Preprocessing: Outlier removal via IQR and Log-Transformation (log1p) for target smoothing.
- Optimization: Hyperparameter tuning of Random Forest using GridSearchCV.
- Architecture: Hybrid Ensemble Model combining Random Forest and XGBoost (VotingRegressor).
- Insights: Extracts and visualizes feature importance to identify key consumption drivers.
- Artifacts: Exports trained ensemble, encoder, scaler, and column metadata for prediction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("🚀 Training Professional EV Energy Consumption Model..!")

# --- 1. SETUP PATHS & LOAD DATA ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# use dynamic path
dataset_path = os.path.join(script_dir, "Data", "EV_dataset.csv") 

if not os.path.exists(dataset_path):
    dataset_path = os.path.join(script_dir, "EV_dataset.csv")

try:
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded from: {dataset_path}")
except FileNotFoundError:
    print("❌ Error: EV_dataset.csv not found! Make sure it's in the same folder.")
    exit()

# Target and categorical columns
target = "Energy_Consumption_kWh"
categorical_cols = ['Driving_Mode', 'Road_Type', 'Traffic_Condition', 'Weather_Condition']

# Drop non-predictive columns (IDs) to avoid misleading the model
if 'Vehicle_ID' in df.columns:
    df = df.drop(columns=['Vehicle_ID'])
    print("Removed Vehicle_ID from training features")

# --- 2. DATA CLEANING & LOG TRANSFORM ---
# Outliers Removal
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[target] < (Q1 - 1.5 * IQR)) | (df[target] > (Q3 + 1.5 * IQR)))]

# Log Transform: to smooth the target (increase accuracy)
df[target] = np.log1p(df[target])

# --- 3. ENCODING & SCALING ---
# encoding categorical columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder.get_feature_names_out(categorical_cols)
)

# Scaling the numerical features (imp for XGBoost)
numerical_cols = df.drop(columns=categorical_cols + [target]).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# merge everything
X = pd.concat([df[numerical_cols].reset_index(drop=True), encoded_df], axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. HYPERPARAMETER TUNING (Optimizing RF) ---
print("\n🔍 Finding best parameters for Random Forest...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# --- 5. ENSEMBLE MODEL (RF + XGBoost) ---
print("🤖 Creating Ensemble Model (RF + XGBoost)...")
best_xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

# Combined both models
ensemble_model = VotingRegressor(estimators=[
    ('rf', best_rf),
    ('xgb', best_xgb)
])

ensemble_model.fit(X_train, y_train)

# --- 6. EVALUATION ---
# Predictions (Inverse log to get original kWh scale)
y_pred_log = ensemble_model.predict(X_test)
y_pred = np.expm1(y_pred_log) 
y_test_original = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

print(f"\n✅ Final Model Performance:")
print(f"RMSE: {rmse:.3f} kWh")
print(f"R² Score: {r2:.3f}")

# Cross-Validation to check stability
print("\n🔄 Performing 5-Fold Cross-Validation...")
# We use negative mean squared error because CV expects a 'score' (higher is better)
cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='r2')

print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

if cv_scores.std() * 2 < 0.05:
    print("✨ Model is highly stable across different data samples!")
else:
    print("⚠️ Model performance varies slightly depending on data split.")

# --- NEW: FEATURE IMPORTANCE FOR ENSEMBLE ---
# We take importance from the Random Forest component of our Ensemble
importances = best_rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Model Insight: Top 10 Factors ---")
print(feature_importance_df.head(10))

# --- PLOT GENERATION ---
show_plot = input("\nShow Feature Importance plot? (yes/no): ").lower()
if show_plot == 'yes':
    plt.figure(figsize=(10, 6))
    # Plotting top 10
    top_10 = feature_importance_df.head(10)
    plt.barh(top_10['Feature'], top_10['Importance'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('Top 10 Factors Influencing EV Energy Consumption')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "feature_importance_plot.png"))
    print("Plot saved as feature_importance_plot.png")
    plt.show()
    
# --- 7. SAVE EVERYTHING ---
joblib.dump(ensemble_model, os.path.join(script_dir, "ev_model.pkl"))
joblib.dump(encoder, os.path.join(script_dir, "encoder.pkl"))
joblib.dump(scaler, os.path.join(script_dir, "scaler.pkl"))
joblib.dump(X_train.columns.tolist(), os.path.join(script_dir, "model_columns.pkl"))

print("\n🎉 Training Complete! All components saved.")