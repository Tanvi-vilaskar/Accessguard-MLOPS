# import os
# import pandas as pd
# import joblib
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, classification_report
# from data_handler import load_logins
# from config import MODEL_FILE
# # Assuming config and data_handler are available

# # --- Utility Preprocessing Function ---

# def _preprocess_logins(logins: pd.DataFrame) -> pd.DataFrame:
#     """
#     Cleans the DataFrame and engineers the 'LoginHour' feature.
#     """
#     # 1. Drop rows with missing target (Outcome)
#     logins = logins.dropna(subset=["Outcome"])

#     # 2. Filter out rows missing core features
#     logins = logins.dropna(subset=["IP", "Device", "Browser", "MFA Enabled", "Username"])
    
#     # 3. Clean 'MFA Enabled' column 
#     mfa_numeric = pd.to_numeric(logins["MFA Enabled"], errors='coerce')
#     logins = logins[mfa_numeric.notna()].copy()
#     logins["MFA Enabled"] = mfa_numeric.dropna().astype(int)
    
#     # 4. Feature Engineering: Extract Login Hour
#     logins['Timestamp'] = pd.to_datetime(logins['Timestamp'])
#     logins['LoginHour'] = logins['Timestamp'].dt.hour
    
#     # 5. Feature Engineering: User-Specific New IP (CRITICAL NEW FEATURE)
    
#     # Group by username and collect all unique IPs seen so far
#     known_ips = {}
#     logins['User_New_IP'] = 0 # Default to 0 (Known IP)

#     # Sort data by timestamp to ensure history is processed chronologically
#     logins_sorted = logins.sort_values(by='Timestamp').reset_index(drop=True)

#     for index, row in logins_sorted.iterrows():
#         user = row['Username']
#         ip = row['IP']
        
#         if user not in known_ips:
#             # First time seeing this user
#             known_ips[user] = set()

#         if ip not in known_ips[user]:
#             # This is a new IP for this user
#             logins_sorted.loc[index, 'User_New_IP'] = 1
        
#         # Update the known IPs set AFTER evaluation (simulating real-time learning)
#         known_ips[user].add(ip)
        
#     return logins_sorted


# # --- Model Training and Saving ---

# def train_login_model():
#     """
#     Loads, preprocesses, trains a RandomForest model with class weighting, 
#     and saves the model and encoders.
#     """
#     logins = load_logins()
#     logins = _preprocess_logins(logins) # Now includes LoginHour AND User_New_IP

#     if len(logins) < 10:
#         print("Not enough data to train the model after cleaning.")
#         return None, None

#     # Define features, now including LoginHour and User_New_IP
#     features = ["IP", "Device", "Browser", "MFA Enabled", "LoginHour", "User_New_IP"]
#     # NOTE: We keep IP/Device/Browser as strings for LabelEncoder
#     X = logins[features].astype(str) 

#     # Encode categorical columns (IP, Device, Browser)
#     encoders = {}
#     for col in ["IP", "Device", "Browser"]:
#         enc = LabelEncoder()
#         X[col] = enc.fit_transform(X[col])
#         encoders[col] = enc

#     # Convert numeric features back to their appropriate type
#     X["MFA Enabled"] = logins["MFA Enabled"].astype(int)
#     X["LoginHour"] = logins["LoginHour"].astype(int) 
#     X["User_New_IP"] = logins["User_New_IP"].astype(int) # New integer feature
    
#     # Target variable setup (same as before)
#     y_str = logins["Outcome"].astype(str).fillna("Unknown")
#     outcome_encoder = LabelEncoder()
#     y = outcome_encoder.fit_transform(y_str)
#     encoders['Outcome'] = outcome_encoder

#     # Split data into training and testing sets (using the new X and y)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     # Train model with class_weight='balanced'
#     model = RandomForestClassifier(
#         n_estimators=50, 
#         random_state=42, 
#         class_weight='balanced'
#     )
#     model.fit(X_train, y_train)

#     # Evaluate model and display confusion matrix
#     display_confusion_matrix(model, X_test, y_test, outcome_encoder)

#     # Save model and encoders
#     joblib.dump((model, encoders), MODEL_FILE)
#     print(f"\nModel trained and saved to {MODEL_FILE}")

#     return model, encoders


# # --- Model Evaluation Function (No Change) ---
# def display_confusion_matrix(model, X_test, y_test, outcome_encoder):
#     # ... (Your existing display_confusion_matrix code)
#     print("\n--- Model Evaluation (Test Set) ---")
    
#     # Predict on the test set
#     y_pred = model.predict(X_test)
    
#     # Get the unique class labels in their original string form
#     labels = outcome_encoder.classes_
    
#     # Calculate the confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
    
#     # Print the confusion matrix
#     print("\nConfusion Matrix:")
#     # Print header for classes
#     header = [""] + [f"Pred {l}" for l in labels]
#     print(" | ".join(f"{h:<8}" for h in header))
#     print("-" * (9 * len(header)))
    
#     # Print rows
#     for i, true_label in enumerate(labels):
#         row = [f"True {true_label}"] + [f"{cm[i, j]:<8}" for j in range(len(labels))]
#         print(" | ".join(row))

#     # Print the classification report (includes Precision, Recall, F1-score)
#     print("\nClassification Report:")
#     report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
#     print(report)


# # --- Model Loading (No Change) ---
# def load_model():
#     if not os.path.exists(MODEL_FILE):
#         return None, None
#     return joblib.load(MODEL_FILE)

import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from data_handler import load_logins
from config import MODEL_FILE

# -------------------------------
# 1. Utility Preprocessing Function
# -------------------------------
def _preprocess_logins(logins: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame and engineers the 'LoginHour' and 'User_New_IP' features.
    """
    # Drop rows with missing target (Outcome)
    logins = logins.dropna(subset=["Outcome"])

    # Drop rows missing core features
    logins = logins.dropna(subset=["IP", "Device", "Browser", "MFA Enabled", "Username"])

    # Clean 'MFA Enabled' column
    mfa_numeric = pd.to_numeric(logins["MFA Enabled"], errors='coerce')
    logins = logins[mfa_numeric.notna()].copy()
    logins["MFA Enabled"] = mfa_numeric.dropna().astype(int)

    # Feature Engineering: Extract Login Hour
    logins['Timestamp'] = pd.to_datetime(logins['Timestamp'])
    logins['LoginHour'] = logins['Timestamp'].dt.hour

    # Feature Engineering: User-Specific New IP
    known_ips = {}
    logins['User_New_IP'] = 0
    logins_sorted = logins.sort_values(by='Timestamp').reset_index(drop=True)

    for index, row in logins_sorted.iterrows():
        user = row['Username']
        ip = row['IP']

        if user not in known_ips:
            known_ips[user] = set()

        if ip not in known_ips[user]:
            logins_sorted.loc[index, 'User_New_IP'] = 1

        known_ips[user].add(ip)

    return logins_sorted


# -------------------------------
# 2. Model Training and Saving
# -------------------------------
def train_login_model():
    """
    Loads, preprocesses, trains a RandomForest model with class weighting,
    and saves the model and encoders.
    """
    logins = load_logins()
    logins = _preprocess_logins(logins)

    if len(logins) < 10:
        print("Not enough data to train the model after cleaning.")
        return None, None

    # Define features
    features = ["IP", "Device", "Browser", "MFA Enabled", "LoginHour", "User_New_IP"]
    X = logins[features].astype(str)

    # Encode categorical columns
    encoders = {}
    for col in ["IP", "Device", "Browser"]:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col])
        encoders[col] = enc

    # Convert numeric features
    X["MFA Enabled"] = logins["MFA Enabled"].astype(int)
    X["LoginHour"] = logins["LoginHour"].astype(int)
    X["User_New_IP"] = logins["User_New_IP"].astype(int)

    # Target variable
    y_str = logins["Outcome"].astype(str).fillna("Unknown")
    outcome_encoder = LabelEncoder()
    y = outcome_encoder.fit_transform(y_str)
    encoders['Outcome'] = outcome_encoder

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train RandomForest with balanced class weight
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate model (text + graph)
    display_confusion_matrix(model, X_test, y_test, outcome_encoder)
    plot_confusion_matrix(model, X_test, y_test, outcome_encoder)

    # Save model and encoders
    joblib.dump((model, encoders), MODEL_FILE)
    print(f"\nModel trained and saved to {MODEL_FILE}")

    return model, encoders


# -------------------------------
# 3. Text-based Confusion Matrix
# -------------------------------
def display_confusion_matrix(model, X_test, y_test, outcome_encoder):
    print("\n--- Model Evaluation (Test Set) ---")

    y_pred = model.predict(X_test)
    labels = outcome_encoder.classes_
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    header = [""] + [f"Pred {l}" for l in labels]
    print(" | ".join(f"{h:<8}" for h in header))
    print("-" * (9 * len(header)))

    for i, true_label in enumerate(labels):
        row = [f"True {true_label}"] + [f"{cm[i, j]:<8}" for j in range(len(labels))]
        print(" | ".join(row))

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    print(report)


# -------------------------------
# 4. Graphical Confusion Matrix
# -------------------------------
def plot_confusion_matrix(model, X_test, y_test, outcome_encoder):
    """
    Plots a heatmap confusion matrix for the test predictions.
    """
    y_pred = model.predict(X_test)
    labels = outcome_encoder.classes_
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='GnBu', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.show()


# -------------------------------
# 5. Model Loading
# -------------------------------
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None, None
    return joblib.load(MODEL_FILE)


# -------------------------------
# 6. Run Training (Optional)
# -------------------------------
if __name__ == "__main__":
    train_login_model()
