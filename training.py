import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random
import joblib

# Load dataset
df = pd.read_csv('sign_language_data.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Label encode letters A-Z into 0-25
le = LabelEncoder()
y = le.fit_transform(y)

# Augmentation function: adds mirrored, rotated, zoomed samples
def augment_data(X, y, multiplier=1):
    augmented_X, augmented_y = [], []
    for i in range(len(X)):
        sample = X[i].reshape(-1, 2)
        for _ in range(multiplier):
            # Mirror (flip x)
            if random.random() < 0.3:
                sample[:, 0] = 1 - sample[:, 0]
            # Zoom
            if random.random() < 0.3:
                factor = random.uniform(0.9, 1.1)
                sample = sample * factor
            # Rotation
            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                theta = np.radians(angle)
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                            [np.sin(theta),  np.cos(theta)]])
                center = np.mean(sample, axis=0)
                sample = (sample - center) @ rotation_matrix + center
            # Add back
            augmented_X.append(sample.flatten())
            augmented_y.append(y[i])
    return np.array(augmented_X), np.array(augmented_y)

# Apply augmentation
aug_X, aug_y = augment_data(X, y, multiplier=2)
X_all = np.vstack((X, aug_X))
y_all = np.concatenate((y, aug_y))

# Split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)

joblib.dump((model, le), "sign_language_model.pkl")

# Predict
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
