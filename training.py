import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

# Load dataset
try:
    data = pd.read_csv('data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'data.csv' file not found.")
    exit()

# Data Cleaning: Check columns
required_columns = ['director_name', 'duration', 'actor_1_name', 'budget', 'genres', 'title_year', 'gross']
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Error: Missing columns in data.csv - {missing_columns}")
    exit()

# Remove rows with missing values in essential columns
data = data.dropna(subset=required_columns)
print(f"Data shape after removing rows with missing values: {data.shape}")

# Define target and features
X = data[['director_name', 'duration', 'actor_1_name', 'budget', 'genres', 'title_year']]
y = data['gross']

# Preprocessing pipeline
numerical_features = ['duration', 'budget', 'title_year']
categorical_features = ['director_name', 'actor_1_name', 'genres']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models
models = {
    'RandomForest': RandomForestRegressor(),
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'KNeighbors': KNeighborsRegressor()
}

# Training and selecting the best model
best_model = None
best_score = -np.inf

for model_name, model in models.items():
    try:
        # Create pipeline with preprocessing and model
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        scores = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_absolute_error')
        avg_score = np.mean(scores)
        print(f'{model_name} Score: {avg_score}')
        if avg_score > best_score:
            best_score = avg_score
            best_model = pipe
    except Exception as e:
        print(f"Error with model {model_name}: {e}")

# Train the best model on the full dataset
if best_model:
    best_model.fit(X, y)
    with open('model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print("Training complete. Best model saved as model.pkl.")
else:
    print("Error: No model was trained successfully.")
