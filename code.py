# Import All Necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set plot style for better visualizations
sns.set_style('whitegrid')

# -------------------------------------------------------------------
# 1. DATA LOADING AND INITIAL EXPLORATION
# -------------------------------------------------------------------

# Load the dataset
try:
    # Use the filename confirmed in previous steps
    df = pd.read_csv('netflix_titles.csv')
except FileNotFoundError:
    print("Error: 'netflix_titles.csv' not found. Make sure the file is in the same directory.")
    exit()

print("--- Initial Data Overview ---")
print("Shape of the dataset (rows, columns):", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
df.info()

# -------------------------------------------------------------------
# 2. DATA WRANGLING AND FEATURE ENGINEERING
# -------------------------------------------------------------------

print("\n--- Data Wrangling ---")

# --- 2.1 Handling Missing Values ---
# Fill highly missing categorical columns with 'Unknown'
for col in ['director', 'cast', 'country']:
    df[col].fillna('Unknown', inplace=True)
    
# Fill low missing count columns with the mode
df['date_added'].fillna(df['date_added'].mode()[0], inplace=True)
df['rating'].fillna(df['rating'].mode()[0], inplace=True)
df['duration'].fillna(df['duration'].mode()[0], inplace=True)

print("Handled missing values by filling with 'Unknown' or mode.")


# --- 2.2 Feature Engineering: Duration, Time, and Origin ---

# 1. Extract 'main_country' (first country listed)
df['main_country'] = df['country'].apply(lambda x: x.split(',')[0].strip())
print("Created 'main_country' feature.")

# 2. Extract 'duration_int' (the numeric part) and 'duration_type'
df['duration_int'] = df['duration'].str.extract('(\d+)').astype(int)
df['duration_type'] = df['duration'].apply(lambda x: 'min' if 'min' in x else 'Season')
print("Created 'duration_int' and 'duration_type' features.")

# 3. Extract 'year_added'
df['year_added'] = pd.to_datetime(df['date_added']).dt.year
print("Created 'year_added' feature.")

# 4. Feature for the model: Drop columns not needed for prediction
df_model = df.drop([
    'show_id', 'title', 'director', 'cast', 'description', 
    'date_added', 'country', 'duration', 'release_year'
], axis=1)

# Drop any rows that might have become NaN after feature engineering (e.g., in edge cases)
df_model.dropna(inplace=True) 

# --- 2.3 Preprocessing for Modeling ---
# Encode the target variable 'type' (Movie: 0, TV Show: 1)
le = LabelEncoder()
df_model['type'] = le.fit_transform(df_model['type'])
print("Target variable 'type' encoded (Movie=0, TV Show=1).")


# -------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------------

print("\n--- Generating EDA Visualizations ---")

# Chart 1: Distribution of Content Type (Univariate)
plt.figure(figsize=(8, 8))
df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#E50914', '#221e1f'])
plt.title('Distribution of Content Type (Movies vs. TV Shows)', fontsize=16)
plt.ylabel('')
plt.show()

# Chart 2: Content Acquisition Trend by Year (Univariate)
plt.figure(figsize=(12, 6))
df.groupby('year_added')['show_id'].count().plot(kind='line', marker='o', color='#E50914')
plt.title('Content Volume Added to Netflix by Year', fontsize=16)
plt.xlabel('Year Added', fontsize=12)
plt.ylabel('Number of Titles Added', fontsize=12)
plt.grid(axis='y')
plt.show()

# Chart 3: Average Duration by Content Type (Bivariate)
plt.figure(figsize=(10, 6))
# Filter for Movies (duration_type == 'min') to calculate average runtime
movie_runtime = df[df['duration_type'] == 'min']['duration_int']
# Filter for TV Shows (duration_type == 'Season') to calculate average seasons
tv_seasons = df[df['duration_type'] == 'Season']['duration_int']

# Plotting the average duration metric for the two types
avg_data = pd.DataFrame({
    'Type': ['Movie (Avg. Min)', 'TV Show (Avg. Seasons)'],
    'Average Duration': [movie_runtime.mean(), tv_seasons.mean()]
})

sns.barplot(x='Type', y='Average Duration', data=avg_data, palette=['#E50914', '#221e1f'])
plt.title('Average Duration Metric by Content Type', fontsize=16)
plt.ylabel('Duration (Min or Seasons)', fontsize=12)
plt.xlabel('Content Type', fontsize=12)
plt.show()

# -------------------------------------------------------------------
# 4. MACHINE LEARNING MODEL: PREDICTING CONTENT TYPE
# -------------------------------------------------------------------

print("\n--- Building Machine Learning Model ---")

# Define features (X) and target (y)
X = df_model.drop('type', axis=1)
y = df_model['type']

# Identify categorical and numerical features for the model
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create preprocessing pipelines
numerical_transformer = StandardScaler()
# Use handle_unknown='ignore' to safely ignore new categories in the test set/future data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the logistic regression model pipeline
# Using LogisticRegression as a binary classifier (0/1 for Movie/TV Show)
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000, random_state=42))])

# Split the data into training and testing sets
# Stratify ensures the split maintains the original proportion of Movies/TV Shows
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train the model
print("\nTraining the Logistic Regression model to predict Content Type...")
model.fit(X_train, y_train)
print("Model training complete.")

# -------------------------------------------------------------------
# 5. MODEL EVALUATION
# -------------------------------------------------------------------

print("\n--- Model Evaluation ---")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Display the classification report
print("\nClassification Report (Target: Movie=0, TV Show=1):")
print(classification_report(y_test, y_pred, target_names=['Movie', 'TV Show']))

# Display the confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Movie', 'TV Show'], yticklabels=['Movie', 'TV Show'])
plt.title('Confusion Matrix for Content Type Prediction', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

print("\n--- End of Netflix Analysis ---")
