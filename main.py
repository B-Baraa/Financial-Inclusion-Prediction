import pandas as pd
# load the dataset
df = pd.read_csv("Financial_inclusion_dataset.csv")
# Display the first 5 rows
df.head()
# Display data types, non-null counts, and memory usage
df.info()
print(df.shape)
df_cleaned = df.copy()
# check missing values
print(df.isnull().sum())
# check duplicates values

print(df.duplicated().sum())
# Drop duplicate rows if any
# Drop duplicate rows if any
df_cleaned = df_cleaned.drop_duplicates()
# Handle outliers
# Import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
#  List of numeric columns to check for outliers
numeric_columns = ['household_size', 'age_of_respondent', 'year']
#  Set up the plot size and number of subplots
plt.figure(figsize=(15, 5))
# Create one boxplot for each numeric column
for i, column in enumerate(numeric_columns):
    plt.subplot(1, 3, i + 1)  # 1 row, 3 columns, subplot i+1
    sns.boxplot(x=df_cleaned[column])
    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("All_Outliers_Boxplots.png")
print("✔ All boxplots saved as 'All_Outliers_Boxplots.png'")
# Function to remove outliers using the IQR method
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    original_count = data.shape[0]
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    removed = original_count - data.shape[0]
    print(f"✔ Outliers removed from '{column}': {removed} rows")
    return data

# Apply to 'household_size' and 'age_of_respondent'
df_cleaned = remove_outliers_iqr(df_cleaned, 'household_size')
df_cleaned = remove_outliers_iqr(df_cleaned, 'age_of_respondent')
# Encode categorical features
from sklearn.preprocessing import LabelEncoder

# 🧹 List of categorical columns to encode
categorical_cols = [
    'country', 'location_type', 'cellphone_access', 'gender_of_respondent',
    'relationship_with_head', 'marital_status', 'education_level', 'job_type', 'bank_account'
]

# Create a LabelEncoder object
le = LabelEncoder()

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    print(f"✔ Encoded '{col}'")

# Check the result
print("\n🎯 Encoded Data Sample:")
print(df_cleaned[categorical_cols].head())
# train and test a machine learning classifier
from sklearn.model_selection import train_test_split

# Target variable
y = df_cleaned['bank_account']

# Features (drop the target and any unnecessary ID columns)
X = df_cleaned.drop(['bank_account', 'uniqueid'], axis=1)
# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 # Import and train model XGBoost
#  Import the XGBoost classifier
from xgboost import XGBClassifier

# Initialize the model (with common good default parameters)
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'  # to suppress warning
)

# Train the model
xgb_model.fit(X_train, y_train)
print("✅ XGBoost model trained successfully!")

# Predict on test data
y_pred = xgb_model.predict(X_test)

# Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy of trained model: {accuracy:.4f}")
# Train the model
xgb_model.fit(X_train, y_train)
print(" XGBoost model trained successfully!")

# ✅ Generate predictions and evaluate
# ✅ Generate predictions and evaluate
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Accuracy of model: {accuracy:.4f}")

import joblib
#  Save the trained model to a file
joblib.dump(xgb_model, "xgboost_model.pkl")
print("✔ Model saved as 'xgboost_model.pkl'")

import streamlit as st
import pandas as pd
# Load model
model = joblib.load('xgboost_model.pkl')

st.title("Financial Inclusion Prediction")

# Input fields
household_size = st.number_input("Household Size", min_value=1)
age_of_respondent = st.number_input("Age of Respondent", min_value=18)
location_type = st.selectbox("Location Type", ["Urban", "Rural"])
gender_of_respondent = st.selectbox("Gender", ["Male", "Female"])
education_level = st.selectbox("Education Level", ["No formal education", "Primary", "Secondary", "Tertiary"])
job_type = st.selectbox("Job Type", ["Self-employed", "Government", "Formally employed", "Other"])

if st.button("Predict"):
    # Convert input to dataframe
    input_df = pd.DataFrame({
        'household_size': [household_size],
        'age_of_respondent': [age_of_respondent],
        'location_type': [location_type],
        'gender_of_respondent': [gender_of_respondent],
        'education_level': [education_level],
        'job_type': [job_type]
    })

    # Encode using the same label encoder logic or one-hot encoding as used in training
    # (You may need to save the encoder too for consistent transformation)

    # For now, assume encoded correctly
    prediction = model.predict(input_df)
    st.write("Prediction: Has Bank Account" if prediction[0] == 1 else "Prediction: No Bank Account")
