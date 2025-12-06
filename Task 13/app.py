# Import the Streamlit library for building web apps
import streamlit as st

# Import pandas for data manipulation and analysis
import pandas as pd

# Import seaborn for statistical data visualization
import seaborn as sns

# Import matplotlib's pyplot module for basic plotting
import matplotlib.pyplot as plt

# Import Support Vector Regression (SVR) for predictive modeling
from sklearn.svm import SVR

# Import train_test_split to divide the dataset
from sklearn.model_selection import train_test_split

# Import preprocessing tools for encoding and scaling
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import metrics to evaluate regression model performance
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("🎓 Student Performance Analysis and Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")  # Load data from CSV file
    return df

df = load_data()

# Basic Overview
st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

# Fill missing values: numeric columns with mean, categorical with mode
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])     # Male=1, Female=0
df['Stress_Level'] = le.fit_transform(df['Stress_Level'])  # Depends on actual classes

# Save mappings for custom prediction later
gender_mapping = {'Female': 0, 'Male': 1}
stress_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Sidebar for feature selection
st.sidebar.title("🔧 Configure Prediction Model")
features = df.columns.tolist()
features.remove("Student_ID")
features.remove("Grades")

selected_features = st.sidebar.multiselect("Select Features for Prediction", features, default=features)

# Feature-target split
X = df[selected_features]
y = df["Grades"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
st.subheader("📈 Model Performance")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")

# Plot actual vs predicted
st.subheader("🎯 Actual vs Predicted Grades")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual Grades")
ax.set_ylabel("Predicted Grades")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

# Custom prediction
st.subheader("🧠 Predict Grade for a Custom Input")
input_data = {}

for feature in selected_features:
    if feature == "Gender":
        gender_input = st.selectbox("Select Gender", options=["Female", "Male"])
        input_data["Gender"] = gender_mapping[gender_input]

    elif feature == "Stress_Level":
        stress_input = st.selectbox("Select Stress Level", options=list(stress_mapping.keys()))
        input_data["Stress_Level"] = stress_mapping[stress_input]

    else:
        val = st.number_input(f"Enter value for {feature}", value=float(df[feature].mean()))
        input_data[feature] = val

if st.button("Predict Grade"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    st.success(f"🎓Predicted Grade: {pred[0]:.2f}")
