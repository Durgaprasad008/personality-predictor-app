#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd


# In[10]:


# Reload the uploaded file
personality_df = pd.read_csv("personality_dataset.csv")


# In[11]:


personality_df


# In[12]:


# Show the first few rows and dataset shape
personality_df.head(), personality_df.shape


# In[13]:


personality_df.info()


# In[14]:


personality_df.describe()


# In[15]:


personality_df.isnull().sum()


# In[16]:


# Copy the original dataframe to keep it safe
df_encoded = personality_df.copy()


# In[17]:


# Step 1: Encode binary Yes/No columns
yes_no_mapping = {'Yes': 1, 'No': 0}
df_encoded['Stage_fear'] = df_encoded['Stage_fear'].map(yes_no_mapping)
df_encoded['Drained_after_socializing'] = df_encoded['Drained_after_socializing'].map(yes_no_mapping)


# In[18]:


# Step 2: Encode target variable (Personality: Introvert = 0, Extrovert = 1)
personality_mapping = {'Introvert': 0, 'Extrovert': 1}
df_encoded['Personality'] = df_encoded['Personality'].map(personality_mapping)


# In[19]:


# Display first few rows to verify
df_encoded.head()


# In[20]:


df_encoded.dtypes


# In[21]:


from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df_encoded.drop('Personality', axis=1)  # Remove 'Personality' column from features
y = df_encoded['Personality']  # Target label we want to predict

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check shapes
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[23]:


# Step 1: Initialize the model
model = LogisticRegression()


# In[24]:


# Step 2: Train the model
model.fit(X_train, y_train)


# In[25]:


# Step 3: Predict on test data
y_pred = model.predict(X_test)


# In[26]:


y_pred


# In[27]:


# Step 4: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# In[28]:


# Display results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)


# In[29]:


# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


# In[30]:


# Train KNN
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can experiment with different k values
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)


# In[31]:


# Evaluate Random Forest
print("🌲 Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Classification Report:\n", classification_report(y_test, rf_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))


# In[32]:


# Evaluate KNN
print("\n👬 K-Nearest Neighbors (K=5):")
print("Accuracy:", accuracy_score(y_test, knn_preds))
print("Classification Report:\n", classification_report(y_test, knn_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_preds))


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from the trained Random Forest model
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

importance_df


# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from the trained Random Forest model
importances = rf_model.feature_importances_

# Create a dataframe for visualization
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('🔍 Feature Importance - Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()


# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Define parameter grid to try
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize base model
rf = RandomForestClassifier(random_state=42)

# Randomized search with 5-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,  # tries 20 random combinations
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the randomized search model
random_search.fit(X_train, y_train)

# Best model
best_rf_model = random_search.best_estimator_

# Evaluate on test set
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred_best = best_rf_model.predict(X_test)

# Metrics
print("🌟 Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_best))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred_best))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))


# In[46]:


import pickle

# Suppose your trained model is named 'personality_classifier'
pickle_out = open("personality_classifier.pkl", "wb")
pickle.dump(best_rf_model, pickle_out)
pickle_out.close()


# In[ ]:


import pickle
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# Load the pickled model
pickle_in = open('personality_classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

# Define the prediction function
def prediction(time_spent_alone, stage_fear, social_event_attendance, going_outside,
               drained_after_socializing, friends_circle_size, post_frequency):

    # Convert categorical inputs to numerical if needed
    # For example, convert 'Yes'/'No' to 1/0
    stage_fear_num = 1 if stage_fear.lower() == 'yes' else 0
    drained_num = 1 if drained_after_socializing.lower() == 'yes' else 0

    # Prepare feature list as expected by the model
    features = [int(time_spent_alone), stage_fear_num, int(social_event_attendance),
                int(going_outside), drained_num, int(friends_circle_size), int(post_frequency)]

    prediction = classifier.predict([features])
    return prediction[0]

# Main function to define the webpage
def main():
    st.title("Personality Prediction ML App")

    html_temp = """
    <div style="background-color:green;padding:13px">
    <h1 style="color:black;text-align:center;">Streamlit Personality Classifier</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields for user to enter data
    time_spent_alone = st.text_input("Time Spent Alone (numeric)", "Type Here")
    stage_fear = st.text_input("Stage Fear (Yes/No)", "Type Here")
    social_event_attendance = st.text_input("Social Event Attendance (numeric)", "Type Here")
    going_outside = st.text_input("Going Outside (numeric)", "Type Here")
    drained_after_socializing = st.text_input("Drained After Socializing (Yes/No)", "Type Here")
    friends_circle_size = st.text_input("Friends Circle Size (numeric)", "Type Here")
    post_frequency = st.text_input("Post Frequency (numeric)", "Type Here")

    result = ""

    if st.button("Predict"):
        try:
            result = prediction(time_spent_alone, stage_fear, social_event_attendance, going_outside,
                                drained_after_socializing, friends_circle_size, post_frequency)
            st.success(f'The predicted personality is: {result}')
        except Exception as e:
            st.error(f"Error in input or prediction: {e}")

if __name__ == '__main__':
    main()

