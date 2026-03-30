import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. DATA LOADING & AUGMENTATION
# Creating a larger internal dataset to simulate a real CSV file
data = {
    'Category': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']*10,
    'Message': [
        "Are we meeting at the VIT library for the project?",
        "URGENT: Your mobile number has won £1,000,000. Claim now by calling 09061701461.",
        "I'll be home late, don't wait for dinner.",
        "FREE RINGTONE! Text 'GO' to 89545 to receive your gift.",
        "Can you send me the notes for the Digital Literacy class?",
        "WINNER! You've been selected for a free holiday. Click the link below.",
        "Did you see the new placement notification on VTOP?",
        "Account Alert: Unusual activity detected. Log in at bit.ly/secure-link.",
        "Let's catch up over coffee this weekend.",
        "Get a personal loan at 0% interest today! No documents required."
    ]*10
}
df = pd.DataFrame(data)

# 2. HUMAN-LIKE PREPROCESSING FUNCTION
# This shows you know how to clean "noisy" digital data
def clean_text(text):
    text = text.lower() # Lowercase
    text = "".join([char for char in text if char not in string.punctuation]) # Remove punctuation
    return text

df['Clean_Message'] = df['Message'].apply(clean_text)

# 3. FEATURE ENGINEERING (TF-IDF)
# TF-IDF is better for Consulting as it weighs "rare" spam words more heavily
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['Clean_Message'])
y = df['Category']

# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. MODEL COMPARISON (The "Analyst" Approach)
# We test two models to see which is better for this business case
models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"--- {name} Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print(classification_report(y_test, predictions))

# 6. VISUALIZATION: CONFUSION MATRIX
# High-quality visual for your project report
best_model = models["Naive Bayes"]
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap='RdPu', fmt='g', 
            xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
plt.title('Final Model: Spam vs Ham Accuracy')
plt.show()

# 7. LIVE PREDICTION SYSTEM
def spam_checker(user_input):
    cleaned = clean_text(user_input)
    vectorized = tfidf.transform([cleaned])
    prediction = best_model.predict(vectorized)
    probability = best_model.predict_proba(vectorized)
    return prediction[0], np.max(probability)

# Test with a new "Human" example
msg = "Hey, can you help me with the Fourier Transform numericals?"
label, confidence = spam_checker(msg)
print(f"Input: {msg}\nResult: {label.upper()} (Confidence: {confidence*100:.2f}%)")
