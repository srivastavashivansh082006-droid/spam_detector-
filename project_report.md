Project Documentation Report
Spam Message Classifier (Python / ML)
Generated from uploaded source file: app.py

1. Executive Summary
This project is a basic machine learning spam classifier that labels text messages as either 'spam' or 'ham' (normal message). It demonstrates an end-to-end NLP workflow including dataset construction, text cleaning, TF-IDF feature extraction, model training, model comparison, visual evaluation using a confusion matrix, and a live prediction function.
2. Project Purpose
The goal of this project is to show how text classification can be used to automatically detect unwanted promotional or suspicious messages. This is a common real-world use case in email, SMS, support inboxes, and security systems.
3. Technology Stack
•	Python
•	Pandas and NumPy for data handling
•	Matplotlib and Seaborn for visualization
•	Scikit-learn for machine learning and NLP pipeline
4. Workflow Overview
1.	Create an internal dataset of spam and ham messages
2.	Clean raw text using lowercase conversion and punctuation removal
3.	Convert text into TF-IDF numerical vectors
4.	Split data into training and test sets
5.	Train and compare two models: Multinomial Naive Bayes and Random Forest
6.	Evaluate performance using accuracy, classification report, and confusion matrix
7.	Deploy a simple prediction function for new user input
5. Code Analysis
5.1 Imports
The script imports data science, NLP, visualization, and machine learning libraries. These are sufficient for a beginner-level spam detection pipeline.
5.2 Dataset Construction
Instead of reading from an external CSV file, the code manually creates a small synthetic dataset using a Python dictionary. The list of 10 messages is repeated 10 times, creating 100 rows. This helps the script run independently, but it also introduces duplicated data and unrealistic patterns.
5.3 Text Preprocessing
The clean_text() function lowercases all text and removes punctuation. This is a useful baseline preprocessing step, but it does not include stemming, lemmatization, URL normalization, number masking, or stopword customization.
5.4 Feature Engineering
The project uses TfidfVectorizer(stop_words='english') to transform text into weighted numerical features. This is a strong choice for spam detection because unusual promotional terms receive more importance.
5.5 Train/Test Split
The dataset is split using train_test_split() with test_size=0.2 and random_state=42. This creates a reproducible evaluation setup.
5.6 Model Training and Comparison
Two models are trained: MultinomialNB and RandomForestClassifier. This is a good educational choice because it compares a classic NLP baseline with a more general ensemble method.
5.7 Evaluation
The script prints accuracy and a classification report for each model, then visualizes a confusion matrix for the Naive Bayes model. This provides both numerical and visual interpretation of performance.
5.8 Prediction Function
The spam_checker() function allows new text messages to be classified. It returns both the predicted label and the model confidence score, which makes the project more practical.
6. Strengths
•	Simple and readable end-to-end machine learning workflow
•	Good beginner-friendly NLP project structure
•	Compares multiple classifiers instead of using only one model
•	Includes visualization and a reusable prediction function
•	Can be demonstrated easily in a notebook or viva/project presentation
7. Limitations / Issues
•	The dataset is synthetic and heavily duplicated, which can inflate model performance.
•	There is no external CSV loading despite comments suggesting a realistic dataset.
•	RandomForestClassifier is used without a fixed random_state, reducing reproducibility.
•	No stratified split is used, which may matter on real datasets.
•	The confusion matrix assumes label order manually as ['Normal', 'Spam']; this may not always match encoded class order.
•	No persistence mechanism is included (e.g., saving the trained model with joblib/pickle).
•	The project lacks modular structure (all logic is inside one script).
•	No input validation or exception handling is implemented.
8. Recommended Improvements
•	Replace the synthetic dataset with a real SMS spam CSV dataset.
•	Use stratify=y in train_test_split() for balanced splitting.
•	Set random_state in RandomForestClassifier for reproducibility.
•	Add more preprocessing such as stemming, lemmatization, URL and number normalization.
•	Save the trained vectorizer and best model using joblib.
•	Refactor the script into reusable functions or classes.
•	Add ROC-AUC, precision-recall metrics, and cross-validation.
•	Build a simple web interface using Streamlit or Flask for live testing.
9. Suggested Production Project Structure
spam-classifier/
│
├── data/
│   └── spam.csv
├── models/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── app.py
├── requirements.txt
└── README.md
10. Function Documentation
Function	Purpose	Notes
clean_text(text)	Cleans raw message text	Lowercases and removes punctuation
spam_checker(user_input)	Predicts spam/ham for new input	Returns predicted class and confidence
11. Appendix: Source Code
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
12. Conclusion
Overall, this is a solid academic mini-project for demonstrating machine learning-based text classification. Its main value is educational rather than production readiness. With a real dataset, better evaluation, and modular packaging, it could be upgraded into a more robust spam detection application.
