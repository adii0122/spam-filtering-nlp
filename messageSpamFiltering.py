import os
import sys
import string

# Ensure Windows terminal handles UTF-8 output safely
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# Download required NLTK resources quietly without opening GUI
nltk.download('stopwords', quiet=True)

# Locate dataset in data/ or root directory
dataset_candidates = [
    os.path.join(os.path.dirname(__file__), "data", "spam.csv"),
    "data/spam.csv",
    "spam.csv"
]
dataset_path = None
for path in dataset_candidates:
    if os.path.exists(path):
        dataset_path = path
        break

if not dataset_path:
    raise FileNotFoundError("Could not find 'spam.csv' in 'data/' or the current working directory.")

data = pd.read_csv(dataset_path, encoding='latin-1')

# Support both Kaggle format (v1: label, v2: message) and processed format (Class: label, Message: text)
text_col = 'v2' if 'v2' in data.columns else ('Message' if 'Message' in data.columns else data.columns[0])
label_col = 'v1' if 'v1' in data.columns else ('Class' if 'Class' in data.columns else data.columns[1])

data['length'] = data[text_col].apply(len)
print(f"Loaded dataset from: {dataset_path}")
print(f"Using text column: '{text_col}', label column: '{label_col}'")
print(data.head())

# Instantiate stemmer and stopwords set once for high performance
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))
punct_table = str.maketrans('', '', string.punctuation)

def preprocess(message):
    message = str(message).translate(punct_table)
    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stop_words]
    return " ".join(words)

textFeatures = data[text_col].copy()
textFeatures = textFeatures.apply(preprocess)

# TF-IDF Vectorization with proper keyword argument
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(textFeatures)

df = pd.DataFrame(textFeatures)
output_features_path = os.path.join("data", "Text Features.csv") if os.path.exists("data") else "textFeatures.csv"
df.to_csv(output_features_path, index=False)
print(f"Saved text features to: {output_features_path}")

features_train, features_test, labels_train, labels_test = train_test_split(
    features, data[label_col], test_size=0.3, random_state=111
)

# 1. Support Vector Machine
print("\nEvaluation for SVM \n")
svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(features_train, labels_train)
prediction = svc.predict(features_test)

acc = accuracy_score(labels_test, prediction)
prec = precision_score(labels_test, prediction, pos_label='spam')
recall = recall_score(labels_test, prediction, pos_label='spam')
f1 = f1_score(labels_test, prediction, pos_label='spam')

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(labels_test, prediction))
print("\nClassification Report:")
print(classification_report(labels_test, prediction))

# 2. Multinomial Naive Bayes
print("\nEvaluation for MultNB \n")
mnb = MultinomialNB(alpha=0.2)
mnb.fit(features_train, labels_train)
prediction = mnb.predict(features_test)

acc = accuracy_score(labels_test, prediction)
prec = precision_score(labels_test, prediction, pos_label='spam')
recall = recall_score(labels_test, prediction, pos_label='spam')
f1 = f1_score(labels_test, prediction, pos_label='spam')

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(labels_test, prediction))
print("\nClassification Report:")
print(classification_report(labels_test, prediction))

# 3. Decision Tree
print("\nEvaluation for Decision Tree \n")
dtree = DecisionTreeClassifier(random_state=111)
dtree.fit(features_train, labels_train)
prediction = dtree.predict(features_test)

acc = accuracy_score(labels_test, prediction)
prec = precision_score(labels_test, prediction, pos_label='spam')
recall = recall_score(labels_test, prediction, pos_label='spam')
f1 = f1_score(labels_test, prediction, pos_label='spam')

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(labels_test, prediction))
print("\nClassification Report:")
print(classification_report(labels_test, prediction))

# ============================================================================
# Interactive Message Scanner (Real-Time Inference)
# ============================================================================
import math
import sys

def scan_message(message_text, model=svc, vectorizer=vectorizer):
    """
    Preprocess, vectorize, and classify a single SMS message.
    Returns the predicted label ('spam' or 'ham') and confidence probability.
    """
    cleaned = preprocess(message_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    # Calculate calibrated confidence using decision function
    if hasattr(model, "decision_function"):
        dist = model.decision_function(vec)[0]
        prob = 1 / (1 + math.exp(-dist))
        confidence = prob if pred == 'spam' else (1 - prob)
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        confidence = max(probs)
    else:
        confidence = 0.95

    return pred, confidence

print("\n" + "=" * 25 + " SCANNING TEST MESSAGES " + "=" * 25)

# Check if a custom message was passed via CLI: python messageSpamFiltering.py --scan "message"
custom_msg = None
if len(sys.argv) > 1:
    if sys.argv[1] == "--scan" and len(sys.argv) > 2:
        custom_msg = " ".join(sys.argv[2:])
    elif not sys.argv[1].startswith("-"):
        custom_msg = " ".join(sys.argv[1:])

if custom_msg:
    pred, conf = scan_message(custom_msg)
    status_label = "[SPAM DETECTED]" if pred == "spam" else "[HAM / LEGITIMATE]"
    print(f"\nInput Message: \"{custom_msg}\"")
    print(f"Scan Verdict : {status_label}")
    print(f"Confidence   : {conf * 100:.2f}%\n")
else:
    # Scan representative test messages
    sample_messages = [
        "URGENT! You have won a 2,000 Cash Prize Guaranteed! Call 09064019788 from landline. Claim Code: K52. Valid 12hrs only. 150ppm.",
        "Hey are you there in the room? Going for dinner with everyone in 15 minutes, let me know if you want to join!",
        "ALERT: Your 2024 Account Statement shows 786 unredeemed bonus points! Call 08719180248 now. Identifier Code: 45239. Offer expires tonight.",
        "Please find the updated project report attached. I made the changes to section 2 as discussed. Let's sync tomorrow morning.",
        "As per your request 'Melle Melle' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune."
    ]

    for i, msg in enumerate(sample_messages, 1):
        pred, conf = scan_message(msg)
        status_label = "[SPAM]" if pred == "spam" else "[HAM]"
        print(f"\n[{i}] Message: \"{msg}\"")
        print(f"    Scan Verdict: {status_label} (Confidence: {conf * 100:.1f}%)")

    print("\n" + "-" * 70)
    print("Tip: You can scan any custom message from the terminal using:")
    print("     python messageSpamFiltering.py --scan \"Your SMS message text here\"")
    print("-" * 70 + "\n")