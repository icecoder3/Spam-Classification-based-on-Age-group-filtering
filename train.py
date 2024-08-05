import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
import pickle

# Step 1: Load and preprocess the dataset
data = pd.read_csv('newdatasetemail.csv')

# Step 2: Split data into features and target
X = data['EMAILS']
y = data[['advertisement', 'spam', 'children', 'youngadult', 'adult']]

# Step 3: Vectorize the text data
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

# Step 4: Train the model
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
multi_target_model = MultiOutputClassifier(SVC(kernel='linear'), n_jobs=-1)
multi_target_model.fit(X_train, y_train)

# Step 5: Save the model and vectorizer
with open('multi_target_model.pkl', 'wb') as f:
    pickle.dump(multi_target_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
