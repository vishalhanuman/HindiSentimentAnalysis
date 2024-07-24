import pandas as pd
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import pickle

# Check if preprocessed data exists
preprocessed_file = 'Trials/preprocessed_data3.csv'
if os.path.exists(preprocessed_file):
    df = pd.read_csv(preprocessed_file)
    print(True)
else:
    from preproc import preprocess_data  # Import the preprocessing function
    df = pd.read_csv('output2.csv', header=None, names=['text', 'label'])
    print(False)
    df = preprocess_data(df)
    df.to_csv(preprocessed_file, index=False)  # Save the preprocessed data for future use

# Prepare text data for model
df = df.dropna(subset=['text'])
df = df[df['text'].str.strip().astype(bool)]
hindi_data = df['text'].values
labels = df['label'].values
print(len(hindi_data))

# Tokenization and Padding
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(hindi_data)
sequences = tokenizer.texts_to_sequences(hindi_data)
word_index = tokenizer.word_index
max_sequence_len = max([len(x) for x in sequences])
X = pad_sequences(sequences, maxlen=max_sequence_len)
Y = np.array([1 if label == "1" else 0 for label in labels])
with open('Trials/tokenizer_new.pkl', 'wb') as handle:
        joblib.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# LSTM Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_sequence_len))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
history = model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stop])

# Ensemble Model
level0 = list()
level0.append(('lr', make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=1000))))
level0.append(('rf', RandomForestClassifier(n_estimators=100)))
level0.append(('knn', KNeighborsClassifier(n_neighbors=5)))
level0.append(('cart', DecisionTreeClassifier()))
level0.append(('mlp', MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500)))
level0.append(('gb', GradientBoostingClassifier()))
level0.append(('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')))
level0.append(('svm', make_pipeline(StandardScaler(), SVC(probability=True))))
level0.append(('bayes', GaussianNB()))
level1 = LogisticRegression(solver='lbfgs', max_iter=1000)
ensemble_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)




ensemble_model.fit(X_train, Y_train)

# Evaluate the models
def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    roc_auc = roc_auc_score(Y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc



lstm_metrics = evaluate_model(model, X_test, Y_test)
ensemble_metrics = evaluate_model(ensemble_model, X_test, Y_test)

print("LSTM Model - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, ROC-AUC: {:.4f}".format(*lstm_metrics))
print("Ensemble Model - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, ROC-AUC: {:.4f}".format(*ensemble_metrics))

# Save the models
joblib.dump(model, 'Trials/lstm_verbs.pkl')
joblib.dump(ensemble_model, 'Trials/ensemble_model_verbs.pkl')
