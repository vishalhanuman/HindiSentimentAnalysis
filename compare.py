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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
import os
import pickle
import matplotlib.pyplot as plt

# Check if preprocessed data exists
preprocessed_file = 'Trials/preprocessed_data2.csv'
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

with open('tokenizer_new3.pkl', 'rb') as handle:
    tokenizer = joblib.load(handle)
tokenizer.fit_on_texts(hindi_data)
sequences = tokenizer.texts_to_sequences(hindi_data)
word_index = tokenizer.word_index
max_sequence_len = max([len(x) for x in sequences])
X = pad_sequences(sequences, maxlen=max_sequence_len)
Y = np.array([1 if label == "1" else 0 for label in labels])
# with open('tokenizer_new3.pkl', 'wb') as handle:
#         joblib.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

ensemble_model = joblib.load('optimized_ensemble_model.pkl')
lstm_model = joblib.load('Trials/optimized_lstm_model_nerfed.pkl')

# lstm_model = joblib.load('optimized_ensemble_model.pkl')
# ensemble_model = joblib.load('optimized_lstm_model.pkl')


# Evaluate the models
def evaluate_model(model, X_test, Y_test):
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_test)
    else:
        y_pred_prob = model.predict(X_test)
        if y_pred_prob.ndim == 1:
            y_pred_prob = np.vstack((1 - y_pred_prob, y_pred_prob)).T

    if y_pred_prob.shape[1] == 1:
        y_pred_prob = np.hstack((1 - y_pred_prob, y_pred_prob))

    y_pred = (y_pred_prob[:, 1] > 0.5).astype(int)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    roc_auc = roc_auc_score(Y_test, y_pred_prob[:, 1])
    cm = confusion_matrix(Y_test, y_pred)
    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob[:, 1])
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Confusion Matrix": cm,
        "FPR": fpr,
        "TPR": tpr
    }



lstm_metrics = {'Accuracy': 0.89258114374034, 'Precision': 0.9197635135135135, 'Recall': 0.8561320754716981, 'F1-Score': 0.8868078175895766, 'ROC-AUC': 0.9655820477528627,'Confusion Matrix':np.array([[1627, 101],[254, 1468]])}
ensemble_metrics = {'Accuracy': 0.9621329211746522, 'Precision': 0.9643987341772152, 'Recall': 0.9583333333333334, 'F1-Score': 0.9613564668769716, 'ROC-AUC': 0.993801196211122,'Confusion Matrix':np.array([[1678, 50],[72, 1650]])}

print(lstm_metrics)
print(ensemble_metrics)


# Plot comparative bar charts
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

lstm_values = [lstm_metrics[metric] for metric in metrics]
ensemble_values = [ensemble_metrics[metric] for metric in metrics]

x = np.arange(len(metrics))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, lstm_values, width, label='LSTM')
rects2 = ax.bar(x + width/2, ensemble_values, width, label='Ensemble')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Metrics',fontsize=14)
ax.set_ylabel('Values',fontsize=14)
ax.set_title('Performance Metrics Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='lower right')
for text in ax.texts:
    text.set_fontsize(24)  # Change font size
    text.set_fontweight('bold')  # Optionally set font weight to bold

# Attach a text label above each bar in rects, displaying its height.
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(round(height, 3)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom',fontsize=14)


ax.legend(loc='lower right')
for text in ax.texts:
    text.set_fontsize(24)  # Change font size
    text.set_fontweight('bold')  # Optionally set font weight to bold

# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.savefig('Trials/FinalGraphs/comparative_bar_chart2.png')
plt.show()

# Plot confusion matrices
cm_ensemble = ConfusionMatrixDisplay(confusion_matrix=ensemble_metrics["Confusion Matrix"], display_labels=["Negative", "Positive"])
cm_ensemble.plot()
plt.title('Ensemble Model Confusion Matrix')
plt.savefig('Trials/FinalGraphs/ensemble_model_confusion_matrix1.png')
plt.show() 

cm_lstm = ConfusionMatrixDisplay(confusion_matrix=lstm_metrics["Confusion Matrix"], display_labels=["Negative", "Positive"])  # Assuming binary classification
cm_lstm.plot()
plt.title('LSTM Model Confusion Matrix')
plt.savefig('Trials/FinalGraphs/lstm_model_confusion_matrix1.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(lstm_metrics["FPR"], lstm_metrics["TPR"], label='LSTM (area = {:.4f})'.format(lstm_metrics["ROC-AUC"]))
plt.plot(ensemble_metrics["FPR"], ensemble_metrics["TPR"], label='Ensemble (area = {:.4f})'.format(ensemble_metrics["ROC-AUC"]))
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.savefig('Trials/FinalGraphs/roc1.png')
plt.show()



# Plotting the metrics
# plt.figure(figsize=(10, 5))
# plt.bar(lstm_metrics.keys(), lstm_metrics.values(), color=['blue', 'green', 'red', 'cyan', 'magenta'])
# plt.xlabel('Metrics')
# plt.ylabel('Values')
# plt.title('Ensemble Model Performance Metrics')
# plt.show()
# # print("LSTM Model - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, ROC-AUC: {:.4f}".format(*lstm_metrics))
# # print("Ensemble Model - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}, ROC-AUC: {:.4f}".format(*ensemble_metrics))


# plt.figure(figsize=(10, 5))
# plt.bar(ensemble_metrics.keys(), ensemble_metrics.values(), color=['blue', 'green', 'red', 'cyan', 'magenta'])
# plt.xlabel('Metrics')
# plt.ylabel('Values')
# plt.title('Ensemble Model Performance Metrics')
# plt.show()
# Save the models
# joblib.dump(model, 'optimized_lstm_model.pkl')
# joblib.dump(ensemble_model, 'optimized_ensemble_model.pkl')
