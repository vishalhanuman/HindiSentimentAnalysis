import pandas as pd
import re
import stanza

# Load Stanza model for Hindi POS tagging
stanza.download('hi')
nlp = stanza.Pipeline('hi',use_gpu=True)

# Load stop words and punctuations
with open('hi_stopwords.txt', 'r', encoding='utf-8') as file:
    hi_stopwords = set(file.read().splitlines())

with open('stopwords_en.txt', 'r', encoding='utf-8') as file:
    en_stopwords = set(file.read().splitlines())

punctuations = set(['nn', 'n', '।', '/', '`', '+', '\\', '"', '?', '(', '$', '@', '[', '_', "'", '!', ',', ':', '^', '|', ']', '=', '%', '&', '.', ')', '(', '#', '*', '', ';', '-', '}', '|', '"'])

stopword_punct_list = hi_stopwords.union(en_stopwords).union(punctuations)

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_urls(text):
    text = text.lower()
    text = re.sub(r'((www.[^s]+)|(https?://[^s]+))', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\bbit\.ly/\S+', '', text)
    text = re.sub(r'\b\w*[A-Za-z]\w*\b', '', text)  # remove English words
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = remove_emojis(text)
    text = remove_urls(text)
    if pd.isna(text):
        return None
    # Tokenize and POS tagging using Stanza
    doc = nlp(text)
    tokens = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text.lower() not in stopword_punct_list and word.upos in ['NOUN', 'ADJ', 'VERB']:
                tokens.append(word.text)
    if len(tokens) == 0:
        return None
    return ' '.join(tokens)

def preprocess_data(df):
    print("Preprocessing")
    print(df['text'])
    df['text'] = df['text'].apply(preprocess_text)
    
    df.to_csv('Trials/preprocessed_data3.csv', index=False) 

    return df

if __name__ == "__main__":
    file_path = "Datasets/hi_3500_1.ods"
    df = pd.read_excel(file_path, engine='odf')
    df.columns = ['text', 'label']
    df = preprocess_data(df)
    df.to_csv('preprocessed_data.csv', index=False)  # Save the preprocessed data
