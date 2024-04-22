import re
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def data_cleaning(text: str) -> str:
    cleaned_txt = re.sub(r'http\S+', '', text)
    cleaned_txt = re.sub(r'[^a-zA-Z\s\d]', '', cleaned_txt)
    # cleaned_txt = lower_case(cleaned_txt)
    cleaned_txt = remove_stop_words(cleaned_txt)
    cleaned_txt = lemmatize(cleaned_txt)
    return cleaned_txt


# def lower_case(cleaned_text: str) -> str:
#     return cleaned_text.lower()


def remove_stop_words(cleaned_txt: str) -> str:
    stop_words = set(stopwords.words('english'))
    token = word_tokenize(cleaned_txt)
    filtered_tokens = [word for word in token if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


def lemmatize(cleaned_text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    token = word_tokenize(cleaned_text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in token]
    return ' '.join(lemmatized_tokens)