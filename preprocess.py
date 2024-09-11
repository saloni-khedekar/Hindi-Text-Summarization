import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file
df_train = pd.read_csv("F:/VIT 3rd YEAR/6th Semester/EDI/newtrain.csv")

# Define a function to clean the text data
def clean_text(text):
    if isinstance(text, str):  # Check if the value is a string
        # Lowercasing
        text = text.lower()
        
        # Removing special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Removing stopwords
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        text = ' '.join(filtered_words)
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = text.split()
        text = ' '.join([lemmatizer.lemmatize(token) for token in tokens])
        
        return text.strip()  # Example: removing extra whitespaces and strip leading/trailing spaces
    else:
        return ''  # Return an empty string for missing values

# Clean the 'headline' and 'article' columns using the clean_text function
df_train['headline'] = df_train['headline'].apply(clean_text)
df_train['article'] = df_train['article'].apply(clean_text)

# Print the first few rows of the DataFrame to verify the changes
print(df_train.head())
