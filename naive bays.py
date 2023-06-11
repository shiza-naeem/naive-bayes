import pandas as pd

try:
    df = pd.read_csv('spam.csv')
    print(df)
except FileNotFoundError:
    print("Error: File 'spam.csv' not found.")
except pd.errors.ParserError:
    print("Error: Unable to parse the CSV file.")
    df.groupby('Category').describe()
    df['spam'] = df['Category'].apply (lambda x: 1 if x== 'spam' else 0)
df.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.Message, df.spam)
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Example data for x_train
x_train = pd.Series(["This is the first document.", "This document is the second document.", "And this is the third one.", "Is this the first document?"])

try:
    v = CountVectorizer()
    x_train_count = v.fit_transform(x_train.values)
    print(x_train_count.toarray()[:4])
except Exception as e:
    print("Error:",e)
    from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_count,y_train)