import pandas as pd

df = pd.read_csv('emails.csv')

df['text'] = df['text'].str.replace(r'^(Subject:\s*|re\s*:\s*)', '', regex=True)

df.to_csv('emails_cleaned.csv', index=False)
