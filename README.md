# 📧 Spam Email Detection
made by Btin7

## Features
- **Dataset Preview**: Displays an overview of the dataset used for training the model.
- **Data Preparation**: Processes and encodes email content using `TfidfVectorizer`.
- **Model Training**: Uses a `RandomForestClassifier` to classify emails.
- **Real-Time Classification**: Allows users to input email text and get predictions instantly.


Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Requirements
- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas
- Numpy

## File Structure
- `app.py`: Main Streamlit application.
- `requirements.txt`: List of required Python libraries.
- `spam.csv`: Dataset used for training and testing.

## Usage
- Paste the email content into the text box on the sidebar.
- Click "Classify" to see if the email is predicted as spam or not.
- View the prediction probability for both classes (Spam and Not Spam).

## Example
Input:
```
Congratulations! You've won a free gift card. Click here to claim your prize.
```
Output:
- Prediction: **Spam**
- Probability: Spam: 0.92, Not Spam: 0.08



