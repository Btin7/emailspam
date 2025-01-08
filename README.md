# Spam Email Detection
made by Btin7

## Features
- **Dataset Preview**: Displays an overview of the dataset used for training the model.
- **Data Preparation**: Processes and encodes email content using `TfidfVectorizer`.
- **Model Training**: Uses a `RandomForestClassifier` to classify emails.
- **Real-Time Classification**: Allows users to input email text and get predictions instantly.

## Dataset
The app uses the SMS Spam Collection Dataset, which contains labeled SMS messages as spam or ham (not spam). The dataset can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) or other sources.

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd spam-email-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Open the app in your browser at `http://localhost:8501`.

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



