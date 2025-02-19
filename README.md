# Email/SMS Spam Classifier

## Overview

This is a simple Streamlit web application that classifies a given text message as either "Spam" or "Not Spam" using a trained machine learning model. The model utilizes Natural Language Processing (NLP) techniques to preprocess and analyze text data.

## Features

- Accepts user input for an SMS or email message.
- Preprocesses the text by tokenization, stemming, and removing stopwords and punctuation.
- Converts the processed text into a numerical vector representation using TF-IDF.
- Uses a pre-trained machine learning model to predict whether the message is spam or not.
- Displays the classification result in real-time.

## Requirements

Ensure you have the following dependencies installed before running the application:

```bash
pip install streamlit nltk scikit-learn pickle5 pandas numpy
```

## File Structure

- `app.py`: The main application script.
- `Untitled13.ipynb`: The Jupyter notebook for data preprocessing and model training.
- `vectorizer.pkl`: The TF-IDF vectorizer model.
- `model.pkl`: The trained spam classification model.
- `spam.csv`: The dataset used for training.

## How to Run the Application

1. Install the required dependencies.
2. Run the Streamlit app with the following command:
   ```bash
   streamlit run app.py
   ```
3. Enter an SMS or email message in the text input area.
4. Click the "Predict" button to classify the message.
5. The result will be displayed as either "Spam" or "Not Spam."

## Data Preprocessing and EDA

The data preprocessing and exploratory data analysis (EDA) steps are performed in `Untitled13.ipynb`. The notebook includes:

1. **Loading Data**: The dataset (`spam.csv`) is read using Pandas.
2. **Data Cleaning**: Unnecessary columns are removed, and missing values are handled.
3. **Text Preprocessing**:
   - Conversion to lowercase.
   - Tokenization using NLTK.
   - Removal of stopwords and punctuation.
   - Stemming using the Porter Stemmer.
4. **Exploratory Data Analysis (EDA)**:
   - Understanding class distributions.
   - Visualizing word frequencies.
   - Checking message length distributions.
5. **Model Training**:
   - TF-IDF transformation of text data.
   - Training a machine learning model (e.g., Naive Bayes or Logistic Regression) on preprocessed text.
   - Evaluating the model’s performance.

## Model Details

- The model was trained on a dataset of SMS and email messages labeled as spam or not spam.
- A TF-IDF vectorizer was used to transform text data into numerical representations.
- A machine learning model (e.g., Naive Bayes, Logistic Regression, or similar) was trained to classify messages based on their content.

## Notes

- The `vectorizer.pkl` and `model.pkl` files must be present in the same directory as `app.py`.
- The first time running the application, you may need to download NLTK stopwords using:
  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('punkt')
  ```
- The accuracy and effectiveness of the classifier depend on the dataset used for training.

## Author

- Developed by Lakshay
- Contact: connect.lakshay@outlook.com

## License

This project is licensed under the MIT License.

