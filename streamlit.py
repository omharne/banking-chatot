from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import pandas as pd
import numpy as np

# Load the preprocessed data
df_result = pd.read_csv('df_result_ofdf1plusdf2_index.csv')

# Define the TD-IDF vectorizer and fit it to the data
tvec3 = TfidfVectorizer()
tvec3.fit(df_result['Question'].str.lower())

# Define the support vector machine model and fit it to the data
clf3 = SVC(kernel='linear')
clf3.fit(tvec3.transform(df_result['Question'].str.lower()), df_result['Class'])

# Define a function to get the answer to a given question
def get_answer(question):
    # Vectorize the question
    question_tdidf = tvec3.transform([question.lower()])
    
    # Calculate the cosine similarity between both vectors
    cosine_sims = cosine_similarity(question_tdidf, tvec3.transform(df_result['Question'].str.lower()))

    # Get the index of the most similar text to the query
    most_similar_idx = np.argmax(cosine_sims)

    # Get the predicted class of the query
    predicted_class = clf3.predict(question_tdidf)[0]
    
    # If the predicted class is not the same as the actual class, return an error message
    if predicted_class != df_result.iloc[most_similar_idx]['Class']:
        return {'error': 'Could not find an appropriate answer.'}
    
    # Get the answer and construct the response
    answer = df_result.iloc[most_similar_idx]['Answer']
    response = {
        'answer': answer,
        'predicted_class': predicted_class
    }
    
    return response
from flask import Flask, request, jsonify, render_template

# Create a Flask app
app = Flask(__name__,template_folder='template')

# Define the route for the chatbot web interface
@app.route('/')
def index():
    return render_template('bank.html')

# Define the API route for predicting answers
@app.route('/predict', methods=['POST'])
def predict():
    # Get the question from the request
    question = request.form['question']

    # Get the answer to the question
    response = get_answer(question)
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
