from flask import Flask, request,app, jsonify,url_for, render_template
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app=Flask(__name__)
model=pickle.load(open('sentiment_model.pkl','rb'))
vectorizer=pickle.load(open('tfidf_vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html') #render_template function looks for html file in templates folder 


@app.route('/predict_api',methods=['POST'])
def predict():
    data = request.form['review'] #get data from form #review is name of textarea in html
    print(data)
    vect=vectorizer.transform([data])
    my_prediction=model.predict(vect)[0] #take element out of list
    print(my_prediction)
    return render_template('index.html', prediction_text=f"Sentiment: {my_prediction}")



if __name__ == '__main__':
    app.run(debug=True)

    print("Server is running...")