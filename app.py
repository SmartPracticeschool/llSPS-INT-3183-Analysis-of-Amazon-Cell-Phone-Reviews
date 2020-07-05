import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = models.load_model('sentiment.h5')

@app.route('/')
def home() -> 'html':
    return render_template('index.html')


@app.route('/ratings', methods = ['POST'])
def predict():
    '''
    For rendering results in the html page.
    '''

    review = request.form.values['review']

    with open('final_model', 'rb') as fout:
        cv, model = pickle.load(fout)
        X = cv.transform([review])
        pred = model.predict(X)
        if pred > 0.5:
            prediction_text = 'This is a positive review '
        else:
            prediction_text = 'This is a negative review'
    return render_template('index.html', prediction_text = prediction_text)

if __name__ == '__main__':
    app.run(debug = True)
