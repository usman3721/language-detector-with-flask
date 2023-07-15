
from flask import Flask, request, render_template
import pickle




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index_templates.html')



@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        txt = request.form['text']
        model = pickle.load(open('model.pkl', 'rb'))
        loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
        label_encoder = pickle.load(open('label_encoder', 'rb'))
        
        t_o_b =loaded_vectorizer.transform([txt])# convert text to bag of words model (Vector)
        language = model.predict(t_o_b) # predict the language
        corr_language = label_encoder.inverse_transform(language) # find the language corresponding with the predicted value

        output = corr_language[0]

    return render_template('index_templates.html', prediction='Language is in {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

