from flask import Flask, render_template, request, jsonify
from pickle import load

# Load the saved files
model = load(open('spam_model.pkl', 'rb'))
vectorizer = load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    
    # Preprocess
    vec = vectorizer.transform([message.lower()])
    prediction = model.predict(vec)[0]
    
    result = "SPAM" if prediction == 1 else "HAM"
    color = "red" if prediction == 1 else "green"
    
    return jsonify({
        'result': result,
        'color': color,
        'message': message
    })

if __name__ == '__main__':
    app.run(debug=True)