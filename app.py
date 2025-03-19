from flask import Flask, render_template, request, redirect, url_for, jsonify
import joblib
app = Flask(__name__)


model = joblib.load('spam_detection_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/fetch',methods=['POST'])
def fetch():
    print("fetch the content")
    new_text=request.form.get("message")
    new_text_transformed = tfidf.transform([new_text])
    prediction = model.predict(new_text_transformed)
    if prediction[0] == 1:
        result = "This is spam!"
    else:
        result = "This is not spam."

    return render_template("home.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)