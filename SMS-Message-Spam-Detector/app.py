from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

app = Flask(__name__)

@app.route("/")

def home():
    return render_template('home.html')


@app.route("/predict", methods = ['POST'])
def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df["Category"] = df["Category"].map({'ham':1, 'spam':2})
    X = df['Message']
    y = df['Category']

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    clf.score(X_test,y_test)

    # joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template('result.html', prediction = my_prediction)

if __name__  == '__main__':
    app.run(debug=True)



