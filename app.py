from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
loaded_model, vectorizer = joblib.load("spam_detection_model.pkl")

# Create a route to display the form
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the email input from the form
        email = request.form.get("email")

        # Preprocess the email using CountVectorizer
        email_vectorized = vectorizer.transform([email])

        # Use the trained model to make a prediction
        prediction = loaded_model.predict(email_vectorized)[0]

        return render_template("result.html", email=email, prediction=prediction)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
