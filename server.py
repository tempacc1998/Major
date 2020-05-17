from flask import Flask, render_template, request
from predict import main

app = Flask(__name__)

#Handling index page
@app.route('/')
def homepage():
    return render_template('index.html')


#Handling result page
@app.route('/result', methods=['POST', 'GET'])
def result():
    file = request.files['dataset']
    interval = request.form['interval']

    actualValue,lPredictedValue,sPredictedValue = main(file, interval)

    return render_template('result.html', a = actualValue,lp =lPredictedValue,sp=sPredictedValue,len=len(actualValue))


# Running the app
if __name__ == '__main__':
    app.run(debug=True)
