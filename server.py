import model # Import the python file containing the ML model
from flask import Flask, request, render_template,jsonify # Import flask libraries

app = Flask(__name__,template_folder="templates")


@app.route('/home')
def home():
    return render_template('home.html') # Render home.html


@app.route('/classify',methods=['POST','GET'])
def classify_type():
    try:
        sepal_len = request.args.get('slen') # Get parameters for sepal length
        sepal_wid = request.args.get('swid') # Get parameters for sepal width
        petal_len = request.args.get('plen') # Get parameters for petal length
        petal_wid = request.args.get('pwid') # Get parameters for petal width


        variety = model.classify(sepal_len, sepal_wid, petal_len, petal_wid)

        return render_template('output.html', variety=variety)
    except:
        return 'Error'

# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)        