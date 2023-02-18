from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'divorce-pred-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        se=int(request.form['Sorry_end'])
        igd=int(request.form['Ignore_diff'])
        bc=int(request.form['begin_correct'])
        c=int(request.form['Contact'])
        st=int(request.form['Special_time'])
        nht=int(request.form['No_home_time'])
        s2=int(request.form['2_strangers'])
        eh=int(request.form['enjoy_holiday'])
        et=int(request.form['enjoy_travel'])
        cg=int(request.form['common_goals'])

        
        
        data = np.array([[se, igd, bc, c, st, nht, s2, eh, et, cg]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)