from flask import Flask, request , render_template
import joblib
import numpy as np
from joblib.parallel import method

model = joblib.load('shopping_model.pkl')
app= Flask(__name__)
@app.route('/', methods= ['GET', 'POST'])
def index():
    if request.method== 'POST':
        try :
            admin = float(request.form['Administrative'])
            info = float(request.form['Informational'])
            product = float(request.form['ProductRelated'])
            exit_rate = float(request.form['ExitRates'])
            page_value= float(request.form['PageValues'])

            user= np.array([[admin,info,product,exit_rate,page_value]])
            pred=  model.predict(user)[0]
            result= 'most likely to purchase' if pred == 1 else 'not likely to purchase'

            return render_template('index.html', result = result)
        except Exception as e:
            return render_template('index.html', result = f'error: {e}')
    else:
        return render_template('index.html', result= None)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=False)
