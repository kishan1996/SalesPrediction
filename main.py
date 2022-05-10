from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__) 

@app.route('/',methods=['GET']) 
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) 
@cross_origin()
def index():
    if request.method == 'POST':
        try:
       
            store=float(request.form['Store'])
            Dept = float(request.form['Dept'])
            IsHoliday = float(request.form['IsHoliday'])
            Size = float(request.form['Size'])
            Week = float(request.form['Week'])
            Type = float(request.form['Type'])
            Year = request.form['Year']
            load_saved_model_file = open('src/SavedModel.sav', 'rb')
            loaded_model = pickle.load(load_saved_model_file)
      
            if Type == 1:
                Type_A = 1
                Type_B = 0
                Type_C = 0
            elif Type == 2:
                Type_A = 0
                Type_B = 1
                Type_C = 0
            elif Type == 3:
                Type_A = 0
                Type_B = 0
                Type_C = 1


        
            prediction=loaded_model.predict([[store, Dept, IsHoliday, Size, Week,  Year,Type_A,Type_B,Type_C]])
            print('prediction is', prediction)
            return render_template('results.html',prediction=prediction)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
  
    else:
        return render_template('index.html')



if __name__ == "__main__":
  
	app.run(debug=False)