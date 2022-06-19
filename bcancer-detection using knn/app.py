from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

@app.route('/',methods=['get'])
def input():
    return render_template('input.html')

@app.route('/output',methods=['post'])
def output():

    radius_mean=float(request.form['radius_mean'])
    perimeter_mean=float(request.form['perimeter_mean'])
    area_mean=float(request.form['area_mean'])
    con_points_mean=float(request.form['concave points_mean'])
    radius_worst=float(request.form['radius_worst'])
    perimeter_worst=float(request.form['perimeter_worst'])
    area_worst=float(request.form['area_worst'])
    con_points_worst=float(request.form['concave points_worst'])
        

    filename='knn4bcancer.pickle'
    loaded_model=pickle.load(open(filename,'rb'))
    data=np.array([[radius_mean, perimeter_mean, area_mean,con_points_mean,radius_worst,perimeter_worst,area_worst,con_points_worst]])
    mypred = loaded_model.predict(data)

    return render_template('output.html',prediction=mypred)



if __name__=='__main__':
    app.run()