from flask import Flask, render_template, request
from make_prediction import make_prediction, get_company_prediction

# for altair
import numpy as np
import pandas as pd
import altair as alt
import requests
import folium
import matplotlib.pyplot as plt

app = Flask(__name__)

# @app.route('/')
# def index():
#     return 'Index Page'

# @app.route('/hello')
# def hello():
#     return 'Hello, World'

# @app.route('/hello/')
# @app.route('/hello/<name>')
# def hello(name=None):
#     print (name)
#     return render_template('hello.html', name=name)

@app.route('/')
def plot():
    return render_template('hello.html')

@app.route('/', methods=['POST'])
def make_plot():
    text = request.form['search']
    processed_text = text.upper()

    # return render_template('hello.html', input_text=processed_text)
    # n = int(request.args.get('search', 100))
    # x = np.random.random(n) * 10
    # y = np.random.random(n) * 10
    # s = np.random.random(n)
    
    x = [1,2,3,4,5,6,7,8,9,10,11,12]
    # y = make_prediction(processed_text)
    y, rating = get_company_prediction(processed_text)
    # [63.185500,64.134474,64.753913,66.238947,68.828182,70.561364,71.843250,72.715652,74.365500,77.889091,83.645238,84.836000]

    if not y:
        return render_template('hello.html', json1=None, input_text=('Symbol: ' + processed_text), rating_text='Lack of company social media information!')
    else:
        df = pd.DataFrame({
            'Month': x,
            'Price': y
        })

        max_y = df['Price'].max()
        min_y = df['Price'].min()

        chart = alt.Chart(df, width=300,
                          height=200).mark_line().encode(
                              alt.X('Month'),
                              alt.Y('Price',scale=alt.Scale(domain=(min_y,max_y)))
                          )
        # ).interactive()

        json = chart.to_json()

        return render_template('hello.html', json1=json, input_text=('Symbol: ' + processed_text), rating_text=('Rating: ' + rating ) )
