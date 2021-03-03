# A Stock Rating App

This stock rating app uses the social media data (LinkedIn, Facebook, etc.) to predict the possible stock movement and make stock ratings in the coming year. It offers a perspective and reference to investors when they are trading stocks. This web app is published at [asrapp.herokuapp.com](http://asrapp.herokuapp.com/).

## Data set
The LinkedIn and Facebook data set were obtained in Gigabyte level CSV fiels. Preprocessing on this data set was done to keep only NASDAQ-100 companies, and the `like_count` and `follower_count` features associated to those companies. The preprocessed data are about ~ 6 MB large, stored in directory _datalab_ in this repo.

The stock data set are weekly historical data pulled from AlphaVantage using it's API, and only those of NASDAQ-100 companies are considered. The stock data set is stored in directory _nasdaq100_ in this repo.

A standalone file to match a company's name across all above data sets is written in `nasdaq100_lk_fb_names.csv`



# STEP
* export FLASK_APP=hello.py
* export FLASK_ENV=development
* flask run
## link
* [1](https://stackoverflow.com/questions/46698134/how-to-post-the-output-result-on-the-same-page-in-flask-app)
* [2](https://stackoverflow.com/questions/37211791/flask-post-to-the-same-page)
* [3](https://flask.palletsprojects.com/en/1.1.x/quickstart/)

