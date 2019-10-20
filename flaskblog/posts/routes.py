from flask import (render_template, url_for, flash,
                   redirect, request, abort, Blueprint)
from flask_login import current_user, login_required
from flaskblog import db
from flaskblog.models import Post
from flaskblog.posts.forms import PostForm

#ml joint
#from flask_bootstrap import Bootstrap
import pandas as pd # add to requirements
import numpy as np


#ML Packages
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.externals import joblib


# TIME SERIES PACKGAES

from pandas import DataFrame
from pandas import concat
from pandas import read_csv

import tensorflow as tf # add to requirements

import pandas as pd
import datetime # add to requirements
import pandas_datareader.data as web # add to requirements
from pandas import Series, DataFrame
import pygal # add to requirements
import stripe # add to requirements

pub_key = 'pk_test_D53d7w8AcdROmnyDOo2EfFFx00azCXbW0c'
secret_key = 'sk_test_iRSrpQN5FTQlWu3VPGGZg60H00S3wUa5s4'

stripe.api_key = secret_key


posts = Blueprint('posts', __name__)

@posts.route('/post/pay', methods=['POST'])
def pay():
    print(request.form)
    return ''




@posts.route("/post/new", methods=['GET', 'POST'])
@login_required
def new_post():
    form = PostForm()
    if form.validate_on_submit():
        post = Post(title=form.title.data, content=form.content.data, author=current_user)
        db.session.add(post)
        db.session.commit()
        flash('Your post has been created!', 'success')
        return redirect(url_for('main.home'))
    return render_template('create_post.html', title='Payment',
                           form=form, legend='Payment Plans')


@posts.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post=post)


@posts.route("/post/<int:post_id>/update", methods=['GET', 'POST'])
@login_required
def update_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    form = PostForm()
    if form.validate_on_submit():
        post.title = form.title.data
        post.content = form.content.data
        db.session.commit()
        flash('Your post has been updated!', 'success')
        return redirect(url_for('posts.post', post_id=post.id))
    elif request.method == 'GET':
        form.title.data = post.title
        form.content.data = post.content
    return render_template('create_post.html', title='Update Post', 
                           form=form, legend='Update Post', pub_key=pub_key)


@posts.route("/post/<int:post_id>/delete", methods=['POST'])
@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    if post.author != current_user:
        abort(403)
    db.session.delete(post)
    db.session.commit()
    flash('Your post has been deleted!', 'success')
    return redirect(url_for('main.home'))

#ml joint

@posts.route('/post/set')
def set():
    return render_template('set.html')



@posts.route('/post/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        stock_name = request.form['namequery']
        start = datetime.datetime(2019, 1, 7)
        end = datetime.datetime(2019, 10, 19)

        df = web.DataReader(stock_name, 'yahoo', start, end)
        last_date = df.index[-1]
        day_list = []
        for d in range(5):
            day_list.append(df.index[-1] + datetime.timedelta(days=d+1))


        data = df['Adj Close']
        #read time series dataset assume only one column = "univariate"
        dataset = data


        n_input = 5
        n_nodes = [100, 50, 25, 15, 10]
        n_epochs = 300
        n_batch = 30
        num_step = 5
        n_test = 1

        n_in = 5
        n_out = num_step

        data = df['Adj Close'].values


        def series_to_supervised(data, n_in, n_out=1):
            df = DataFrame(data)
            cols = list()
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
            # put it all together
            agg = concat(cols, axis=1)
            # drop rows with NaN values
            agg.dropna(inplace=True)
            return agg.values



        prepared_data = series_to_supervised(data, n_in, n_out)
        train_x, train_y = prepared_data[:, :-num_step], prepared_data[:, -num_step:]
        X_train, X_test = train_x[:-n_test,:], train_x[-n_test:,:]
        y_train, y_test = train_y[:-n_test], train_y[-n_test:]

        model = tf.keras.Sequential([tf.keras.layers.Dense(n_nodes[0], activation='relu', input_dim=n_input),
                                        tf.keras.layers.Dense(n_nodes[1]),
                                        tf.keras.layers.Dense(n_nodes[2]),
                                        tf.keras.layers.Dense(n_nodes[3]),
                                        tf.keras.layers.Dense(n_nodes[4]),
                                        tf.keras.layers.Dense(num_step)
                                                                    ])
        model.compile(loss='mse', optimizer='adam')
        # fit model
        model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch, verbose=0)
        my_predictions = model.predict(X_test)
        starting_year = str(day_list[0].year)
        starting_month = str(day_list[0].month)
        starting_day = str(day_list[0].day)
        first_day = starting_month+'-'+starting_day+'-'+starting_year

        graph = pygal.Line(x_label_rotation=20)
        graph.title = '5-Day Prediction'
        graph.x_labels = day_list
        graph.add(stock_name,  my_predictions[0])
        graph_data = graph.render_data_uri()







    return render_template('results.html', graph_data = graph_data, prediction1 = my_predictions[0][0],
                            prediction2 = my_predictions[0][1], prediction3 = my_predictions[0][2],
                            prediction4 = my_predictions[0][3], prediction5 = my_predictions[0][4],
                            starting_day = first_day, stock_name = stock_name ) #, name = namequery.upper())