import io
from datetime import datetime, timedelta
import keras
from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
import uuid
import numpy as np
from scipy.stats import kurtosis
from sklearn import preprocessing
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import pandas as pd
import tensorflow as tf
from functools import wraps
from utils import data_generator, data_life_generator

app = Flask(__name__)

app.config['SECRET_KEY'] = 'thisissecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///todo.db'

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(50))
    password = db.Column(db.String(80))
    admin = db.Column(db.Boolean)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']

        if not token:
            return jsonify({'message' : 'Token is missing!'}), 401

        try: 
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = User.query.filter_by(public_id=data['public_id']).first()
        except:
            return jsonify({'message' : 'Token is invalid!'}), 401

        return f(current_user, *args, **kwargs)

    return decorated

@app.route('/user', methods=['GET'])
@token_required
def get_all_users(current_user):

    if not current_user.admin:
        return jsonify({'message' : 'Cannot perform that function!'})

    users = User.query.all()

    output = []

    for user in users:
        user_data = {}
        user_data['public_id'] = user.public_id
        user_data['name'] = user.name
        user_data['password'] = user.password
        user_data['admin'] = user.admin
        output.append(user_data)

    return jsonify({'users' : output})

@app.route('/user/<public_id>', methods=['GET'])
@token_required
def get_one_user(current_user, public_id):

    if not current_user.admin:
        return jsonify({'message' : 'Cannot perform that function!'})

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({'message' : 'No user found!'})

    user_data = {}
    user_data['public_id'] = user.public_id
    user_data['name'] = user.name
    user_data['password'] = user.password
    user_data['admin'] = user.admin

    return jsonify({'user' : user_data})

@app.route('/user', methods=['POST'])
@token_required
def create_user(current_user):
    if not current_user.admin:
        return jsonify({'message' : 'Cannot perform that function!'})

    data = request.get_json()

    hashed_password = generate_password_hash(data['password'], method='sha256')

    new_user = User(public_id=str(uuid.uuid4()), name=data['name'], password=hashed_password, admin=False)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message' : 'New user created!'})

@app.route('/user/<public_id>', methods=['PUT'])
@token_required
def promote_user(current_user, public_id):
    if not current_user.admin:
        return jsonify({'message' : 'Cannot perform that function!'})

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({'message' : 'No user found!'})

    user.admin = True
    db.session.commit()

    return jsonify({'message' : 'The user has been promoted!'})

@app.route('/user/<public_id>', methods=['DELETE'])
@token_required
def delete_user(current_user, public_id):
    if not current_user.admin:
        return jsonify({'message' : 'Cannot perform that function!'})

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({'message' : 'No user found!'})

    db.session.delete(user)
    db.session.commit()

    return jsonify({'message' : 'The user has been deleted!'})

@app.route('/login')
def login():
    auth = request.authorization

    if not auth or not auth.username or not auth.password:
        return make_response('Could not verify', 401, {'WWW-Authenticate' : 'Basic realm="Login required!"'})

    user = User.query.filter_by(name=auth.username).first()
    if not user:
        return make_response('Could not verify', 401, {'WWW-Authenticate' : 'Basic realm="Login required!"'})

    if check_password_hash(user.password, auth.password):
        token = jwt.encode({'public_id' : user.public_id, 'exp' : datetime.utcnow() + timedelta(minutes=30)}, app.config['SECRET_KEY'])

        return jsonify({'token' : token.decode('UTF-8')})

    return make_response('Could not verify', 401, {'WWW-Authenticate' : 'Basic realm="Login required!"'})


@app.route('/diagnose', methods=['POST'])
@token_required
def diagnose(current_user):
    model = keras.models.load_model("IAI_IMS_final.h5")
    content = request.files['file'].read()
    outputsFile = data_generator(content, 0)
    result = model.predict_classes(outputsFile)
    fault_name = ''
    for i in range(int(len(result) / 20)):
        total = sum(result[i * 20:i * 20 + 20])
        if total <= 5:
            fault_name = 'normal'
        elif total >= 14 and total < 26:
            fault_name = 'inner race failure'
        elif total >= 34 and total < 46:
            fault_name = 'outer race failure'
        elif total >= 54 and total <= 60:
            fault_name = 'roller element failure'
        else:
            fault_name = 'Can\'t predict failure'
    return fault_name

@app.route('/life', methods=['POST'])
@token_required
def life(current_user):
    model = keras.models.load_model("life_predict.h5")
    content = request.files['file'].read()
    header = request.headers['time']
    print(header)
    filename = request.files['file'].filename
    columns = ["B1"]
    df = pd.read_csv(io.BytesIO(content))
    prediction = model.predict(df[int(header)-1:int(header)].drop('time_use', axis=1))[0][0]
    life_percent = prediction * 100
    # life_time = ((100 - prediction * 100) / (prediction * 100)) * df['time_use'][int(header)-1:int(header)]
    life_time = (100*df['time_use'][int(header)-1:int(header)])/(prediction * 100) - df['time_use'][int(header)-1:int(header)]

    print(life_time)
    d = datetime(1, 1, 1) + timedelta(seconds=int(life_time))
    expectancy = "%d days - %d hours - %d minutes - %d seconds" % (d.day - 1, d.hour, d.minute, d.second)

    left_d = datetime(1, 1, 1) + timedelta(seconds=int(df['time_use'][int(header)-1:int(header)]))
    print("Left_d: " + str(left_d))
    left_expectancy = "%d days - %d hours - %d minutes - %d seconds" % (
        left_d.day - 1, left_d.hour, left_d.minute, left_d.second)
    print(expectancy + " " + str(life_percent) + " " + left_expectancy)
    return '{}\n{}\n{}'.format(expectancy, str(life_percent), left_expectancy)

if __name__ == '__main__':
	#host='127.0.0.1', port=3000,
    app.run(debug=True)