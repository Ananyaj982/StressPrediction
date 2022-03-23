from flask import Flask,jsonify
from app import app
from models import User
@app.route('/user/signup',methods=['GET'])
def signup1():
    #user=User()
    #return user.signup()
    return User().signup()
