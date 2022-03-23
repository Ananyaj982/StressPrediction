from flask import Flask
from flask import jsonify
import pymongo
#from app import db

from passlib.hash import pbkdf2_sha256
import uuid
client1=pymongo.MongoClient('localhost',27017)
db=client1.user_login_system
class User:
    def signup(self):
        print(request.form)
        user={
        "_id":uuid.uuid4().hex,
        "name":request.form.get('name'),
        "email":request.form.get('email'),
        "password":request.form.get('password')
        }
        user['password']=pbkdf2_sha256.hash(user['password'])
        #a=pbkdf2_sha256.encrypt("jsjsj")
        if db.users.find_one({"email":user['email']}):
            return jsonify({"error":"email address already exists"}),400
        cursor = db.users.find()
        for record in cursor:
            print("*ascx",record)
        db.users.insert_one(user)



        #print("inserted $$$$$$$$$$$$$$$$$$$",user)
        #cursor = db.users.find()
        #for record in cursor:
            #print("*ascx",record)
        return jsonify(user),200
        #encrypt the password
