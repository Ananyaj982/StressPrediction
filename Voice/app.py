from flask import Flask
from flask_cors import CORS
import pymongo
from flask import flash
from flask import Flask,render_template
from flask import Flask, render_template,request,redirect,url_for,session
from firebase.firebase import FirebaseApplication
from firebase.firebase import FirebaseAuthentication
from pymongo import MongoClient
from flask import jsonify
import json
from threading import Thread
import datetime
import val
import time
from flask_executor import Executor
from bson import ObjectId
import certifi
import time
import random


ca = certifi.where()

client = pymongo.MongoClient("mongodb+srv://Useran:useran@cluster0.um20t.mongodb.net/myFirstDatabase?retryWrites=true&w=majority&ssl=true", tlsCAFile=ca)

#client = pymongo.MongoClient("mongodb+srv://Useran:useran@cluster0.um20t.mongodb.net/myFirstDatabase?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE")
db = client.myFirstDatabase
SampleTable = db.SampleTable
#client = pymongo.MongoClient("mongodb+srv://Useran:useran@cluster0.um20t.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
#db = client.myFirstDatabase
#connection_url = "mongodb+srv://Useran:<password>@cluster0.um20t.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
app = Flask(__name__)
app.secret_key = 'your secret key'
firebase = FirebaseApplication('https://stress-1c46d-default-rtdb.asia-southeast1.firebasedatabase.app/', None)
subject_id = firebase.get('/Subject_id',None)

def generate_1(name):
    new_user ='/'+name+'/'
    while(1):
                randomnumber = random.randint(68,100)
                randomnumber1 = round(random.uniform(60.00,99.455),2)
                print(randomnumber,randomnumber1)
                result = firebase.put(new_user,'HR',randomnumber)
                result = firebase.put(new_user,'SpO2',randomnumber1)
                time.sleep(7)

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/register")
def register():
    return render_template('register.html')

@app.route('/insert-one/', methods=['GET', 'POST'])
def insertOne():
    global subject_id
    subject_id = 1000
    fullname = request.form['fullname']
    email = request.form['email']
    phonenumber = request.form['phonenumber']
    username = request.form['username']
    password = request.form['password']
    drive = request.form['drive']

    if(findOne("phonenumber",phonenumber)!="NULL"):
        #flash("That phonenumber is already taken, please choose another")
        error = 'Phone number already exists'
        return render_template('register.html',msg=error)
    if(findOne("username",username)!="NULL"):
        #flash("That username is already taken, please choose another")
        error = 'That username is already taken, please choose another'
        return render_template('register.html',msg=error)

    if(findOne("email",email)!="NULL"):
        #flash("That email is already taken, please choose another")
        error = 'That email is already taken, please choose another'
        return render_template('register.html',msg=error)

    if(findOne("drive",drive)!="NULL"):
        #flash("That email is already taken, please choose another")
        error = 'That drive link is already in use, please choose another'
        return render_template('register.html',msg=error)

    queryObject = {
        'fullname':fullname,
        'email': email,
        'phonenumber':phonenumber,
        'username': username,
        'password': password,
        'subject_id' : subject_id,
        'drive':drive
    }
    query = SampleTable.insert_one(queryObject)
    print(query.acknowledged)
    if query.acknowledged:
        new_user ='/'+username
        result = firebase.put(new_user, 'HR',72)
        result = firebase.put(new_user, 'SpO2', 96)
        result = firebase.put(new_user, 'Res1', 0)
        result = firebase.put(new_user, 'Res2', 0)
        result = firebase.put(new_user, 'Level', 0)
        result = firebase.put(new_user, 'Subject_ID', subject_id)
        result = firebase.put(new_user, 'Drive', drive)
        subject_id  = subject_id + 1
        result = firebase.put('https://stress-1c46d-default-rtdb.asia-southeast1.firebasedatabase.app/','Subject_id',(subject_id))
        return redirect('/login')
    else:
        error = 'Registration not successful. Try again after sometime'
        return redirect('/register', msg = error)


# To find the first document that matches a defined query,
# find_one function is used and the query to match is passed
# as an argument.

#@app.route('/find-one/<argument>/<value>/', methods=['GET'])
def findOne(argument, value):
    queryObject = {argument:value}
    query = SampleTable.find_one(queryObject)
    if query:
        query.pop('_id')
        return jsonify(query)
    else:
        return "NULL"

@app.route('/dashboard', methods = ['GET', 'POST'])
def dash():
    if session['loggedin']==False:
        return redirect('login')
    if 'loggedin' in session:
        executor.submit(get_data)
        executor.submit(del_data)
        executor.submit(gen)
        username = session['username']
    return render_template('dashboard.html', user = username)


@app.route('/login', methods=['GET', 'POST'])
def login_admin():
    error = None
    msg=''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        a=findOne("username",username)
        if a !="NULL":
            msg = 'Logged in successfully !'
            session['loggedin'] = True
            #session['id'] = account['id']
            session['username'] = request.form['username']
            #session['username'] = a['username']
            session['T_ID'] = 0
            result = firebase.put('/','session',username)
            return redirect('/dashboard')
        else:
            msg='Enter Valid Credentials!'
    return render_template('login.html',msg=msg)

# To find all the entries/documents in a table/collection,
# find() function is used. If you want to find all the documents
# that matches a certain query, you can pass a queryObject as an
# argument.
@app.route('/find/', methods=['GET'])
def findAll():
    query = SampleTable.find()
    output = {}
    i = 0
    for x in query:
        output[i] = x

        output[i].pop('_id')
        i += 1
    return jsonify(output)


# To update a document in a collection, update_one()
# function is used. The queryObject to find the document is passed as
# the first argument, the corresponding updateObject is passed as the
# second argument under the '$set' index.
@app.route('/update/<key>/<value>/<element>/<updateValue>/', methods=['GET'])
def update(key, value, element, updateValue):
    queryObject = {key: value}
    updateObject = {element: updateValue}
    query = SampleTable.update_one(queryObject, {'$set': updateObject})
    if query.acknowledged:
        return "Update Successful"
    else:
        return "Update Unsuccessful"

#@app.route('/get', methods=['GET'])
def get_data():
    while(1):
        if session['loggedin']==False:
            return redirect('login')
        if 'loggedin' in session:
            username = session['username']
            #userna = db.session['username']
            url = '/'+username+'/'
            hr = firebase.get(url,None)
            userna = db[username]
            hr["timestamp"] = datetime.datetime.utcnow()
            print(hr)
            query = userna.insert_one(hr) #Uncomment these!!!
            #print(query.acknowledged)
            time.sleep(60)
    return redirect('/dashboard')

#@app.route('/display', methods=['GET'])
def del_data():
    while(1):
        if session['loggedin']==False:
            return redirect('login')
        if 'loggedin' in session:
            username = session['username']
            userna = db[username]
            for doc1 in userna.find().sort("timestamp"):
                X = datetime.datetime.utcnow()
                Y = X - doc1["timestamp"]
                days = datetime.timedelta(31)
                sub1 = days - Y
                if sub1.days < 0:
                    print('Deleting: ', doc1)
                    userna.delete_one(doc1)
    return redirect('/dashboard')

def gen():
    if session['loggedin']==False:
        return redirect('login')
    if 'loggedin' in session:
        username = session['username']
        generate_1(username)

@app.route('/display', methods=['GET','POST'])
def display():
    if session['loggedin']==False:
        return redirect('login')
    if 'loggedin' in session:
        select = request.form.get('time')
        try:
            days1 = int(select)
        except:
            days1 =1
        username = session['username']
        userna = db[username]
        output = {}
        data = []
        i = 0
        for doc1 in userna.find().sort("timestamp"):
            X = datetime.datetime.utcnow()
            Y = X - doc1["timestamp"]
            days = datetime.timedelta(days1)
            sub1 = days - Y
            print(sub1, sub1.days)
            sub = str(sub1)
            if (sub1.days >= 0):# ((int(sub[0]) >0)):
                #print(doc1['_id'])
                output[str(i)] = doc1
                output[str(i)].pop('_id')
                output[str(i)]['timestamp'] = output[str(i)]['timestamp'].strftime('%B %d %Y - %H:%M:%S')
                print(output[str(i)]['timestamp'])
                i += 1
        #print(output)
        msg = jsonify(output)
        return render_template('display.html',msg=output)

@app.route("/logout")
def logout():
    session['loggedin']=False
    session['username']=""
    return render_template('home.html')


if __name__ == '__main__':
    executor = Executor(app)
    #app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)
