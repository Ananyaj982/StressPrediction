import threading
from datetime import datetime
import os
import logistic
import cnn1
import cnn2
from firebase.firebase import FirebaseApplication
from firebase.firebase import FirebaseAuthentication
import time
import random
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
firebase = FirebaseApplication('https://stress-1c46d-default-rtdb.asia-southeast1.firebasedatabase.app/', None)
users = []

def check(username):
    url1 ='/'+username+'/HR'
    url2 ='/'+username+'/SpO2'
    url3='/'+username+'/Subject_ID'
    hr = firebase.get(url1,None)
    sp = firebase.get(url2,None)
    Subject_ID= firebase.get(url3,None)
    value = [[Subject_ID,hr,sp]]
    result = logistic.pred(value)
    res = int(result[0][0][0])
    print("HRV res is",res)
    if res == 0:
        #print("Result is ",result)
        prob = result[0][1][0][0]
        print(prob)
        #level = 0
    else:
        #print("Result is ",result)
        prob = result[0][1][0][1]
        #print(prob)
    url='https://stress-1c46d-default-rtdb.asia-southeast1.firebasedatabase.app/'+username+'/'
    time.sleep(10)
    url2 = url + '/Res2'
    url3 = url + '/Level_cnn'
    resu2 = firebase.get(url2,None)
    print('Result of CNN is ',resu2)
    level2 = float(firebase.get(url3,None))
    if resu2 == -1:
        prob_fin = prob
        result = res
    else:
        if resu2 == res:
            result = resu2
            prob_fin = (0.5 * prob)+(0.5 * level2)
        else:
            if res == 1:
                prob_fin = (0.5* prob) + (0.5*(1-level2))
                result = res
            elif resu2 == 1:
                prob_fin = (0.5* (1 - prob)) + (0.5*level2)
                result = resu2
    print('The HR is :', hr)
    print('The SpO2 is : ', sp)
    print("The combined probabilty is: ", prob_fin)
    print('Result is ',result)
    if result == 0:
        level = 0
    else:
        if ((prob_fin >= 0.5) and (prob_fin < 0.6)):
            level = 1
        elif ((prob_fin >= 0.6) and (prob_fin < 0.7)):
            level = 2
        elif ((prob_fin >= 0.7) and (prob_fin < 0.8)):
            level = 3
        elif ((prob_fin >= 0.8) and (prob_fin < 0.9)):
            level = 4
        else:
            level = 5
    rest = firebase.put(url,'Comb',str(prob_fin))
    rest = firebase.put(url,'Res1',res)
    rest = firebase.put(url,'Level',level)
    rest = firebase.put(url,'Result',int(result))

def check1(username):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    url='https://stress-1c46d-default-rtdb.asia-southeast1.firebasedatabase.app/'+username+'/'
    res1=cnn1.main(username)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    if(res1==-1):
        time.sleep(10)
        firebase.put(url,'Res2',(-1))
        firebase.put(url,'Level_cnn',(0))
    elif(len(res1)>0):
        for i in res1:
            print(i)
            k=i[1]['Predicted Values'].to_string()
            if("not stressed" in k):
                print('K[0][1] is :',i[0][0][0])
                print("not stressed")
                #level3 = i[0][0][0]
                firebase.put(url,'Level_cnn',str(i[0][0][0]))
                resu = firebase.put(url,'Res2',(0))

            else:
                print("stressed")
                print('K[0][1] is :',i[0][0][1])
                level2 = i[0][0][1]
                resu = firebase.put(url,'Level_cnn',str(level2))
                resu = firebase.put(url,'Res2',(1))

def main():

    while(1):
        user_name = firebase.get('/session',None)
        print(users, user_name)
        if user_name not in users:
            print(user_name,'is added')
            users.append(user_name)
            #Thread(target=get_files, args=(user_name,)).start()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        for user_name in users:
            t1 = threading.Thread(target=check,args=(user_name,))
            t2 = threading.Thread(target=check1,args=(user_name,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

if __name__ == '__main__':
    #freeze_support()
    main()
