from firebase.firebase import FirebaseApplication
from firebase.firebase import FirebaseAuthentication
import time
import random

app = FirebaseApplication('https://stress-1c46d-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

def generate_1(name):
    new_user ='/'+name+'/'
    while(1):
                randomnumber = random.randint(68, 91)
                randomnumber1 = round(random.uniform(60.00,99.455),2)
                print(randomnumber,randomnumber1)
                result = app.put(new_user,'HR',randomnumber)
                result = app.put(new_user,'SpO2',randomnumber1)
                time.sleep(7)           
 
