from Google import Create_Service
import pandas as pd
import requests
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.http import MediaFileUpload
from googleapiclient import errors
from google.oauth2 import service_account
from mimetypes import MimeTypes
import os
import shutil
import io
from io import BytesIO
from firebase.firebase import FirebaseApplication
from firebase.firebase import FirebaseAuthentication
from threading import Thread
from multiprocessing import Process
import time

firebase = FirebaseApplication('https://stress-1c46d-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

CLIENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']
service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
users = []


def url_to_id(url):
    x = url.split("/")
    try:
        return (x[5].split("?")[0])
    except:
        return (x[5])

def get_files(username):
    while(1):
        url = 'https://drive.google.com/drive/folders/1TUgm9v9u-LfU9UIHbrvWWsHal0EA3Aye?usp=sharing'
        url1 ='/'+username+'/Drive'
        url = firebase.get(url1,None)
        #print('Hellooo')
        folder_id =  url_to_id(url)
        query = f"parents = '{folder_id}'"
        response = service.files().list(q=query, fields="nextPageToken,files(id,name, createdTime, modifiedTime, mimeType)").execute()
        files = response.get('files')
        nextPageToken = response.get('nextPageToken')

        while nextPageToken:
            response = service.files().list(q =query, pageToken = nextPageToken, fields="nextPageToken,files(id,name, createdTime, modifiedTime, mimeType)").execute()
            files.extend(response.get('nextPageToken'))
            nextPageToken = response.get('nextPageToken')

        df = pd.DataFrame(files)
        df.sort_values(by=['createdTime'],ascending = False)
        #print(df)
        audiop1="D://sem6//capstone//stuff1//check1"+"//"+username
        try:
            os.mkdir(audiop1)
        except:
            print()
        os.chdir(audiop1)
        for i in range(len(df)) :
            #print(df.loc[i, "name"], df.loc[i,"id"])
            file_name = df.loc[i, "name"]
            if 'used' in file_name:
                print('Skipping file')
                continue
            if '.wav' not in file_name:
                print('Skipping file')
                time.sleep(5)
                continue
            file_ids = df.loc[i, "id"]
            request = service.files().get_media(fileId=file_ids)

            fh = io.BytesIO()
            #fh = io.FileIO(file_name, 'wb') # this can be used to write to disk
            downloader = MediaIoBaseDownload(fd=fh, request = request)
            done = False
            try:
                while not done:
                    status, done = downloader.next_chunk()
                fh.seek(0)
                with open(file_name, 'wb') as f:
                    shutil.copyfileobj(fh, f)
                print("File Downloaded")
            except:
                print("Something went wrong.")
            try:
                file = service.files().get(fileId=file_ids).execute()
                contents = service.files().get_media(fileId=file_ids).execute()
                file_name1 =file['name'].split(".")
                mimetype = MimeTypes().guess_type(file_name)[0]
                file_title = file_name1[0]+ '_used_' + "." +file_name1[1]
                file_metadata = { 'name' : file_title, 'description' : 'this is a test' }
                fh  = BytesIO(contents)
                media = MediaIoBaseUpload(fh, mimetype=mimetype)
                updated_file = service.files().update(
                body=file_metadata,
                #uploadType = 'media',
                fileId=file_ids,
                #fields = fileID,
                media_body=media).execute()
                print('Successfully Updated')
            except errors.HttpError as error:
                print('An error occurred ', error)
                #return
        del df
        time.sleep(5)

def main():
    while(1):
        user_name = firebase.get('/session',None)
        print(users, user_name)
        if user_name not in users:
            print(user_name,'is added')
            users.append(user_name)
            Process(target=get_files, args=(user_name,)).start()
            time.sleep(5)
        else:
            #print('Jojo Moyes')
            time.sleep(5)


if __name__ == '__main__':
    main()
