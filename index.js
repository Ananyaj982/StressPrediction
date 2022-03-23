//import { device } from "peer";
//console.log(device.modelName);
//import { app } from "peer";
//console.log(app.readyState);

/*
// Set the configuration for your app
// TODO: Replace with your project's config object
var config = {
apiKey: "apiKey",
authDomain: "projectId.firebaseapp.com",
// For databases not in the us-central1 location, databaseURL will be of the
// form https://[databaseName].[region].firebasedatabase.app.
// For example, https://your-database-123.europe-west1.firebasedatabase.app
databaseURL: "https://databaseName.firebaseio.com",
storageBucket: "bucket.appspot.com"
};
firebase.initializeApp(config);

// Get a reference to the database service
var database = firebase.database();
*/

import { device } from "peer";
console.log(device.modelName);

import { app } from "peer";
console.log(app.readyState);

import * as messaging from "messaging";

const data = { "heartRate": "evt.data"};   

messaging.peerSocket.addEventListener("open", (evt) => {});

messaging.peerSocket.addEventListener("message", (evt) => {
  
    fetch('https://demcap123-default-rtdb.asia-southeast1.firebasedatabase.app/', {
      
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
  
        .then(response => {
            console.log(response);})
  
        .catch(error => {
            console.log(error); });
});