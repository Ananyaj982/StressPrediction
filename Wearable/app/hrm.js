import document from "document";
import { HeartRateSensor } from "heart-rate";
import { display } from "display";
import { me } from "appbit";
/*import {firebase} from 'firebase';

// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  databaseURL: "https://stress-tracker-capstone-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "stress-tracker-capstone",
  storageBucket: "stress-tracker-capstone.appspot.com",
  messagingSenderId: "7714918487",
  appId: "1:7714918487:web:8d64dabf643bc563a09312",
  measurementId: "G-YE5Y0K8P54"
};

// Initialize Firebase
var app = initializeApp(firebaseConfig);
var analytics = getAnalytics(app);
var app = firebase.initializeApp({
  apiKey: "AIzaSyCZrMWy25Uj6RMZWicYUjyq1xTwt_6evRk",
  authDomain: "stress-tracker-capstone.firebaseapp.com",
  databaseURL: "https://stress-tracker-capstone-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "stress-tracker-capstone",
  storageBucket: "stress-tracker-capstone.appspot.com",
  messagingSenderId: "7714918487"
});


var database = firebase.database();

*/

import { peerSocket } from "messaging";
console.log("Max message size=" + peerSocket.MAX_MESSAGE_SIZE);
if (peerSocket.readyState === peerSocket.OPEN) {
   peerSocket.send("Hello");
}

var hrImage = document.getElementById("hrImage");
var hrIcon = document.getElementById("hrIcon");
var hrText = document.getElementById("hrText");

var hrm = null;
var lastMeasuredHR = 0;
var timeOfLastMeasuredHR = 0;
var lastHRMReading = 0;
var hrmActive = false;
var hrTimer = null;

function getHRMReading() {
  let timeToNextReading = 1000;   //check every second even when no HR is detected
  
  let now = new Date().getTime();
  let hr = hrm.heartRate;
  if (hrm.timestamp === lastHRMReading || !hr) {
    if (now - timeOfLastMeasuredHR >= 3000) {
      //more then 3 seconds no new HR reading, watch probably off wrist
      if (hrmActive) {
        //show as not active
        hrmActive = false;
        setHRIconColor();
        showHRMValue("--");
      }
    }
    else {
      //no new reading, but less then 3 seconds ago we still had a valid reading, so keep animating at same speed
      timeToNextReading = 60000/lastMeasuredHR;
    }
  } else {
    //new reading
    if (!hrmActive) {
      hrmActive = true;
      setHRIconColor();
    }

    //store last measured to use when we get no new readings next time
    timeOfLastMeasuredHR = now;
    lastMeasuredHR = hr;
    showHRMValue(lastMeasuredHR);
    timeToNextReading = 60000/lastMeasuredHR;
  }
  lastHRMReading = hrm.timestamp;
  
  //animate when active
  if (hrmActive) {
    hrImage.animate("enable");
  }
  
  //set next reading timeout depending on HR
  if (hrTimer) {
    clearTimeout(hrTimer);
    hrTimer = null;
  }
  hrTimer = setTimeout(getHRMReading, timeToNextReading);
}

function setHRIconColor() {
  if (hrmActive) {
    hrIcon.style.fill = "#FF0000";
  }
  else {
    hrImage.animate("disable");
    hrIcon.style.fill = "#CCCCCC";
  }
}

function showHRMValue(newHRMValue) {
  hrText.text = newHRMValue;
}

/*function writeUserData(heartRate) {
  firebase.database().ref('heartRate/').set({
    heartrate: heartRate
 });
}*/

function startHRMeasurements() {
  if (hrm) {
    if (hrmActive) {
      timeOfLastMeasuredHR = new Date().getTime();    //make sure the icon doesn't show as gray after the screen was off a long time
    }
    hrm.start();
    getHRMReading();
  }
}

function stopHRMeasurements() {
  if (hrTimer) {
    clearTimeout(hrTimer);
    hrTimer = null;
  }
  if (hrm) {
    hrm.stop();
  }
}

export function initialize() {
  hrText.text = '--';
  if (me.permissions.granted("access_heart_rate")) {
    hrm = new HeartRateSensor();
    if (display.on) {
      //already start measurements
      startHRMeasurements();
    }
  }

  //react on display on/off
  display.onchange = function() {
    if (display.on) {
      startHRMeasurements();
    } else {
      stopHRMeasurements();
    }
  }
}