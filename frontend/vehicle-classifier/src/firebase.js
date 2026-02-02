// src/firebase.js
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

const firebaseConfig = {
  // Replace with your Firebase configuration
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY || "AIzaSyDemo1234567890_firebase_api_key",
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN || "vehicle-classifier-demo.firebaseapp.com",
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID || "vehicle-classifier-demo",
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET || "vehicle-classifier-demo.appspot.com",
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID || "123456789",
  appId: process.env.REACT_APP_FIREBASE_APP_ID || "1:123456789:web:abcdef123456"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication
export const auth = getAuth(app);

export default app;
