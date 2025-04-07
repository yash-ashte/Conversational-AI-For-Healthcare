"use client";  // Ensure this file is treated as a client component

import { useState, useEffect, useRef } from "react";
import axios from 'axios';

export default function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);  // Track if the chatbot is processing the request
  const messagesEndRef = useRef(null);

  // Handle message submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessages([...messages, { text: userInput, sender: "user" }]);
    setIsLoading(true);  // Set loading to true when request is made
    await processChatbotResponse(userInput);
    setUserInput("");
  };

  // Process responses from the chatbot (include plot if requested)
  const processChatbotResponse = async (input) => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/chat', { input: input });

      if (response.data.plots && response.data.plots.length > 0) {
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: response.data.response, sender: "bot", plots: response.data.plots },
        ]);
      } else {
        setMessages((prevMessages) => [
          ...prevMessages,
          { text: response.data.response, sender: "bot" },
        ]);
      }
    } catch (error) {
      console.error("Error connecting to backend:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "Sorry, there was an error connecting to the backend.", sender: "bot" },
      ]);
    } finally {
      setIsLoading(false);  // Set loading to false after the response is processed
    }
  };

  // Scroll to the bottom whenever the messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-container">
      <h1 className="chat-title">Iris Data Classification Chatbot</h1>
      <div className="chatbox">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender === "bot" ? "bot-message" : "user-message"}`}>
            <p dangerouslySetInnerHTML={{ __html: msg.text }} />
            {msg.plots && msg.plots.map((plot, i) => (
              <img
                key={i}
                src={plot}
                alt={`SHAP plot ${i + 1}`}
                style={{ maxWidth: "100%", objectFit: "contain", marginTop: "10px", borderRadius: "8px" }}
              />
            ))}
          </div>
        ))}
        {isLoading && (
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>Loading...</p>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Ask me something..."
          className="input-field"
        />
        <button type="submit" className="send-button">Send</button>
      </form>
      <style jsx global>{`
        html, body {
          margin: 0;
          padding: 0;
          height: 100%;
          width: 100%;
          background-color: #121212;  /* Dark background */
          display: flex;
          flex-direction: column;
        }

        .chat-title {
          text-align: left;
          font-size: 2em;  /* Adjust to your preferred size */
          margin-top: 0px;
          margin-bottom: 5px;
          color: rgb(180, 194, 200);
          font-family: 'Roboto', sans-serif;
          overflow-wrap: break-word; 
        }

        .chat-container {
          display: flex;
          flex-direction: column;
          justify-content: flex-end;  /* Align chatbox to the bottom */
          height: 100%;
          width: 100%;
          padding: 20px;
          box-sizing: border-box;
          color: white;
        }

        .chatbox {
          flex-grow: 1;                  
          overflow-y: auto;             
          margin-bottom: 20px;
          padding: 10px;
          border-radius: 8px;
          background-color: #1e1e1e;
          max-height: 100%;              
          display: flex;
          flex-direction: column;    
          word-wrap: break-word;   
          word-break: break-word; 
          overflow-wrap: break-word;
        }

        /* Scrollbar styling */
        .chatbox::-webkit-scrollbar {
          width: 8px;  /* Width of the scrollbar */
        }

        .chatbox::-webkit-scrollbar-thumb {
          background-color: #4a90e2;  /* Color of the thumb */
          border-radius: 4px;  /* Rounded edges */
        }

        .chatbox::-webkit-scrollbar-track {
          background-color: #333;  /* Color of the track */
          border-radius: 4px;  /* Rounded edges */
        }

        .message {
          margin-bottom: 15px;
          padding: 10px;
          border-radius: 8px;
          max-width: 100%;  /* Make sure the message container doesn't overflow */
          word-wrap: break-word;  /* Ensure text wraps within the message box */
          word-break: break-word;
          overflow-wrap: break-word;
        }

        .bot-message {
          display: inline-block;
          background-color: #333;
          color: #f1f1f1;
          padding: 10px;
          border-radius: 8px;
          margin-bottom: 10px;
          max-width: 100%; /* Set a max width */
          font-family: 'Roboto', sans-serif;
        }

        .user-message {
          display: inline-block;
          padding: 10px;
          background-color: #4a90e2;
          color: #ffffff;
          text-align: right;
          border-radius: 8px;
          min-width: 40%;
          align-self: flex-end;
          font-family: 'Roboto', sans-serif;
        }

        .input-form {
          display: flex;
          justify-content: space-between;
          align-items: center;
          position: relative;
        }

        .input-field {
          width: 85%;
          padding: 12px;
          background-color: #2c2c2c;
          color: white;
          border: none;
          border-radius: 4px;
          box-sizing: border-box;
        }

        .send-button {
          padding: 12px 20px;
          background-color: #4a90e2;
          border: none;
          border-radius: 4px;
          color: white;
          cursor: pointer;
        }

        .send-button:hover {
          background-color: #357ab7;
        }

        /* Loading spinner styles */
        .loading-spinner {
          display: flex;
          align-items: center;
          justify-content: center;
          flex-direction: column;
          margin-top: 20px;
        }

        .spinner {
          border: 4px solid #f3f3f3;  /* Light grey background */
          border-top: 4px solid #3498db;  /* Blue spinner */
          border-radius: 50%;
          width: 50px;
          height: 50px;
          animation: spin 2s linear infinite;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
