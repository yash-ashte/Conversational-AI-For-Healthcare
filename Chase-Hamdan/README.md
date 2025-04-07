# 🌸 Conversational Iris Dataset Chatbot

An interactive chatbot that answers questions about the Iris dataset using a trained classifier and SHAP-based interpretability. Built with a React frontend and Python backend, powered by the OpenAI GPT API.

## 🛠 Project Structure

```
Chatbot_Backend/    # Python server and ML logic
Chatbot_Frontend/   # React frontend
```

---

## 🔑 Setup: OpenAI GPT API Key

1. Navigate to the backend `keys` directory:

   ```bash
   cd Chatbot_Backend/keys
   ```

2. Create a file named `gpt.key` and paste your OpenAI API key into it:

   ```bash
   echo "sk-..." > gpt.key
   ```

---

## 🚀 Running the Frontend

1. Navigate to the frontend directory:

   ```bash
   cd Chatbot_Frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

---

## 🧠 Running the Backend

1. Navigate to the backend directory:

   ```bash
   cd Chatbot_Backend
   ```

2. Install Python dependencies (if not already installed):

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python server:

   ```bash
   python3 server.py
   ```

---

## ✅ Features

- Natural language question handling with GPT
- Structured output formats based on question types
- SHAP explanations for model predictions
- Lightweight frontend with real-time chat

---

## 📌 Requirements

- Node.js (for frontend)
- Python 3.8+ (for backend)
- OpenAI API key
- Required Python packages listed in `requirements.txt`