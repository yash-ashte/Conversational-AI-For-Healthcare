import gradio as gr
from functools import cache
import requests, json, io, base64
import re
from gtts import gTTS
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import whisper
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report
)

whisper_model = whisper.load_model("medium")

@cache
def load_and_prepare_data(path="hospital_dataset.csv"):
    df = pd.read_csv(path)
    df.drop(['Case ID','primary icd','secondary diag code','dept OU'], axis=1, inplace=True)
    df.dropna(subset=['Readm Indicator'], inplace=True)
    cat_cols = ['gender','class','adm source','dis location','specialty','year','month','day of week']
    num_cols = ['LOS','age','num of transfers','Charlson','vanWalraven','Time to Readmission']
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder().fit(df[c].astype(str))
        df[c] = le.transform(df[c].astype(str))
        encoders[c] = le
    raw = df.copy()
    scaler = StandardScaler().fit(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    X = df.drop('Readm Indicator', axis=1)
    y = df['Readm Indicator']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    explainer = shap.Explainer(model, X_test)
    shap_vals = explainer(X)
    return raw, X, X_test, y_test, model, encoders, scaler, {
        'acc': acc, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc
    }, shap_vals

raw_data, X, X_test, y_test, model, encoders, scaler, metrics, shap_vals = load_and_prepare_data()

def get_model_accuracy(): return f"Model accuracy: {metrics['acc']:.2%}"
def get_roc_auc():
    fig, ax = plt.subplots()
    ax.plot(metrics['fpr'], metrics['tpr'], label=f"AUC = {metrics['roc_auc']:.2f}")
    ax.plot([0,1],[0,1],'--', color='gray')
    ax.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend()
    return fig
def get_confusion_matrix():
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    return fig
def get_classification_report(): return classification_report(y_test, model.predict(X_test))
def get_precision_score():     return f"Precision: {precision_score(y_test, model.predict(X_test)):.2f}"
def get_recall_score():        return f"Recall: {recall_score(y_test, model.predict(X_test)):.2f}"
def get_f1_score():            return f"F1 Score: {f1_score(y_test, model.predict(X_test)):.2f}"
def get_feature_importance():
    fig = plt.figure()
    shap.plots.bar(shap_vals, show=False)
    return fig

def explain_instance(arg):
    try:
        idx = int(re.search(r'\d+', arg).group()) if arg else 0
        fig = plt.figure()
        shap.plots.waterfall(shap_vals[idx], show=False)
        return fig
    except (ValueError, AttributeError):
        return "Error: Please specify a valid numeric index (e.g. 'explain instance 2')"

def get_shap_value(arg):
    try:
        idx = int(re.search(r'\d+', arg).group()) if arg else 0
        return dict(zip(X.columns, shap_vals[idx].values))
    except (ValueError, AttributeError):
        return "Error: Please specify a valid numeric index (e.g. 'get shap value 5')"

# Other functions can be added as needed..

keyword_map = {
    "accuracy":            (get_model_accuracy, False),
    "roc auc":             (get_roc_auc, False),
    "confusion matrix":    (get_confusion_matrix, False),
    "classification report": (get_classification_report, False),
    "precision":           (get_precision_score, False),
    "recall":              (get_recall_score, False),
    "f1 score":            (get_f1_score, False),
    "feature importance":  (get_feature_importance, False),
    "explain instance":    (explain_instance, True),
    "get shap value":      (get_shap_value, True),
}

LLM_API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME  = "meta-llama/Llama-3.2-1B-Instruct"

def call_llama(messages):
    resp = requests.post(
        LLM_API_URL,
        headers={"Content-Type": "application/json"},
        json={"model": MODEL_NAME, "messages": messages},
        timeout=20
    )
    return resp.json()["choices"][0]["message"]["content"]

def text_to_speech(text):
    """Convert text to speech and return temp file path"""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tts.save(tmp.name)
        return tmp.name

def transcribe_and_fill(audio_path):
    if audio_path is None:
        return ""
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Transcription error: {str(e)}"

def messages_to_gradio_tuples(messages):
    tuples = []
    user_msg = None
    for msg in messages:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg is not None:
            tuples.append((user_msg, msg["content"]))
            user_msg = None
    return tuples

def process_and_respond(user_input, messages):
    if not messages or len(messages) == 0:
        messages = [{"role": "system", "content": "You are an assistant. Use functions when needed, then based on the returned values, provide a short, clear and concise answer. Just give the user what they're looking for, don't output unnecessary information."}]

    messages.append({"role": "user", "content": user_input})

    audio_output = None
    image_output = None  # For storing base64 image

    for key, (fn, needs_arg) in keyword_map.items():
        if key in user_input.lower():
            arg = user_input.lower().split(key,1)[1].strip() if needs_arg else None
            result = fn(arg) if needs_arg else fn()

            # Handle matplotlib figures
            if isinstance(result, plt.Figure):
                buf = io.BytesIO()
                result.savefig(buf, format='png')
                buf.seek(0)
                image_output = base64.b64encode(buf.getvalue()).decode('utf-8')
                result = f"<img src='data:image/png;base64,{image_output}'>"
                plt.close(result)  # Clean up figure

            messages.append({"role": "function", "name": fn.__name__, "content": json.dumps(result, default=str)})
            reply = call_llama(messages)

            # Append image to reply if exists
            if image_output:
                reply += f"\n![plot](data:image/png;base64,{image_output})"

            messages.append({"role": "assistant", "content": reply})

            if reply.strip():
                audio_output = text_to_speech(reply)
            break
    else:
        reply = call_llama(messages)
        messages.append({"role": "assistant", "content": reply})
        if reply.strip():
            audio_output = text_to_speech(reply)

    gradio_history = messages_to_gradio_tuples(messages)
    return gradio_history, messages, audio_output

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("Hospital Chatbot", elem_id="header")
    chatbot = gr.Chatbot(label="Chat", elem_id="chatbot", height=450, render_markdown=True)
    state = gr.State([])
    audio_output = gr.Audio(label="Assistant Voice", type="filepath")

    with gr.Row():
        text_input = gr.Textbox(
            show_label=False,
            placeholder="Type or speak your message...",
            elem_id="input_box",
            scale=8
        )
        send_btn = gr.Button("➤", elem_id="send_btn", scale=1)
        mic_btn = gr.Audio(
            sources=["microphone"],
            type="filepath",
            show_label=False,
            elem_id="mic_btn",
            scale=1
        )

    send_btn.click(
        fn=process_and_respond,
        inputs=[text_input, state],
        outputs=[chatbot, state, audio_output]
    ).then(
        lambda: "", None, text_input
    )

    mic_btn.stop_recording(
        fn=transcribe_and_fill,
        inputs=mic_btn,
        outputs=text_input
    )

if __name__ == "__main__":
    demo.launch(share=True)
