import gradio as gr
from groq import Groq
from pypdf import PdfReader
import os

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chat_with_pdf(user_input, file):
    if file is None:
        system_msg = "You are a helpful AI assistant"
    else:
        pdf_text = read_pdf(file)
        system_msg = f"Answer based on this document: {pdf_text[:3000]}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content

gr.Interface(
    fn=chat_with_pdf,
    inputs=[
        gr.Textbox(label="Ask Uzma AI anything..."),
        gr.File(label="Upload PDF (optional)", file_types=[".pdf"])
    ],
    outputs=gr.Textbox(label="Answer"),
    title="AskUzma 🤖",
    description="""
### Your Personal AI Assistant — Built by Uzma Latif
💬 Ask me anything | 📄 Upload a PDF to chat with it
Built with Groq API • LLaMA 3.3 70B • Python • Gradio
"""
).launch(server_name="0.0.0.0", server_port=7860)