import gradio as gr
from app import pdf_qa_interface

with gr.Blocks(title="PDF Q&A Bot") as demo:
    pdf_file=gr.File(label="Upload file",file_count="multiple",type="filepath")
    query=gr.Textbox(label="ask a question")
    chat_history=gr.State([])

    output=gr.Textbox(label="Answer",lines=4)
    send_btn=gr.Button("Ask")

    send_btn.click(pdf_qa_interface, 
                   inputs=[pdf_file,query,chat_history],
                   outputs=[output,chat_history])
demo.launch()
