
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
model = AutoModelForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
import PyPDF2

class pdf_qa:
    
    def __init__(self):
        pass
    
    def pdf_text(self,file):
        self.file=file
        # pdfFileObj = open(file, 'rb')
        path = r"C:\Users"
        pdfobj = open(path + "\\"+ file.name,'rb')
        # pdfobj =open(f'{file.name}','rb')
        # pdfobj =open(file,'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfobj)
        print(pdfReader.numPages)
        pageObj = pdfReader.getPage(0)
        text= pageObj.extractText()
        return text
    
    
    def answering(self,question,text):
        self.question=question
        self.text=text
        
        encoding = tokenizer(question, text, return_tensors="pt")
        input_ids = encoding["input_ids"]
        
        attention_mask = encoding["attention_mask"]

        start_scores, end_scores = model(input_ids, attention_mask=attention_mask).values()
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
        answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
        return answer
    

class driver:
    
    def __init__(self):
        pass
    
    def driver_func(self,file,question):
        text_piece=pdf_qa().pdf_text(file)
        ans=pdf_qa().answering(question, text_piece)
        return ans
        


###  Backend Testing 

# file=r"C:\amazon_text_01.pdf"
# question="What is amazon"
# test1=driver().driver_func(file, question)
# test1



### Streamlit Deployment

import streamlit as st

def main_page():
    st.write("# Welcome to QA Model 👋")
    # st.sidebar.markdown("# Upload a document")
    pdf_file=st.file_uploader('Upload the .pdf file', type="pdf")

    st.write("# Ask a Question")
    question=st.text_input("Ask a Question:")
    # text_obj=pdf_qa().pdf_text(pdf_file)

    # st.button("Click to get answer")
    if st.button("Click to get answer"):
        answer=driver().driver_func(pdf_file, question)
        # st.header()
        st.write(answer)
        




page_names_to_funcs = {
    "Upload and Query": main_page,
    
}

selected_page = st.sidebar.selectbox("", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()