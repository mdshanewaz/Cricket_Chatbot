from django.shortcuts import render
from langchain import document_loaders

# Create your views here.

def load_cricket_knowledge(docx_path):
    return document_loaders(docx_path)

cricket_knowledge = load_cricket_knowledge("Cricket.docx")

