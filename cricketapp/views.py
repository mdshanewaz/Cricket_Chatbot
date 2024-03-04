from django.shortcuts import render
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from langchain import document_loaders

# Create your views here.

def chatbot_view(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            response = get_chatbot_response(user_input)
            return render(request, 'cricket_chatbot/index.html', {'user_input': user_input, 'chatbot_response': response})

    return render(request, 'cricket_chatbot/index.html')


def get_chatbot_response(user_input):
    # Load Cricket knowledge from the document
    cricket_knowledge = document_loaders("Cricket.docx")

    # Initialize RAG model and components
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact_match", use_dummy_dataset=True)
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base", retriever=retriever)

    # Generate chatbot response
    inputs = tokenizer("User: " + user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response
