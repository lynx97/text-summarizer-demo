from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import JsonResponse
from nltk.tokenize import sent_tokenize
import sent2vec

#library for text_summary
from langdetect import detect
from nltk.tokenize import sent_tokenize
 
def index(request):
    template = loader.get_template('demo/index.html')
    context = {
        
    }
    return HttpResponse(template.render(context, request))

def summarize(request):
    original_text = request.POST.get('original_text','')
    data = {'result': sent_embedding(original_text)}
    return JsonResponse(data)

def preprocessText(text):
    contents_parsed = text.lower() #Biến đổi hết thành chữ thường
    contents_parsed = contents_parsed.replace('\n', '. ') #Đổi các ký tự xuống dòng thành chấm câu
    contents_parsed = contents_parsed.strip() #Loại bỏ đi các khoảng trắng thừa
    return contents_parsed
def lang_detect(text):
    lang = detect(text)
    if(lang == 'en'):
        return 0
    elif(lang == 'vi'):
        return 1
    else:
        return -1
def sent_embedding(text):
    text_pre = preprocessText(text)
    lang = lang_detect(text_pre)
    sentences = sent_tokenize(text_pre)
    model = sent2vec.Sent2vecModel()
    if(lang == 0):
        path = "" #load model cho tieng anh
    elif(lang == 1):
        path = "" #load model cho tieng viet
    else:
        return 0
    model.load_model(path)
    embs = model.embed_sentences(sentences)
    return embs
