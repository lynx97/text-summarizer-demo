from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import JsonResponse
from nltk.tokenize import sent_tokenize
import sent2vec
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from pyvi import ViTokenizer
from gensim.models import Word2Vec

#library for text_summary
from langdetect import detect
from nltk.tokenize import sent_tokenize
 
def index(request):
    template = loader.get_template('demo/index.html')
    context = {}
    return HttpResponse(template.render(context, request))

def summarize(request):
    original_text = request.POST.get('original_text','')
    
    try:
        num_cluster = int(request.POST.get('num_cluster',''))
    except ValueError:
        num_cluster = None
    X, sentences = sent_embedding(original_text)

    if num_cluster == None:
        num_cluster = int(np.sqrt(len(sentences)))

    print("num cluster = ",num_cluster)

    summary_s2v = k_mean_clustering(X, num_cluster, sentences)

    X = sent_embedding_with_w2v(original_text, sentences)
    summary_w2v = k_mean_clustering(X, num_cluster, sentences)

    data = {'result_w2v': summary_w2v, 'result_s2v': summary_s2v}
    return JsonResponse(data)

def preprocessText(text):
    contents_parsed = text.lower() #Biến đổi hết thành chữ thường
    contents_parsed = contents_parsed.replace('.', '. ') 
    contents_parsed = contents_parsed.replace('?', '? ') 
    contents_parsed = contents_parsed.strip() #Loại bỏ đi các khoảng trắng thừa
    return contents_parsed

def sent_embedding(text):
    lang = detect(text)
    text_pre = preprocessText(text)
    sentences = sent_tokenize(text_pre)
    model = sent2vec.Sent2vecModel()
    _os_path = "/home/thangnd/git/python/NLP_20182/text-summarizer-demo/web/models/"
    if(lang == 'en'):
        path = _os_path + "wiki_unigrams.bin" #load model cho tieng anh
    elif(lang == 'vi'):
        path = _os_path + "my_model.bin" #load model cho tieng viet
    else:
        return 0
    model.load_model(path)
    embs = model.embed_sentences(sentences)
    return embs, sentences

def sent_embedding_with_w2v(text, sentences):
    w2v = Word2Vec.load("/home/thangnd/git/python/Vietnamese_doc_summarization_basic/vi/vi.bin")
    vocab = w2v.wv.vocab
    X = []
    for sentence in sentences:
        sentence = ViTokenizer.tokenize(sentence)
        words = sentence.split(" ")
        sentence_vec = np.zeros((100))
        for word in words:
            if word in vocab:
                sentence_vec+=w2v.wv[word]
        X.append(sentence_vec)
    return X

def k_mean_clustering(X, n_clusters, sentences):
    print(n_clusters)
    
    # for s in sentences:
    #     print(s)
    # print(len(sentences))


    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(X)
    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([sentences[closest[idx]] for idx in ordering])

    return summary
