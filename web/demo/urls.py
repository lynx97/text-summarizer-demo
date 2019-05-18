from django.conf.urls import url 
from django.urls import path
from . import views 
app_name = "demo"
urlpatterns = [ 
    url(r'^$', views.index, name='index'),
    path('summarize/', views.summarize, name='summarize'),
]