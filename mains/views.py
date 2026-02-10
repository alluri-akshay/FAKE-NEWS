from django.shortcuts import render

# Create your views here.
def about(req):
    return render(req,'about.html')

def deals(req):
    return render(req,'deals.html')

def index(req):
    return render(req,'index.html')

def reservation(req):
    return render(req,'reservation.html')
