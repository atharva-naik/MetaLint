#all the imports needed
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from .forms import LoginForm, SignupForm
from django.contrib.auth.models import User
from django.contrib import messages
import re
# Create your views here.
def login_view(request):
    error = False
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            user = authenticate(username=username, password=password)  # We check wither the data are correct
            if user:  # If the object returned is not None
                messages.success(request, 'Vous êtes connecté')
                login(request, user)  # We log the user in
            else:
                error = True # Otherwise, an error is displayed
    else:
        form = LoginForm()
    return render(request, 'accounts/login.html', locals())


def signup_view(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            #return redirect('home') Mettre la page d'accueil à la racine
    else:
        form = SignupForm()
    return render(request, 'accounts/signup.html', {'form': form})