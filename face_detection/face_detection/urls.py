"""
URL configuration for face_detection project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from FLDPLDBM.views import signup_view, loginPage, logoutPage, landingPage, delete_embedding, FaceRecognitionAPI, dashboard, dashboard_data,home_page
from django.urls import path



urlpatterns = [
    path('admin/', admin.site.urls),
    path('signup', signup_view, name='signup'), 
    path('', home_page, name='home'),
    path('login/', loginPage, name='login'),
    path('logout/', logoutPage, name='logout'),
    path('landing/', landingPage, name='landing'),
    path('delete_embedding/', delete_embedding, name='delete_embedding'),
    path('recognition/<str:action>/', FaceRecognitionAPI.as_view(), name='recognition-api'),
    path("dashboard/", dashboard, name="dashboard"),
    path("dashboard/data/", dashboard_data, name="dashboard_data"),
]
