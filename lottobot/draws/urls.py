from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("prediction/", views.prediction, name="prediction"),
    path("<draw_date>/history/", views.history, name="history"),
]
