from django.http import HttpResponse
from django.template import loader

from .models import Drawing, WeatherDrawing

def index(request):
    latest_draw_data = Drawing.objects.order_by("-draw_date")[:1]
    template = loader.get_template("draws/index.html")
    context = {
        "latest_draw_data": latest_draw_data,
    }
    return HttpResponse(template.render(context, request))

def prediction(request):
    # Show next drawing for date with forecast and prediction
    latest_prediction_data = WeatherDrawing.objects.order_by("-draw_date")[:1]
    template = loader.get_template("draws/index.html")
    context = {
        "latest_draw_data": latest_prediction_data,
    }
    return HttpResponse(template.render(context, request))

def history(request, draw_date):
    response = "Show most recent actual vs prediction. Search for results of draw %s."
    return HttpResponse(response % draw_date)


