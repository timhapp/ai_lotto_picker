import requests
from django.core.management.base import BaseCommand

from draws.models import Drawing

from sodapy import Socrata
import requests
import json

import datetime
from datetime import timedelta
from datetime import datetime
import pytz

class Command(BaseCommand):
    help = 'Fetch data from the external API and populate the database'

    def handle(self, *args, **kwargs):
        client = Socrata("data.ny.gov", None)
        results = client.get("8vkr-v8vh", limit=3, order="draw_date DESC", offset=0)
        draws_data = results
        #print(results)

        for draw in draws_data:
            date_obj = datetime.strptime(draw['draw_date'], '%Y-%m-%dT%H:%M:%S.%f')
            number_POS = draw['winning_numbers'].split()

            #get weather data
            response = requests.request("GET", "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/250%20Marriott%20Dr%2C%20Ste%20250%2C%20Tallahassee%2C%20FL/" + str(datetime.date(date_obj)) + "/" + str(datetime.date(date_obj)) + "?unitGroup=us&maxDistance=2000&elements=datetime%2Ctemp%2Cfeelslike%2Cdew%2Chumidity%2Cprecip%2Cprecipcover%2Cwindspeedmean%2Cwinddir%2Cpressure%2Ccloudcover%2Cmoonphase%2Csource&include=days%2Cobs%2Cremote&key=PBNL7ZQAHHHJNZ3KXGKZ56G4J&options=stnslevel1&contentType=json")
            weather_data = response.json()

            Drawing.objects.update_or_create(
                draw_date = str(datetime.date(date_obj)),
                draw_night = str(datetime.date(date_obj).weekday()),
                pos1 = number_POS[0],
                pos2 = number_POS[1],
                pos3 = number_POS[2],
                pos4 = number_POS[3],
                pos5 = number_POS[4],
                pwrb = number_POS[5],
                Mx = draw['multiplier'],
                mega = 0,
                Barometer = weather_data['days'][0]['pressure'],
                Dewpoint = weather_data['days'][0]['dew'],
                Heat_Index = weather_data['days'][0]['feelslike'],
                Hygrometer = weather_data['days'][0]['humidity'],
                Thermometer = weather_data['days'][0]['temp'],
                Percipitation = weather_data['days'][0]['precip'],
                WindDirection = weather_data['days'][0]['winddir'],
                CloudCover = weather_data['days'][0]['cloudcover'],
                WindSpeed = weather_data['days'][0]['windspeedmean'],
                MoonPhase = weather_data['days'][0]['moonphase'],
                Station = weather_data['days'][0]['source'],
            )
            
            print(str(datetime.date(date_obj)), draw['winning_numbers'])
