from django.db import models

# past draw data + weather data for training predictions
class Drawing(models.Model):
    draw_date = models.CharField(max_length=20, primary_key=True)
    draw_night = models.CharField(max_length=2)
    pos1 = models.CharField(max_length=2)
    pos2 = models.CharField(max_length=2)
    pos3 = models.CharField(max_length=2)
    pos4 = models.CharField(max_length=2)
    pos5 = models.CharField(max_length=2)
    pwrb = models.CharField(max_length=2)
    mega = models.CharField(max_length=2)
    Mx = models.CharField(max_length=2)
    Barometer = models.FloatField()
    Dewpoint = models.FloatField()
    Heat_Index = models.FloatField()
    Hygrometer = models.FloatField()
    Thermometer = models.FloatField()
    Percipitation = models.FloatField()
    WindDirection = models.FloatField()
    CloudCover = models.FloatField()
    WindSpeed = models.FloatField()
    MoonPhase = models.FloatField()
    Station = models.CharField(max_length=20)

    def __str__(self):
        return self.draw_date

# historic record of predictions per draw date
class WeatherDrawing(models.Model):
    draw_date = models.CharField(max_length=20, primary_key=True)
    draw_night = models.CharField(max_length=2)
    pos1 = models.CharField(max_length=2)
    pos2 = models.CharField(max_length=2)
    pos3 = models.CharField(max_length=2)
    pos4 = models.CharField(max_length=2)
    pos5 = models.CharField(max_length=2)
    pwrb = models.CharField(max_length=2)
    mega = models.CharField(max_length=2)
    Mx = models.CharField(max_length=2)
    Barometer = models.FloatField()
    Dewpoint = models.FloatField()
    Heat_Index = models.FloatField()
    Hygrometer = models.FloatField()
    Thermometer = models.FloatField()
    Percipitation = models.FloatField()
    WindDirection = models.FloatField()
    CloudCover = models.FloatField()
    WindSpeed = models.FloatField()
    MoonPhase = models.FloatField()
    Station = models.CharField(max_length=20)

    def __str__(self):
        return self.draw_date

    def get_powerball_draws(self):
        return self.pwrb != 0

    def get_megamillions_draws(self):
        return self.mega != 0


