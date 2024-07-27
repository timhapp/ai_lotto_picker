from django.contrib import admin

from .models import Drawing, WeatherDrawing

class DrawingAdmin(admin.ModelAdmin):
    list_display = ('draw_date', 'draw_night', 'pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'pwrb', 'mega', 'Mx')

admin.site.register(Drawing, DrawingAdmin)
admin.site.register(WeatherDrawing, DrawingAdmin)
