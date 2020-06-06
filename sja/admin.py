from django.contrib import admin

# Register your models here.
from .models import Detection, Image

admin.site.register(Detection)
admin.site.register(Image)