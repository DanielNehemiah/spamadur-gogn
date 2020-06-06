from django.forms import ModelForm
from .models import Detection

class DetectionForm(ModelForm):
    class Meta:
        model = Detection
        fields = ['name', 'model_name', 'input_file']