import datetime, os

from django.core.validators import FileExtensionValidator

from django.db import models
from django.forms import ModelForm
from django.utils import timezone

from django.db.models.signals import post_delete
from django.dispatch import receiver

def defect_upload_location(instance, filename):
	return "%s/%s" %(instance.id, filename)

def image_upload_location(instance, filename):
	return "%s/%s" %(instance.detection.id, filename)

MODEL_CHOICES = (
    ('fasterrcnn_resnet50_fpn_cocotrained','Faster R-CNN ResNet 50 FPN Model Trained on COCO dataset'),
)

class Detection(models.Model):
    name = models.CharField("Detection Name",max_length=200)
    model_name = models.CharField(max_length=100, choices=MODEL_CHOICES, default="fasterrcnn_resnet50_fpn_cocotrained")
    input_file = models.FileField(upload_to=defect_upload_location, validators=[FileExtensionValidator(allowed_extensions=['zip', 'jpg', 'png', 'jpeg'])])
    status = models.CharField(max_length=50, default="In Progress")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class Image(models.Model):
    detection = models.ForeignKey(Detection, on_delete=models.CASCADE)
    image = models.ImageField(upload_to=image_upload_location, height_field = "height", width_field = "width")
    height = models.IntegerField(default=0)
    width = models.IntegerField(default=0)
    detection_bounding_boxes = models.TextField()

    def __str__(self):
        return os.path.basename(self.image.name)


@receiver(models.signals.post_delete, sender=Detection)
def auto_delete_file_on_delete_detection(sender, instance, **kwargs):
    """
    Deletes file from filesystem
    when corresponding `MediaFile` object is deleted.
    """
    if hasattr(instance, 'input_file'):
	    if instance.input_file:
	        if os.path.isfile(instance.input_file.path):
	            os.remove(instance.input_file.path)


@receiver(models.signals.post_delete, sender=Image)
def auto_delete_file_on_delete_image(sender, instance, **kwargs):
    """
    Deletes file from filesystem
    when corresponding `MediaFile` object is deleted.
    """

    if hasattr(instance, 'image'):
	    if instance.image:
	        if os.path.isfile(instance.image.path):
	            os.remove(instance.image.path)