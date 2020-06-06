from threading import Thread
import zipfile
from django.db import connection
import numpy

from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.forms import modelform_factory
from django.forms import DateTimeInput
from django.core.files.base import ContentFile

from .forms import DetectionForm
from .models import Image, Detection

import torchvision
import torchvision.transforms as T
import cv2, io, os
from PIL import Image as PILImage
import matplotlib.pyplot as plt

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img, threshold):
  # img = Image.open(img_path) # Load the image
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img = transform(img) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return pred_boxes, pred_class


def object_detection_api(img, threshold=0.5, rect_th=3, text_size=3, text_th=3):

  boxes, pred_cls = get_prediction(img, threshold) # Get predictions
  # img = cv2.imread(img_path) # Read image with cv2
  img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
  for i in range(len(boxes)):
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
    cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
  plt.figure(figsize=(20,30)) # display the output image
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  buf = io.BytesIO()
  plt.savefig(buf, format='jpg')
  buf.seek(0)
  return(buf, boxes, pred_cls)

def start_new_thread(function):
    def decorator(*args, **kwargs):
        t = Thread(target = function, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
    return decorator


@start_new_thread
def run_detection(detection):
	input_file_path = detection.input_file.path
	# If it's a zip file
	if input_file_path.split(".")[-1].lower() == 'zip':
		# Read the input file and run detections on it
		archive = zipfile.ZipFile(input_file_path)
		for image_file in [x for x in archive.namelist() if any(z in x for z in ['jpg', 'png', 'jpeg'])]:
			buf, boxes, pred_cls = object_detection_api(PILImage.open(archive.open(image_file)))
			# Save detections in Image model
			detection_image = detection.image_set.create(image = '', detection_bounding_boxes = str((boxes, pred_cls)))
			detection_image.image.save('output_'+image_file,ContentFile(buf.getvalue()))
			detection_image.save()
		detection.status = 'Completed'
	# If it's a single image file
	elif any(x in input_file_path.split(".")[-1].lower() for x in ['jpg', 'png', 'jpeg']):
		# Read the input file and run detections on it
		buf, boxes, pred_cls = object_detection_api(PILImage.open(input_file_path))
		# Save detections in Image model
		detection_image = detection.image_set.create(image = '', detection_bounding_boxes = str((boxes, pred_cls)))
		detection_image.image.save('output_'+os.path.basename(detection.input_file.name), ContentFile(buf.getvalue()))
		detection_image.save()
		detection.status = 'Completed'
	else:
		detection.status = 'Error-Unsupported-File-Uploaded'


	# Change status and save model
	
	detection.save()
	connection.close()

class IndexView(generic.ListView):
    template_name = 'sja/index.html'

    def get_queryset(self):
        """Return the last five published questions."""
        return Detection.objects.all()

class DetailView(generic.DetailView):
    model = Detection
    template_name = 'sja/detail.html'


def createDetection(request):
	form = DetectionForm()
	if request.method == 'POST':
		form = DetectionForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			run_detection(form.instance)
			return HttpResponseRedirect(reverse('sja:detail', args=(form.instance.id,)))
	context = {'form': form}
	return render(request, 'sja/create_detection.html', context)


def delete_detection(request, pk):
	detection= get_object_or_404(Detection, pk=pk)
	if request.method=='POST':
		detection.delete()
		return HttpResponseRedirect(reverse('sja:index'))