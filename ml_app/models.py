from django.db import models
from app.models import Folder

class MLResult(models.Model):
    folder = models.ForeignKey(Folder, on_delete=models.CASCADE, null=True)
    image_name = models.CharField(max_length=255, default='')
    result_image = models.ImageField(upload_to='ml_results/', null=True, blank=True)
    result_image_path = models.CharField(max_length=255, null=True, blank=True)  
    num_detections = models.IntegerField(default=0)
    best_match_standard_image = models.CharField(max_length=255, null=True, blank=True)
    best_match_similarity_score = models.FloatField(default=0.0)
    similarity_score = models.FloatField(default=0.0)  
    best_match_grade = models.CharField(max_length=50, default='N/A', null=True)
    grade = models.CharField(max_length=50, default='N/A', null=True)  
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.image_name} - Grade: {self.best_match_grade}"