from django.db import models
from app.models import Folder

class MLResult(models.Model):
    folder = models.ForeignKey(Folder, on_delete=models.CASCADE)
    image_name = models.CharField(max_length=255)
    result_image = models.ImageField(upload_to='ml_results/')
    num_detections = models.IntegerField()
    best_match_standard_image = models.CharField(max_length=255)
    best_match_similarity_score = models.FloatField()
    best_match_grade = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.image_name} - Grade: {self.best_match_grade}"