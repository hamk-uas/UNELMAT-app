from django.db import models
import os

def upload_to(instance, filename):
    folder_name = instance.folder.name.replace(" ", "_")
    return os.path.join('folders', folder_name, filename)

class Folder(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def get_original_image_name(self):
        # Assumes the first image in the folder is the original
        original_image = self.images.first()
        if original_image:
            return os.path.splitext(os.path.basename(original_image.image.name))[0]
        return self.name  # fallback to folder name if no image found

    def __str__(self):
        return self.name

class Image(models.Model):
    folder = models.ForeignKey(Folder, related_name='images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to=upload_to)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.folder.name} - {os.path.basename(self.image.name)}"

class ProcessedImage(models.Model):
    original_image = models.ForeignKey(Image, related_name='processed_images', on_delete=models.CASCADE)
    contour_image = models.ImageField(upload_to='folders/contours/', null=True, blank=True)
    region_image = models.ImageField(upload_to='folders/regions/', null=True, blank=True)
    rectangles_image = models.ImageField(upload_to='folders/rectangles/', null=True, blank=True)
    removal_image = models.ImageField(upload_to='folders/removals/', null=True, blank=True)
    detected_blisters_image = models.ImageField(upload_to='folders/detection/', null=True, blank=True)
    grade = models.CharField(max_length=50, null=True, blank=True)
    processed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Processed Image for {os.path.basename(self.original_image.image.name)}"
