from django import forms
from .models import Folder, Image

class FolderForm(forms.ModelForm):
    class Meta:
        model = Folder
        fields = ['name']

class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image']
