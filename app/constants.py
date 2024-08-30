# constants.py
import os
from django.conf import settings

# Base directories
MEDIA_ROOT = settings.MEDIA_ROOT
FOLDERS_DIR = os.path.join(MEDIA_ROOT, 'folders')
STANDARD_IMAGES_DIR = os.path.join(MEDIA_ROOT, 'standard_images')

# Processing directories
PROCESSED_DIR = 'processed'
DETECTION_DIR = 'detection'

# Subdirectories
CONTOURS_DIR = 'contours'
REGIONS_DIR = 'regions'
RECTANGLES_DIR = 'rectangles'
REMOVALS_DIR = 'removals'
COMBINED_BLISTERS_DIR = 'combined_blisters'
LARGE_BLISTERS_DIR = 'large_blisters'
SMALL_BLISTERS_DIR = 'small_blisters'

# Template directory
XL_TEMPLATES_DIR = os.path.join(MEDIA_ROOT, 'xl_templates')
