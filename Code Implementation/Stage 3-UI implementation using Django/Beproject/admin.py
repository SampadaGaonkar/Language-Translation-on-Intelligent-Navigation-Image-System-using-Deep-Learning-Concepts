from django.contrib import admin

# Register your models here.
from .models import ImageUploadModel
 
class upload_image_Admin(admin.ModelAdmin):
  list_display = ('document',)

admin.site.register(ImageUploadModel, upload_image_Admin)