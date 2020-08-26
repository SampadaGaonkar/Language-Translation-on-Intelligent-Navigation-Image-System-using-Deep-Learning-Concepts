
from django.conf.urls import  include,url
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    url(r"^admin/", admin.site.urls),
    url(r"^Test/", include("Test.urls")),
    url(r"^Beproject/", include("Beproject.urls")),

]

#urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 