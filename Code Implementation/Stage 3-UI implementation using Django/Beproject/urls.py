from django.conf.urls import url
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
	url(r'^$', views.spanish_language, name='spanish_language'),
	url(r'^spanish_language/$', views.spanish_language, name='spanish_language'),
	url(r'^french_language/$', views.french_language, name='french_language'),
]

 
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 