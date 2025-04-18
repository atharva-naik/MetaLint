from rest_framework import generics
from ..models import Subject
from .serializers import SubjectSerializer,CourseWithContentsSerializer,CourseSerializer
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from ..models import Course
from rest_framework.authentication import BasicAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework import viewsets
from rest_framework.decorators import action
from .permissions import IsEnrolled


class SubjectListView(generics.ListAPIView):
    queryset = Subject.objects.all()      # queryset: The base QuerySet to use to retrieve objects
    serializer_class = SubjectSerializer  # serializer_class: The class to serialize objects
class SubjectDetailView(generics.RetrieveAPIView):
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer

class CourseEnrollView(APIView):
    authentication_classes = (BasicAuthentication,)
    permission_classes = (IsAuthenticated,)  # This will prevent anonymous users from accessing the view

    def post(self, request, pk, format=None):
        course = get_object_or_404(Course, pk=pk)
        course.students.add(request.user)
        return Response({'enrolled': True})

# The CourseEnrollView view handles user enrollment on courses. The preceding
# code is as follows:
# 1. You create a custom view that subclasses APIView.
# 2. You define a post() method for POST actions. No other HTTP method will
# be allowed for this view.
# 3. You expect a pk URL parameter containing the ID of a course. You retrieve
# the course by the given pk parameter and raise a 404 exception if it's not
# found.
# 4. You add the current user to the students many-to-many relationship of the
# Course object and return a successful response.

class CourseViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Course.objects.all()
    serializer_class = CourseSerializer

# You subclass ReadOnlyModelViewSet, which provides the read-only actions list()
# and retrieve() to both list objects, or retrieves a single object. 
    @action(detail=True,methods=['post'],authentication_classes=[BasicAuthentication],permission_classes=[IsAuthenticated])
    def enroll(self, request, *args, **kwargs):
        course = self.get_object()
        course.students.add(request.user)
        return Response({'enrolled': True})

    # In the preceding code, you add a custom enroll() method that represents an
    # additional action for this viewset. The preceding code is as follows:
    # 1. You use the action decorator of the framework with the parameter
    # detail=True to specify that this is an action to be performed on a
    # single object.
    # 2. The decorator allows you to add custom attributes for the action. You
    # specify that only the post() method is allowed for this view and set the
    # authentication and permission classes.
    # 3. You use self.get_object() to retrieve the Course object.
    # 4. You add the current user to the students many-to-many relationship and
    # return a custom success response.


# Let's create a view that mimics the behavior of the retrieve() action, but includes
# the course contents. Edit the api/views.py file and add the following method to the
# CourseViewSet class:

    @action(detail=True,
        methods=['get'],
        serializer_class=CourseWithContentsSerializer,
        authentication_classes=[BasicAuthentication],
        permission_classes=[IsAuthenticated, IsEnrolled])
    def contents(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)
    # The description of this method is as follows:
    # • You use the action decorator with the parameter detail=True to specify an
    # action that is performed on a single object.
    # • You specify that only the GET method is allowed for this action.
    # • You use the new CourseWithContentsSerializer serializer class that
    # includes rendered course contents.
    #  You use both IsAuthenticated and your custom IsEnrolled permissions.
    # By doing so, you make sure that only users enrolled on the course are able
    # to access its contents.
    # • You use the existing retrieve() action to return the Course object.
