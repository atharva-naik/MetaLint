#!/usr/bin/env python3
#
#  plan_recognizer_factory.py
#  ma-goal-recognition
#
#  Created by Felipe Meneguzzi on 2020-03-12.
#  Copyright 2020 Felipe Meneguzzi. All rights reserved.
#
from recognizer.plan_recognizer import PlanRecognizer
# XXX My implementation of the factory relies on all recognizer classes having been imported into the plan_recognition module
from recognizer.ma_plan_recognizer import SATTeamPlanRecognizer
from recognizer.sat_plan_recognizer import SATPlanRecognizer

import importlib

import recognizer as rec


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Comment this out for Python 3
class PlanRecognizerFactory(metaclass=Singleton):
    # class PlanRecognizerFactory(object):
    __metaclass__ = Singleton

    def __init__(self, options=None):
        self.options = options

    def get_recognizer_names(self):
        recognizers = dict([(cls.name, cls) for name, cls in rec.__dict__.items() if isinstance(cls, type) and issubclass(cls, PlanRecognizer)])
        return list(recognizers.keys())
    
    def get_recognizer(self, name, options=None):
        """Returns an instance of PlanRecognizer given the name used in the parameters"""
        # From https://stackoverflow.com/questions/7584418/iterate-the-classes-defined-in-a-module-imported-dynamically
        # to create a dict that maps the names to the classes
        # dict([(name, cls) for name, cls in mod.__dict__.items() if isinstance(cls, type)])
        if options is None:
            options = self.options

        # Finding the objects 
        recognizers = dict([(cls.name, cls) 
                            for _, cls in rec.__dict__.items()
                            if isinstance(cls, object) and hasattr(cls, 'name')])  # Hack to get my instantiation to work w/ Python 2.7
        # The line below works in Python 3.7 (bot not 2.7, and I hate python for that)
        # recognizers = dict([(cls.name, cls) for _, cls in plan_recognition.__dict__.items() if isinstance(cls, object) and issubclass(cls, PlanRecognizer)])
        # print(recognizers)
        recognizer = recognizers[name](options)

        return recognizer


