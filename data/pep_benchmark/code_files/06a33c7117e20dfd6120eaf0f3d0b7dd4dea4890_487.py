from abc import ABCMeta, abstractmethod

import tkinter as tk
from command.classes import *

class Point:

    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

class Drawable(metaclass=ABCMeta):
    @abstractmethod
    def draw(self, x:int, y:int):
        pass

class DrawCanvas(tk.Canvas, Drawable):

    def __init__(self, width:int, height:int, history):
        super().__init__()
        self.configure(width=width, height=height, background="#ffffff")
        self._red_color = "#ff0000"
        self._radius = 6
        self._history = history

    def paint(self):
        self._history.execute()

    def draw(self, x:int, y:int):

        fig = self.create_oval(x, y, x + self._radius, y + self._radius)
        self.itemconfigure(fig, fill=self._red_color)

class DrawCommand(Command):

    def __init__(self, drawable, position):
        self._drawable = drawable
        self._position = position

    def execute(self):
        self._drawable.draw(self._position.x, self._position.y)

