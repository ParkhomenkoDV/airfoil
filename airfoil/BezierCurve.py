import numpy as np
from matplotlib import pyplot as plt


class Bezier:
    def __init__(self, line):
        self.line = line
        self.index_02 = None  # Save the index of this point of the drag
        self.press = None  # Status indicator, 1 is pressed, None is not pressed
        self.pick = None  # Status identifier, 1 is the selected point and pressed, None is not selected
        self.motion = None  # Status identifier, 1 is for dragging, None is for dragging
        self.xs = list()  # Save the x coordinate of the point
        self.ys = list()  # Save the y coordinate of the point
        self.cidpress = line.figure.canvas.mpl_connect('button_press_event', self.on_press)  #
        self.cidrelease = line.figure.canvas.mpl_connect('button_release_event', self.on_release)  #
        self.cidmotion = line.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)  #
        self.cidpick = line.figure.canvas.mpl_connect('pick_event', self.on_picker)  #

    def on_press(self, event):
        if event.inaxes != self.line.axes: return
        self.press = 1

    def on_motion(self, event):
        if event.inaxes != self.line.axes: return
        if self.press is None: return
        if self.pick is None: return
        if self.motion is None:  # The whole if gets the point selected by the mouse
            self.motion = 1
            x = self.xs
            xdata = event.xdata
            ydata = event.ydata
            index_01 = 0
            for i in x:
                if abs(i - xdata) < 0.02:  # 0.02 is the radius of the point
                    if abs(self.ys[index_01] - ydata) < 0.02: break
                index_01 = index_01 + 1
            self.index_02 = index_01
        if self.index_02 is None: return
        self.xs[
            self.index_02] = event.xdata  # The coordinates of the mouse override the coordinates of the selected point
        self.ys[self.index_02] = event.ydata
        self.draw_01()

    def on_release(self, event):
        if event.inaxes != self.line.axes: return
        if self.pick == None:  # If it is not selected, then add a point
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        if self.pick == 1 and self.motion != 1:  # If it is a selected point, but not a drag point, then reduce the order
            x = self.xs
            xdata = event.xdata
            ydata = event.ydata
            index_01 = 0
            for i in x:
                if abs(i - xdata) < 0.02:
                    if abs(self.ys[index_01] - ydata) < 0.02: break
                index_01 = index_01 + 1
            self.xs.pop(index_01)
            self.ys.pop(index_01)
        self.draw_01()
        self.pick = None  # All state recovery, mouse down to release for a cycle
        self.motion = None
        self.press = None
        self.index_02 = None

    def on_picker(self, event):
        self.pick = 1

    def draw_01(self):  # Drawing function
        self.line.clear()  # If it is not cleared, the original image will be retained.
        self.line.axis([0, 1, 0, 1])  # x and y range 0 to 1
        self.bezier(self.xs, self.ys)  # Bezier Curve
        self.line.scatter(self.xs, self.ys, color='b', s=100, marker="o", picker=5)
        self.line.plot(self.xs, self.ys, color='r')  #
        self.line.figure.canvas.draw()  #

    def bezier(self, *args):  # Bezier curve formula conversion, get x and y
        n = len(args[0])
        xarray, yarray = [], []
        x, y = [], []
        index = 0
        for t in np.linspace(0, 1):
            for i in range(1, n):
                for j in range(0, n - i):
                    if i == 1:
                        xarray.insert(j, args[0][j] * (1 - t) + args[0][j + 1] * t)
                        yarray.insert(j, args[1][j] * (1 - t) + args[1][j + 1] * t)
                        continue
                    # i != 1 , calculated by the result of the last iteration
                    xarray[j] = xarray[j] * (1 - t) + xarray[j + 1] * t
                    yarray[j] = yarray[j] * (1 - t) + yarray[j + 1] * t
            if n == 1:
                x.insert(index, args[0][0])
                y.insert(index, args[1][0])
            else:
                x.insert(index, xarray[0])
                y.insert(index, yarray[0])
                xarray = []
                yarray = []
            index = index + 1
        self.line.plot(x, y)


fig = plt.figure(2, figsize=(12, 6))  # Create the second drawing object, 1200*600 pixels
ax = fig.add_subplot(111)
ax.set_title('Bezier')
myBezier = Bezier(ax)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
