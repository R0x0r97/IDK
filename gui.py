from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line, Rectangle


class Gui(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 1) #white
            #Draw a line
            touch.ud['line'] = Line(points=(touch.x, touch.y))
    #Draw that line while mouse is clicked
    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class GuiRunner(App):
    def build(self):
        parent = Widget()
        self.painter = Gui(size=[i/2.0 for i in Window.size])
        with self.painter.canvas:
            Color(1, 0, 0, 0.3) #transparent red
            Rectangle(pos=self.painter.pos, size=self.painter.size)

        #Create he save button
        savebtn = Button(text='Save')
        #Bind the callback
        savebtn.bind(on_release=self.save_canvas)
        parent.add_widget(self.painter)
        parent.add_widget(savebtn)
        return parent

    def save_canvas(self, obj):
        self.painter.export_to_png("number.png")

GuiRunner().run()