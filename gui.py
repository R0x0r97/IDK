'''
GUI class
'''

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.stencilview import StencilView
from kivy.graphics import Ellipse, Line
from kivy.graphics.context_instructions import Color 
from kivy.graphics.vertex_instructions import Rectangle

from kivy.lang import Builder

#Written in kivy language
#Draws a white rectangle on the canvas (so the user knows where to draw)
Builder.load_string("""
<GuiCanvas>:
    canvas:
        Color:
            rgb: (1,1,1)
        Rectangle:
            pos:self.pos
            size:self.size
""")

class GuiCanvas(StencilView):
    #Constructor in python
    #Sets the width, height and pos attributes
    def __init__(self, **kwargs):
        super(GuiCanvas, self).__init__(**kwargs)
        self.width = 200
        self.height = 200
        self.pos = (.5,.5)

    #Start drawing the line
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 0, 0) #red
            #Draw a line
            touch.ud['line'] = Line(points=(touch.x, touch.y))
    #Draw that line while mouse is clicked
    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class GuiRunnerApp(App):
    #on_release callback
    #exports the canvas to png
    #resets the canvas (draws a white rectangle over it)
    def save_canvas(self, obj):
        self.wid.export_to_png("number.png")
        #Peter's script should be called here
        #Andras's script should be called here
        #This label shows the output from (Varga) Andras's script
        self.output_label.text = "1" #Hardcoded "1" for now
        with self.wid.canvas:
            Color(1, 1, 1) #transparent red
            Rectangle(pos = self.wid.pos, size=self.wid.size)

    #Builder method
    def build(self):
        #Create the canvas
        #pos_hint attribute required when adding the widget to a FloatLayout
        self.wid = GuiCanvas(size_hint=(None, None), size=Window.size, pos_hint={'center_x': .5, 'center_y': .5})
        
        #Save button
        btn_save = Button(text='Save', pos_hint={'center_x':.9,'center_y':.9},size_hint = (.1,.1))
        btn_save.bind(on_release=self.save_canvas)

        #Output text
        self.output_label = Label(text='Yo', font_size='20sp', pos_hint={'center_x':.1,'center_y':.1}, size_hint=(.1,.1))
        
        root = FloatLayout()
        root.add_widget(self.wid)
        root.add_widget(btn_save)
        root.add_widget(self.output_label)

        return root

#Run the script
if __name__ == '__main__':
    GuiRunnerApp().run()