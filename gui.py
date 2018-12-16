'''
GUI class
'''
from evaluate import evaluate_input
from formatting import * 

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.stencilview import StencilView
from kivy.graphics import *
from formatting import * 
from kivy.input.motionevent import MotionEvent

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
        self.line = Line(points=(0,0),width=0)
        self.width = 200
        self.height = 200
        self.pos = (.5, .5)

    #Start drawing the line
    def on_touch_down(self, touch):
        with self.canvas:
            Color(0, 0, 0)
            self.line = Line(points=(touch.x, touch.y),width = 8)
    #Draw that line while mouse is clicked
    def on_touch_move(self, touch):
        self.line.points += [touch.x, touch.y]


class GuiRunnerApp(App):
    #on_release callback
    #exports the canvas to png
    #converts the png image to a matrix
    #runs the evaluation on that matrix
    #does this for 3 images
    def save_canvas(self, obj):
        self.wid_1.export_to_png("res\\number0.png")
        self.wid_2.export_to_png("res\\number1.png")
        self.wid_3.export_to_png("res\\number2.png")
        
        output = 0

        for i in range(3):
            
            imName = "res\\number" + str(i) + ".png"
            if isBlank(imName) == True:
                continue
            matName = "res\\matrix" + str(i) + ".txt"
            PNGToIDX(imName, matName)
            output_cpy = evaluate_input(matName) 
            print(output_cpy)
            output = output*10 + output_cpy

        self.output_label.text = output.astype(str)

        self.clear(self.wid_1)
        self.clear(self.wid_2)
        self.clear(self.wid_3)

    #Clears the canvas (draws a white rectangle over it)
    def clear(self, wid):
        with wid.canvas:
            Color(1, 1, 1)
            Rectangle(pos = wid.pos, size = wid.size)

    #Builder method
    def build(self):
        #Create the canvas
        #pos_hint attribute required when adding the widget to a FloatLayout
        self.wid_1 = GuiCanvas(size_hint=(None, None), pos_hint={'center_x': .2, 'center_y': .5})
        self.wid_2 = GuiCanvas(size_hint=(None, None), pos_hint={'center_x': .5, 'center_y': .5})
        self.wid_3 = GuiCanvas(size_hint=(None, None), pos_hint={'center_x': .8, 'center_y': .5})
        
        #Save button
        btn_save = Button(text='Save', pos_hint={'center_x':.9,'center_y':.9},size_hint = (.1,.1))
        btn_save.bind(on_release=self.save_canvas)

        #Output text
        self.output_label = Label(text='Welcome', font_size='20sp', pos_hint={'center_x':.1,'center_y':.1}, size_hint=(.1,.1))
        
        root = FloatLayout()
        root.add_widget(self.wid_1)
        root.add_widget(self.wid_2)
        root.add_widget(self.wid_3)
        root.add_widget(btn_save)
        root.add_widget(self.output_label)

        return root

#Run the script
if __name__ == '__main__':
    GuiRunnerApp().run()