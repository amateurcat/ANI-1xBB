import py3Dmol, io
from ipywidgets import interact, IntSlider
import numpy as np
from converters import std_to_xyz

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def Atomic3D(numbers, coord):
    # converts numbers and coord into .xyz format for py3Dmol visualization
    s = io.StringIO()
    std_to_xyz(numbers, coord, s)
    s = s.getvalue()

    view = py3Dmol.view(width=300, height=300)
    view.addModel(s, "xyz")
    view.setStyle({"stick" : {"colorscheme" : {"prop" : "elem"}}}, viewer=view)
    view.setBackgroundColor(rgb_to_hex((0, 255, 0)), viewer=view)
    
    view.zoomTo()
    view.render()
    view.show()

def SlidableAtomic3D(numbers, coord):
    def view_conformer(idx):
        Atomic3D(np.atleast_2d(numbers[idx]), np.expand_dims(coord[idx], axis=0))
    slider = IntSlider(min=0, max=len(numbers)-1, step=1, value=0, description='Index')
    interact(view_conformer, idx=slider)