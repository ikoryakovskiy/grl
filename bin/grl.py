#!/usr/bin/python

import yaml
from Tkinter import *
from ttk import *
from functools import partial
import StringIO
import os, inspect, sys

from grllib import *
from tooltip import *

class GrlMain:
  """Main window. Contains action buttons (save, run, add toplevel) and scrollable frame for GrlTopObjects."""
  def __init__(self, master):
    self.row = 0

    # Scrollable frame for GrlTopObjects.
    self.canvas = Canvas(master, borderwidth=0)
    self.frame = Frame(self.canvas)
    self.vsb = Scrollbar(master, orient="vertical", command=self.canvas.yview)
    self.canvas.configure(yscrollcommand=self.vsb.set)

    self.vsb.pack(side="right", fill="y")
    self.canvas.pack(side="left", fill="both", expand=True)
    self.canvas.create_window((4,4), window=self.frame, anchor="nw", 
                              tags="self.frame")
    self.frame.bind("<Configure>", self.OnFrameConfigure)

    # Buttons
    self.savebutton = Button(
          self.frame, text="Save", command=self.save
          )
    self.savebutton.grid(column=0, row=99, sticky=SW+E)
    self.runbutton = Button(
          self.frame, text="Run", command=self.run
          )
    self.runbutton.grid(column=1, row=99, sticky=SW+E)
    self.addbutton = Button(
          self.frame, text="+", command=self.add
          )
    self.addbutton.grid(column=2, row=99, sticky=S)
    
    self.objlist = list()
    self.add()
          
  def OnFrameConfigure(self, event):
    '''Reset the scroll region to encompass the inner frame'''
    self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    self.canvas['width'] = self.frame.winfo_width()
    self.canvas['height'] = self.frame.winfo_height()

  def remove(self, child):
    child.destroy()
    self.objlist.remove(child)

  def save(self):
    print "Writing", savefile
    output = open(savefile, "w")
    self.write(output)
    output.close()
    
  def run(self):
    print "Writing", tempfile
    output = open(tempfile, "w")
    self.write(output)
    output.close()
    os.system("cd " + binpath + "/../build && ./deploy " + tempfile)
    os.remove(tempfile)
    
  def add(self):
    obj = GrlTopObject(self, self.row)
    self.objlist.append(obj)
    self.row = self.row + 1
    
    self.refresh()
    
  def refresh(self):
    """Recalculate available parameters at each level of the configuration tree."""
    global params
    params = dict()
  
    for obj in self.objlist:
      obj.refresh()
    
  def write(self, output):
    for obj in self.objlist:
      obj.write(output, 0)
      
  def load(self, config):
    for obj in self.objlist:
      obj.destroy()
    self.objlist = list()
  
    for item in config:
      self.add()
      self.objlist[-1].load(item, config[item])
  
class GrlTopObject:
  """Top-level object. Contains editable name and frame for GrlObjects."""
  def __init__(self, parent, row):
    self.entry = Entry(parent.frame, width=10)
    self.entry.grid(row=row, column=0, sticky=NW)
    if row == 0:
      self.entry.insert(0, "experiment")
    self.frame = Frame(parent.frame)
    self.frame.grid(row=row, column=1, sticky=NW+E)
    self.frame.grid_columnconfigure(0, weight=1)
    self.obj = GrlObject(self, {'type': '', 'description':'Top-level object', 'optional':0, 'mutability':'configuration'})
    self.removebutton = Button(
      parent.frame, text="-", command=partial(parent.remove, self)
      )
    self.removebutton.grid(row=row, column=2, sticky=S)
    
  def change(self, event):
    return True
    
  def refresh(self):
    self.obj.refresh(self.entry.get())
  
  def destroy(self):
    self.removebutton.grid_forget()
    self.removebutton.destroy()
    self.obj.destroy()
    self.frame.grid_forget()
    self.frame.destroy()
    self.entry.grid_forget()
    self.entry.destroy()
    
  def write(self, output, indent):
    print >>output, ''.ljust(indent) + self.entry.get() + ":",
    self.obj.write(output, indent+2)
    
  def load(self, name, config):
    self.entry.delete(0, END)
    self.entry.insert(0, name)
    self.obj.load(config)

class GrlSubObject:
  """Sub-object. Contains fixed name label and frame for Object."""
  def __init__(self, parent, name, spec, row):
    self.name = name
    self.label = Label(parent.frame, text=name)
    self.label.grid(column=0, row=row, sticky=NW)
    self.frame = Frame(parent.frame)
    self.frame.grid(column=1, row=row, sticky=NW+E)
    self.frame.grid_columnconfigure(0, weight=1)
    self.obj = GrlObject(self, spec)
    
  def refresh(self, path):
    self.obj.refresh(path + "/" + self.name)
    
  def destroy(self):
    self.obj.destroy()
    self.frame.grid_forget()
    self.frame.destroy()
    self.label.grid_forget()
    self.label.destroy()
    
  def write(self, output, indent):
    print >>output, ''.ljust(indent) + self.label.cget("text") + ":",
    self.obj.write(output, indent+2)
    
  def load(self, config):
    self.label['text'] = self.name
    self.obj.load(config)
  
class GrlObject:
  """Object. Contains type selector and frame for all subobjects."""
  def __init__(self, parent, spec):
    self.spec = spec
    self.type = Combobox(parent.frame, state="readonly", width=40)
    if spec["mutability"] == "system":
      self.type['style'] = 'System.TCombobox'
    self.type.bind('<<ComboboxSelected>>', self.select)
    self.type.grid(column=0, row=0, sticky=W+E)
    self.tooltip = ToolTip(self.type, text=spec['description'], delay=1000)
    self.frame = Frame(parent.frame)
    self.frame.grid(column=0, row=1, sticky=W+E)
    self.frame.grid_columnconfigure(1, weight=1)
    self.objlist = list()
    
  def refresh(self, path):
    values = findrequests(requests, self.spec["type"])
    values.extend(findparams(params, self.spec["type"]))
    self.type['values'] = values
  
    # Refresh subobjects
    for obj in self.objlist:
      obj.refresh(path)

    type = self.type.get()
    params[path] = type
    
    # Add provided parameters
    if type in requests:
      if requests[type]:
        for key in requests[type]:
          if requests[type][key]["mutability"] == "provided":
            params[path + "/" + key] = requests[type][key]["type"]
    
  def destroy(self):
    self.type.grid_forget()
    self.type.destroy()
    self.frame.grid_forget()
    self.frame.destroy()
    
  def select(self, event):
    # Delete previous subobjects
    for obj in self.objlist:
      obj.destroy()
    self.objlist = list()

    type = self.type.get()
    
    # Add new ones
    row = 0
    if type in requests:
      if requests[type]:
        for key in requests[type]:
          if requests[type][key]["mutability"] != "provided":
            row = row + 1
            if isobject(requests[type][key]["type"]):
              obj = GrlSubObject(self, key, requests[type][key], row)
            else:
              obj = GrlVariable(self, key, requests[type][key], row)
            self.objlist.append(obj)

    if row == 0:
      # Ugly hack to force re-layout
      master = self.frame.master
      self.frame.grid_forget()
      self.frame.destroy()
      self.frame = Frame(master)
      self.frame.grid(column=0, row=1, sticky=W+E)
      self.frame.grid_columnconfigure(1, weight=1)
        
    # Recalculate parameters
    app.refresh()
    
  def write(self, output, indent):
    if self.type.get() in requests:
      print >>output, '\n' + ''.ljust(indent) + "type: " + self.type.get()
      for obj in self.objlist:
        obj.write(output, indent)
    else:
      print >>output, self.type.get()
      
  def load(self, config):
    if type(config) is str:
      self.type.set(config)
    else:
      self.type.set(config["type"])
      
    self.select(0)
    for obj in self.objlist:
      if obj.name in config:
        obj.load(config[obj.name])
    
class GrlVariable:
  """Non-object variable. Contains fixed name label and type-dependent entry field."""
  def __init__(self, parent, name, spec, row):
    self.spec = spec
    self.name = name
    
    self.label = Label(parent.frame, text=name)
    self.label.grid(column=0, row=row, sticky=NW)
    self.frame = Frame(parent.frame)
    self.frame.grid(column=1, row=row, sticky=NW+E)

    # Every type has at least a combobox
    self.value = Combobox(self.frame, width=40)
    if spec["mutability"] == "system":
      self.value['style'] = 'System.TCombobox'
    self.tooltip = ToolTip(self.value, text=spec['description'], delay=1000)
  
    if spec["type"] == 'int' and spec["max"]-spec["min"] < 10:
      self.value.set(spec["default"])
      self.value.grid(sticky=W+E)
      self.frame.grid_columnconfigure(0, weight=1)
    elif spec["type"] == 'double' and spec["max"]-spec["min"] < 1.0001:
      self.value.set(spec["default"])
      self.value.grid(column=0, row=0)
      self.scale = Scale(self.frame, from_=spec["min"], to=spec["max"], value=spec["default"], command=self.change)
      self.scale.grid(column=1, row=0, sticky=W+E)
      self.frame.grid_columnconfigure(1, weight=1)
    elif spec["type"] == 'string' and "options" in spec and len(spec["options"]) > 0:
      self.value.set(spec["default"])
      self.value.grid(sticky=W+E)
      self.frame.grid_columnconfigure(0, weight=1)
    else:
      self.value.set(spec["default"])
      self.value.grid(sticky=W+E)
      self.frame.grid_columnconfigure(0, weight=1)
      
  def change(self, value):
    self.value.delete(0, END)
    self.value.insert(0, value)
    
  def refresh(self, path):
    if self.spec["type"] == 'int' and self.spec["max"]-self.spec["min"] < 10:
      values = range(self.spec["min"], self.spec["max"]+1)
      values.extend(findparams(params, self.spec["type"]))
      self.value['values'] = values
    elif self.spec["type"] == 'double' and self.spec["max"]-self.spec["min"] < 1.0001:
      self.value['values'] = findparams(params, self.spec["type"])
    elif self.spec["type"] == 'string' and "options" in self.spec and len(self.spec["options"]) > 0:
      values = self.spec["options"]
      values.extend(findparams(params, self.spec["type"]))
      self.value['values'] = values
    else:
      self.value['values'] = findparams(params, self.spec["type"])

    params[path + "/" + self.name] = self.spec["type"]

  def destroy(self):
    self.label.grid_forget()
    self.label.destroy()
    self.value.grid_forget()
    self.value.destroy()
    if hasattr(self, 'scale'):
      self.scale.grid_forget()
      self.scale.destroy()
    self.frame.grid_forget()
    self.frame.destroy()
      
  def write(self, output, indent):
    if len(self.value.get()):
      print >>output, ''.ljust(indent) + self.label.cget("text") + ": " + self.value.get()
      
  def load(self, config):
    self.value.delete(0, END)
    self.value.insert(0, config)

# Set up paths
binpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
tempfile = "/tmp/grl." + str(os.getpid()) + ".yaml"
savefile = "configuration.yaml"

# Load object parameter requests, generated by requestgen
stream = file('requests.yaml', 'r')
requests = yaml.load(stream, OrderedDictYAMLLoader)
params = dict()
spec = {'type': '', 'description':'Experiment to run', 'optional':0}

# Setup up windowing system
root = Tk()
root.resizable(0,1)
root.title('GRL configurator')
Style().configure('System.TCombobox', fieldbackground='lightblue')

# Launch window
app = GrlMain(root)

# Load configuration, if specified
if len(sys.argv) > 1:
  savefile = sys.argv[1]
  if os.path.isfile(savefile):
    print "Loading", savefile
    stream = file(sys.argv[1], 'r')
    conf = yaml.load(stream, OrderedDictYAMLLoader)
    app.load(conf)

# Go
root.mainloop()