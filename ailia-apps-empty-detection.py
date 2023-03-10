# ailia APPS Empty Detection
# (C) 2022-2023 AXELL CORPORATION

import sys
import time
from signal import SIGINT

import numpy as np
import cv2
import json
from matplotlib import cm
from PIL import Image, ImageTk

import ailia

# import original modules
sys.path.append('./util')
from utils import get_base_parser, update_parser
from model_utils import check_and_download_models  # noqa: E402
from image_utils import normalize_image  # noqa: E402C
from webcamera_utils import get_capture, get_writer  # noqa: E402
# logger
from logging import getLogger  # noqa: E402

import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import os

logger = getLogger(__name__)

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'ailia APPS empty detection', None, None)

args = update_parser(parser)


# ======================
# Video
# ======================

input_index = 0
listsInput = None
ListboxInput = None
input_list = []

def get_input_list():
    if args.debug:
        return ["Camera:0"]

    index = 0
    inputs = []
    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            inputs.append("Camera:"+str(index))
        else:
            break
        index=index+1
        cap.release()

    if len(inputs) == 0:
        inputs.append("demo.mp4")

    return inputs

def input_changed(event):
    global input_index, input_list, textInputVideoDetail
    selection = event.widget.curselection()
    if selection:
        input_index = selection[0]
    else:
        input_index = 0   
    if "Camera:" in input_list[input_index]:
        textInputVideoDetail.set(input_list[input_index])
    else:
        textInputVideoDetail.set(os.path.basename(input_list[input_index]))
        
    #print("input",input_index)

def input_video_dialog():
    global textInputVideoDetail, listsInput, ListboxInput, input_index, input_list
    fTyp = [("All Files", "*.*"), ("Video files","*.mp4")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        textInputVideoDetail.set(os.path.basename(file_name))
        input_list.append(file_name)
        listsInput.set(input_list)
        ListboxInput.select_clear(input_index)
        input_index = len(input_list)-1
        ListboxInput.select_set(input_index)

def apply_path_to_ui():
    global textOutputVideoDetail
    textOutputVideoDetail.set(os.path.basename(args.savepath))
    global textOutputCsvDetail
    textOutputCsvDetail.set(os.path.basename(args.csvpath))
    global textOutputImageDetail
    textOutputImageDetail.set(os.path.basename(args.imgpath))

def output_video_dialog():
    global textOutputVideoDetail
    fTyp = [("Output Video File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        args.savepath = file_name
        apply_path_to_ui()

def output_csv_dialog():
    global textOutputCsvDetail
    fTyp = [("Output Csv File", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        args.csvpath = file_name
        apply_path_to_ui()

def output_img_dialog():
    fTyp = [("Output Image Folder", "*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askdirectory(initialdir=iDir)
    if len(file_name) != 0:
        args.imgpath = file_name
        apply_path_to_ui()

# ======================
# Environment
# ======================

env_index = args.env_id

def get_env_list():
    env_list = []
    for env in ailia.get_environment_list():
        env_list.append(env.name)
    return env_list  

def environment_changed(event):
    global env_index
    selection = event.widget.curselection()
    if selection:
        env_index = selection[0]
    else:
        env_index = 0
    #print("env",env_index)

# ======================
# Model
# ======================

model_index = 0

def get_model_list():
    model_list = ["SwinB_896_4x-lvis", "SwinB_896_4x-in21k", "R50_640_4x-lvis", "R50_640_4x-in21k"]
    return model_list  

def model_changed(event):
    global model_index
    selection = event.widget.curselection()
    if selection:
        model_index = selection[0]
    else:
        model_index = 0
    #print("model",model_index)

# ======================
# EditableListbox
# ======================

class EditableListbox(tk.Listbox):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.edit_item = None
        self.bind("<Double-1>", self._start_edit)

    def _start_edit(self, event):
        index = self.index(f"@{event.x},{event.y}")
        self.start_edit(index)
        return "break"

    def start_edit(self, index):
        self.edit_item = index
        text = self.get(index)
        y0 = self.bbox(index)[1]
        entry = tk.Entry(self, borderwidth=0, highlightthickness=1)
        entry.bind("<Return>", self.accept_edit)
        entry.bind("<Escape>", self.cancel_edit)

        entry.insert(0, text)
        entry.selection_from(0)
        entry.selection_to("end")
        entry.place(relx=0, y=y0, relwidth=1, width=-1)
        entry.focus_set()
        entry.grab_set()

    def cancel_edit(self, event):
        event.widget.destroy()

    def accept_edit(self, event):
        new_data = event.widget.get()
        new_data = new_data.replace(" ","_")

        self.delete(self.edit_item)
        self.insert(self.edit_item, new_data)
        event.widget.destroy()

        global area_list, area_idx
        area_list[area_idx]["id"] = new_data
        update_area()

# ======================
# Area setting
# ======================

# parking lot
#area_list = [
#    {"id":"area0","area":[(430,553),(485,540),(509,549),(453,561)]},
#    {"id":"area1","area":[(547,598),(608,580),(641,592),(576,611)]},
#    {"id":"area2","area":[(640,633),(718,615),(748,628),(671,647)]},
#]

# default
area_list = [
    {"id":"area0","area":[(0,0),(100,0),(100,100),(0,100)]}
]

area_idx = 0

def display_area(frame):
    global area_list, area_idx
    for a in range(len(area_list)):
        area_id = area_list[a]["id"]
        target_lines = area_list[a]["area"]
        if a == area_idx:
            color = (0,0,255)
        else:
            color = (255,0,0)
        if len(target_lines) >= 2:
            for i in range(len(target_lines) - 1):
                cv2.line(frame, (target_lines[i][0], target_lines[i][1]), (target_lines[i+1][0], target_lines[i+1][1]), color, thickness=1)
            if len(target_lines) >= 4:
                cv2.line(frame, (target_lines[3][0], target_lines[3][1]), (target_lines[0][0], target_lines[0][1]), color, thickness=1)
            cv2.putText(frame, area_id, (target_lines[0][0] + 5,target_lines[0][1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=1)
        for i in range(0, len(target_lines)):
            cv2.circle(frame, center = (target_lines[i][0], target_lines[i][1]), radius = 3, color=color, thickness=3)

g_frame = None
areaWindows = None
areaListWindow = None

def close_area_window():
    global areaWindows, areaListWindow
    if areaWindows != None and areaWindows.winfo_exists():
        areaWindows.destroy()
        areaWindows = None
    if areaListWindow != None and areaListWindow.winfo_exists():
        areaListWindow.destroy()
        areaListWindow = None

def get_video_path():
    global input_list, input_index
    if "Camera:" in input_list[input_index]:
        return input_index
    else:
        return input_list[input_index]

def add_area():
    global area_list, area_idx, listsArea, ListboxArea
    area_list.append({"id":"area"+str(len(area_list)), "area":[]})
    ListboxArea.select_clear(area_idx)
    area_idx = len(area_list) - 1
    listsArea.set(get_area_list())
    ListboxArea.select_set(area_idx)
    update_area()

def remove_area():
    global area_list, area_idx, listsArea, ListboxArea
    if len(area_list) == 1:
        return
    ListboxArea.select_clear(area_idx)
    area_list.remove(area_list[area_idx])
    area_idx = 0
    listsArea.set(get_area_list())
    ListboxArea.select_set(area_idx)
    update_area()

ListboxArea = None
listsArea = None

def area_select_changed(event):
    global area_idx, ListboxArea
    selection = event.widget.curselection()
    if selection:
        area_idx = selection[0]
    else:
        area_idx = 0
    ListboxArea.select_set(area_idx)
    update_area()

def get_area_list():
    global area_list
    id_list = []
    for i in range(len(area_list)):
        id_list.append(area_list[i]["id"])
    return id_list

def set_area():
    global g_frame, g_frame_shown
    global textCrossingLine
    global areaWindows, areaListWindow

    if (areaWindows != None and areaWindows.winfo_exists()) or (areaListWindow != None and areaListWindow.winfo_exists()):
        close_area_window()

    capture = get_capture(get_video_path())
    assert capture.isOpened(), 'Cannot capture source'
    ret, frame = capture.read()
    g_frame = frame

    areaWindows = tk.Toplevel()
    areaWindows.title("Set area")
    areaWindows.geometry(str(g_frame.shape[1])+"x"+str(g_frame.shape[0]))
    tk.Label(areaWindows, text ="Please set area by click").pack()
    areaWindows.canvas = tk.Canvas(areaWindows)
    areaWindows.canvas.bind('<Button-1>', set_line)
    areaWindows.canvas.pack(expand = True, fill = tk.BOTH)

    areaListWindow = tk.Toplevel()
    areaListWindow.title("Select target area")
    areaListWindow.geometry("400x400")
    textAreaListHeader = tk.StringVar(areaListWindow)
    textAreaListHeader.set("Area list")
    labelAreaListHeader = tk.Label(areaListWindow, textvariable=textAreaListHeader)
    labelAreaListHeader.grid(row=0, column=0, sticky=tk.NW)
    area_list = get_area_list()
    global listsArea
    listsArea = tk.StringVar(value=area_list)
    global ListboxArea
    ListboxArea = EditableListbox(areaListWindow, listvariable=listsArea, width=20, height=16, selectmode="single", exportselection=False)
    ListboxArea.bind("<<ListboxSelect>>", area_select_changed)
    ListboxArea.select_set(area_idx)
    ListboxArea.grid(row=1, column=0, sticky=tk.NW, rowspan=1, columnspan=20)

    textAreaAdd = tk.StringVar(areaListWindow)
    textAreaAdd.set("Add area")
    buttonAreaAdd = tk.Button(areaListWindow, textvariable=textAreaAdd, command=add_area, width=14)
    buttonAreaAdd.grid(row=22, column=0, sticky=tk.NW)

    textAreaRemove = tk.StringVar(areaListWindow)
    textAreaRemove.set("Remove area")
    buttonAreaRemove = tk.Button(areaListWindow, textvariable=textAreaRemove, command=remove_area, width=14)
    buttonAreaRemove.grid(row=23, column=0, sticky=tk.NW)

    update_area()

def update_area():
    global g_frame
    frame = g_frame.copy()
    display_area(frame)
    update_frame_image(frame)

def update_frame_image(frame):
    global areaWindows
    cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    areaWindows.photo_image = ImageTk.PhotoImage(image=pil_image)
    areaWindows.canvas.create_image(
            frame.shape[1] / 2,
            frame.shape[0] / 2,                   
            image=areaWindows.photo_image
            )

def set_line(event):
    global area_list, area_idx
    target_lines = area_list[area_idx]["area"]
    x = event.x
    y = event.y
    if len(target_lines)>=4:
        target_lines.clear()
    else:
        target_lines.append((x,y))
    frame = g_frame.copy()
    display_area(frame)
    update_frame_image(frame)

# ======================
# Menu functions
# ======================

def get_settings():
    global area_list
    settings = {}
    settings["area_list"] = area_list

    global model_index
    settings["model_type"] = get_model_list()[model_index]

    global resolutionTextEntry
    settings["detection_width"] = resolutionTextEntry.get()

    global areaThresholdTextEntry
    settings["area_threshold"] = areaThresholdTextEntry.get()

    global labelAcceptTextEntry
    settings["accept_label"] = labelAcceptTextEntry.get()

    global labelDenyTextEntry
    settings["deny_label"] = labelDenyTextEntry.get()

    global checkBoxMultipleAssignBln
    if checkBoxMultipleAssignBln.get():
        settings["multiple_assign"] = True
    else:
        settings["multiple_assign"] = False
    
    settings["savepath"] = args.savepath
    settings["csvpath"] = args.csvpath
    settings["imgpath"] = args.imgpath

    return settings

def set_settings(settings):
    global area_list
    area_list = settings["area_list"]

    global model_index, ListboxModel
    model_list = get_model_list()
    for i in range(len(model_list)):
        if settings["model_type"] == model_list[i]:
            model_index = i
    ListboxModel.select_set(model_index)

    global resolutionTextEntry
    resolutionTextEntry.delete(0, tk.END)
    resolutionTextEntry.insert(0, str(settings["detection_width"]))

    global areaThresholdTextEntry
    areaThresholdTextEntry.delete(0, tk.END)
    areaThresholdTextEntry.insert(0, str(settings["area_threshold"]))

    global labelAcceptTextEntry
    labelAcceptTextEntry.delete(0, tk.END)
    labelAcceptTextEntry.insert(0, str(settings["accept_label"]))

    global labelDenyTextEntry
    labelDenyTextEntry.delete(0, tk.END)
    labelDenyTextEntry.insert(0, str(settings["deny_label"]))

    global checkBoxMultipleAssignBln
    checkBoxMultipleAssignBln.set(settings["multiple_assign"])

    if "savepath" in settings:
        args.savepath = settings["savepath"]
    if "csvpath" in settings:
        args.csvpath = settings["csvpath"]
    if "imgpath" in settings:
        args.imgpath = settings["imgpath"]
    
    apply_path_to_ui()

def menu_file_open_click():
    fTyp = [("Config files","*.json")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        with open(file_name, 'r') as json_file:
            settings = json.load(json_file)
            set_settings(settings)

def menu_file_saveas_click():
    fTyp = [("Config files", "*.json")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file_name = tk.filedialog.asksaveasfilename(filetypes=fTyp, initialdir=iDir)
    if len(file_name) != 0:
        with open(file_name, 'w') as json_file:
            settings = get_settings()
            json.dump(settings, json_file)

def menu(root):
    menubar = tk.Menu(root)

    menu_file = tk.Menu(menubar, tearoff = False)
    menu_file.add_command(label = "Load settings",  command = menu_file_open_click,  accelerator="Ctrl+O")
    menu_file.add_command(label = "Save settings", command = menu_file_saveas_click, accelerator="Ctrl+S")
    #menu_file.add_separator() # ????????????
    #menu_file.add_command(label = "Quit",            command = root.destroy)

    menubar.add_cascade(label="File", menu=menu_file)

    root.config(menu=menubar)

# ======================
# GUI functions
# ======================

root = None
resolutionTextEntry = None
areaThresholdTextEntry = None
labelAcceptTextEntry = None
labelDenyTextEntry = None
checkBoxMultipleAssignBln = None
ListboxModel = None

def ui():
    # root?????????????????????????????????
    global root
    root = tk.Tk()
    root.title("ailia APPS Empty Detection")
    root.geometry("720x360")

    # ??????????????????
    menu(root)

    # ??????????????????
    global input_list
    input_list = get_input_list()
    model_list = get_model_list()
    env_list = get_env_list()

    # ???????????????????????????????????????
    frame = ttk.Frame(root)
    frame.pack(padx=10,pady=10)

    textInputVideo = tk.StringVar(frame)
    textInputVideo.set("Input video")
    buttonInputVideo = tk.Button(frame, textvariable=textInputVideo, command=input_video_dialog, width=14)
    buttonInputVideo.grid(row=0, column=0, sticky=tk.NW)

    global textInputVideoDetail
    textInputVideoDetail = tk.StringVar(frame)
    textInputVideoDetail.set(input_list[input_index])
    labelInputVideoDetail = tk.Label(frame, textvariable=textInputVideoDetail)
    labelInputVideoDetail.grid(row=0, column=1, sticky=tk.NW)

    global textCrossingLine
    textCrossingLine = tk.StringVar(frame)
    textCrossingLine.set("Set area")
    buttonCrossingLine = tk.Button(frame, textvariable=textCrossingLine, command=set_area, width=14)
    buttonCrossingLine.grid(row=1, column=0, sticky=tk.NW)

    textOutputVideo = tk.StringVar(frame)
    textOutputVideo.set("Output video")
    buttonOutputVideo = tk.Button(frame, textvariable=textOutputVideo, command=output_video_dialog, width=14)
    buttonOutputVideo.grid(row=2, column=0, sticky=tk.NW)

    global textOutputVideoDetail
    textOutputVideoDetail = tk.StringVar(frame)
    textOutputVideoDetail.set(args.savepath)
    labelOutputVideoDetail= tk.Label(frame, textvariable=textOutputVideoDetail)
    labelOutputVideoDetail.grid(row=2, column=1, sticky=tk.NW)

    textOutputCsv = tk.StringVar(frame)
    textOutputCsv.set("Output csv")
    buttonOutputCsv = tk.Button(frame, textvariable=textOutputCsv, command=output_csv_dialog, width=14)
    buttonOutputCsv.grid(row=3, column=0, sticky=tk.NW)

    global textOutputCsvDetail
    textOutputCsvDetail = tk.StringVar(frame)
    textOutputCsvDetail.set(args.csvpath)
    labelOutputCsvDetail= tk.Label(frame, textvariable=textOutputCsvDetail)
    labelOutputCsvDetail.grid(row=3, column=1, sticky=tk.NW)

    textOutputImage = tk.StringVar(frame)
    textOutputImage.set("Output image")
    buttonOutputImage = tk.Button(frame, textvariable=textOutputImage, command=output_img_dialog, width=14)
    buttonOutputImage.grid(row=4, column=0, sticky=tk.NW)

    global textOutputImageDetail
    textOutputImageDetail = tk.StringVar(frame)
    textOutputImageDetail.set(args.imgpath)
    labelOutputImageDetail= tk.Label(frame, textvariable=textOutputImageDetail)
    labelOutputImageDetail.grid(row=4, column=1, sticky=tk.NW)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Run")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=run, width=14)
    buttonTrainVideo.grid(row=5, column=0, sticky=tk.NW)

    textTrainVideo = tk.StringVar(frame)
    textTrainVideo.set("Stop")
    buttonTrainVideo = tk.Button(frame, textvariable=textTrainVideo, command=stop, width=14)
    buttonTrainVideo.grid(row=6, column=0, sticky=tk.NW)

    global listsInput, ListboxInput

    textInputVideoHeader = tk.StringVar(frame)
    textInputVideoHeader.set("Inputs")
    labelInputVideoHeader = tk.Label(frame, textvariable=textInputVideoHeader)
    labelInputVideoHeader.grid(row=0, column=2, sticky=tk.NW)

    listsInput = tk.StringVar(value=input_list)
    ListboxInput = tk.Listbox(frame, listvariable=listsInput, width=26, height=4, selectmode="single", exportselection=False)
    ListboxInput.bind("<<ListboxSelect>>", input_changed)
    ListboxInput.select_set(input_index)
    ListboxInput.grid(row=1, column=2, sticky=tk.NW, rowspan=3, columnspan=2)

    lists = tk.StringVar(value=model_list)
    listEnvironment =tk.StringVar(value=env_list)

    global ListboxModel
    ListboxModel = tk.Listbox(frame, listvariable=lists, width=26, height=4, selectmode="single", exportselection=False)
    ListboxEnvironment = tk.Listbox(frame, listvariable=listEnvironment, width=26, height=4, selectmode="single", exportselection=False)

    ListboxModel.bind("<<ListboxSelect>>", model_changed)
    ListboxEnvironment.bind("<<ListboxSelect>>", environment_changed)

    ListboxModel.select_set(model_index)
    ListboxEnvironment.select_set(env_index)

    textModel = tk.StringVar(frame)
    textModel.set("Models")
    labelModel = tk.Label(frame, textvariable=textModel)
    labelModel.grid(row=4, column=2, sticky=tk.NW, rowspan=1)
    ListboxModel.grid(row=5, column=2, sticky=tk.NW, rowspan=2)

    textEnvironment = tk.StringVar(frame)
    textEnvironment.set("Environment")
    labelEnvironment = tk.Label(frame, textvariable=textEnvironment)
    labelEnvironment.grid(row=8, column=2, sticky=tk.NW, rowspan=1)
    ListboxEnvironment.grid(row=9, column=2, sticky=tk.NW, rowspan=4)

    textOptions = tk.StringVar(frame)
    textOptions.set("Options")
    labelOptions = tk.Label(frame, textvariable=textOptions)
    labelOptions.grid(row=0, column=3, sticky=tk.NW)

    textResolution = tk.StringVar(frame)
    textResolution.set("Resolution")
    labelResolution = tk.Label(frame, textvariable=textResolution)
    labelResolution.grid(row=1, column=3, sticky=tk.NW)

    global resolutionTextEntry
    resolutionTextEntry = tkinter.Entry(frame, width=20)
    resolutionTextEntry.insert(tkinter.END,"800")
    resolutionTextEntry.grid(row=2, column=3, sticky=tk.NW, rowspan=1)

    textAreaThreshold= tk.StringVar(frame)
    textAreaThreshold.set("Area Threshold")
    labelAreaThreshold = tk.Label(frame, textvariable=textAreaThreshold)
    labelAreaThreshold.grid(row=3, column=3, sticky=tk.NW)

    global areaThresholdTextEntry
    areaThresholdTextEntry = tkinter.Entry(frame, width=20)
    areaThresholdTextEntry.insert(tkinter.END,"0.125")
    areaThresholdTextEntry.grid(row=4, column=3, sticky=tk.NW, rowspan=1)

    textLabels = tk.StringVar(frame)
    textLabels.set("Accept Label")
    labelLabels = tk.Label(frame, textvariable=textLabels)
    labelLabels.grid(row=5, column=3, sticky=tk.NW)

    global labelAcceptTextEntry
    labelAcceptTextEntry = tkinter.Entry(frame, width=20)
    labelAcceptTextEntry.insert(tkinter.END,"all")
    labelAcceptTextEntry.grid(row=6, column=3, sticky=tk.NW, rowspan=1)

    textDenyLabels = tk.StringVar(frame)
    textDenyLabels.set("Deny Label")
    labelDenyLabels = tk.Label(frame, textvariable=textDenyLabels)
    labelDenyLabels.grid(row=7, column=3, sticky=tk.NW)

    global labelDenyTextEntry
    labelDenyTextEntry = tkinter.Entry(frame, width=20)
    labelDenyTextEntry.insert(tkinter.END,"none")
    labelDenyTextEntry.grid(row=8, column=3, sticky=tk.NW, rowspan=1)

    global checkBoxMultipleAssignBln
    checkBoxMultipleAssignBln = tkinter.BooleanVar()
    checkBoxMultipleAssignBln.set(False)
    checkBoxMultipleAssign = tkinter.Checkbutton(frame, variable=checkBoxMultipleAssignBln, text='Multiple Assign')
    checkBoxMultipleAssign.grid(row=9, column=3, sticky=tk.NW, rowspan=1)

    root.mainloop()

# ======================
# MAIN functions
# ======================

def main():
    args.savepath = ""
    args.csvpath = ""
    args.imgpath = ""
    ui()

import subprocess

proc = None

def run():
    close_area_window()

    global proc

    if not (proc==None):
        proc.kill()
        proc=None

    cmd = sys.executable

    args_dict = {}#vars(args)
    args_dict["video"] = get_video_path()
        
    settings = get_settings()
    if settings["savepath"]:
        args_dict["savepath"] = settings["savepath"]
    if settings["csvpath"]:
        args_dict["csvpath"] = settings["csvpath"]
    if settings["imgpath"]:
        args_dict["imgpath"] = settings["imgpath"]

    global model_index
    args_dict["model_type"] = get_model_list()[model_index].split("-")[0]
    args_dict["vocabulary"] = get_model_list()[model_index].split("-")[1]

    global env_index
    args_dict["env_id"] = env_index

    global resolutionTextEntry
    if resolutionTextEntry:
        args_dict["detection_width"] = int(resolutionTextEntry.get())

    global areaThresholdTextEntry
    if areaThresholdTextEntry:
        args_dict["area_threshold"] = float(areaThresholdTextEntry.get())

    global checkBoxMultipleAssignBln
    if checkBoxMultipleAssignBln.get():
        args_dict["multiple_assign"] = True

    version = ailia.get_version().split(".")
    major_version = int(version[0])
    minor_version = int(version[1])
    revision_version = int(version[2])
    if major_version > 1 or minor_version > 2 or revision_version >= 14:
        args_dict["opset16"] = True

    area_info = ""
    for i in range(len(area_list)):
        if (len(area_list[i]["area"]) >= 4):
            target_lines = area_list[i]["area"]
            area_id = area_list[i]["id"]
            line1 = str(target_lines[0][0]) + " " + str(target_lines[0][1]) + " " + str(target_lines[1][0]) + " " + str(target_lines[1][1])
            line2 = str(target_lines[2][0]) + " " + str(target_lines[2][1]) + " " + str(target_lines[3][0]) + " " + str(target_lines[3][1])
            if area_info != "":
                area_info = area_info + " "
            area_info = area_info + area_id + " " + line1 + " " + line2
    if area_info != "":
        args_dict["area"] = area_info

    options = []
    for key in args_dict:
        if key=="ftype":
            continue
        if args_dict[key] is not None:
            if args_dict[key] is True:
                options.append("--"+key)
            elif args_dict[key] is False:
                continue
            else:
                options.append("--"+key)
                options.append(str(args_dict[key]))

    global labelAcceptTextEntry
    if labelAcceptTextEntry:
        label_text = labelAcceptTextEntry.get().split(",")
        for text in label_text:
                options.append("--accept_label")
                options.append(text)

    global labelDenyTextEntry
    if labelDenyTextEntry:
        label_text = labelDenyTextEntry.get().split(",")
        for text in label_text:
                options.append("--deny_label")
                options.append(text)

    cmd = [cmd, "detic.py"] + options
    print(" ".join(cmd))

    dir = "./object_detection/detic/"

    proc = subprocess.Popen(cmd, cwd=dir)
    try:
        outs, errs = proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def stop():
    global proc
    if not (proc==None):
        proc.send_signal(SIGINT)
        proc=None

if __name__ == '__main__':
    main()
