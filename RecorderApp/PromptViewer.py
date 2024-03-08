import cv2
import numpy as np
from types import NoneType
from typing import Optional
from enum import Enum, StrEnum
from dataclasses import dataclass, asdict
import sys
import os
import asyncio
import aiofiles

sys.path.append(os.path.abspath("."))
print(sys.path)
from PythonWrapper import Unicorn

class PromptViewer:
    def __init__(self, prompts:list[np.ndarray]= [], namedPrompts:dict[str, np.ndarray]={}):
        self.prompt:list[np.ndarray] = prompts
        self.WindowName = "Python-BCI-Recorder"
        self.PromptNameDict = namedPrompts
        cv2.namedWindow(self.WindowName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def LoadImage(self, path:str) -> int:
        img = cv2.imread(path)
        self.prompt.append(img)
        return len(self.prompt) - 1

    def AddImage(self, img:np.ndarray) -> int:
        self.prompt.append(img)
        return len(self.prompt) - 1
    
    def AddNamedImage(self, img:np.ndarray, name:str) -> str:
        self.PromptNameDict[name] = img
        return name
    
    def displayIndexed(self, index:int)->NoneType:
        cv2.imshow(self.WindowName, self.prompt[index])
        cv2.waitKey(0)
    
    def displayNamedPrompt(self, promptName:str)->NoneType:
        cv2.imshow(self.WindowName, self.PromptNameDict[promptName])
        cv2.waitKey(0)

class RecordChoices(StrEnum):
    Up     = "Up"
    Down   = "Down"
    Left   = "Left"
    Right  = "Right"
    Select = "Select"
    Rest   = "Rest"
    
@dataclass
class ExperimentConfig:
    """
    Config For Experiment Instance 
    RecordLength and BreakLength are in milliseconds
    """    
    ExperimentOrder : list[RecordChoices]
    RecordLength : int = 3000
    BreakLength : int = 5000
    AudioQueuePath : Optional[str] = None
    HeadsetConfig: Optional[Unicorn.UnicornAmplifierConfiguration] = None
    SubjectID : Optional[str] = None

class ExperimentInstance:
    def __init__(self, config:ExperimentConfig, promptViewer:PromptViewer):
        self.config = config
        self.PromptViewer = promptViewer
    
    def Config(self):
        self.Unicorn = Unicorn.Unicorn
        try:
            OpenDeviceOut = self.Unicorn.OpenDevice("UN-2021.12.19")
            if OpenDeviceOut[2] != Unicorn.UnicornReturnStatus.Success:
                raise Exception("Device Not Found")
            self.HandleRef = OpenDeviceOut[0]
            self.HandleVal = OpenDeviceOut[1]
            
            CurrentConfig = self.Unicorn.GetConfiguration(self.HandleVal)
            print("Current Config: ", CurrentConfig[1])
            
            if self.config.HeadsetConfig is not None:
                setConfigOut = self.Unicorn.SetConfiguration(self.HandleVal, self.config.HeadsetConfig)
                if setConfigOut != Unicorn.UnicornReturnStatus.Success:
                    raise Exception("Failed to set Configuration")
        except:
            pass

    def StartExperiment(self):
        self.ReadTask = asyncio.create_task(self.ExperimentThread())
        
    async def ExperimentThread(self):
        self.Unicorn.StartAcquisition(self.HandleVal, True)
        for choice in self.config.ExperimentOrder:
            self.PromptViewer.displayNamedPrompt("Rest")
            await asyncio.sleep(self.config.BreakLength/1000)
            self.PromptViewer.displayNamedPrompt(choice)
            await asyncio.sleep(self.config.RecordLength/1000)
            
        self.Unicorn.StopAcquisition(self.HandleVal)
        self.Unicorn.CloseDevice(self.HandleVal)
    
if __name__ == "__main__":
    exp  = ExperimentConfig(ExperimentOrder=[RecordChoices.Up, RecordChoices.Down, RecordChoices.Left, RecordChoices.Right, RecordChoices.Select, RecordChoices.Rest])
    