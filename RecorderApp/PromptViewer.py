import cv2
import numpy as np
from types import NoneType
from typing import Optional
from enum import Enum, StrEnum
from dataclasses import dataclass, asdict
import sys
import os
import asyncio
from threading import Thread
from queue import Queue
import aiofiles
import uvloop
import datetime
from pygame import mixer
import time

uvloop.install()
sys.path.append(os.path.abspath("."))
from PythonWrapper import Unicorn


class PromptViewer:
    def __init__(
        self, prompts: list[np.ndarray] = [], namedPrompts: dict[str, np.ndarray] = {}
    ):
        self.prompt: list[np.ndarray] = prompts
        self.WindowName = "Python-BCI-Recorder"
        self.PromptNameDict = namedPrompts
        cv2.namedWindow(self.WindowName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            self.WindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    def LoadImage(self, path: str) -> int:
        img = cv2.imread(path)
        self.prompt.append(img)
        return len(self.prompt) - 1

    def AddImage(self, img: np.ndarray) -> int:
        self.prompt.append(img)
        return len(self.prompt) - 1

    def AddNamedImage(self, img: np.ndarray, name: str) -> str:
        self.PromptNameDict[name] = img
        return name

    def displayIndexed(self, index: int) -> NoneType:
        cv2.imshow(self.WindowName, self.prompt[index])
        _ = cv2.waitKey(1) & 0xFF == ord("0")

    def displayNamedPrompt(self, promptName: str) -> NoneType:
        cv2.imshow(self.WindowName, self.PromptNameDict[promptName])
        _ = cv2.waitKey(1) & 0xFF == ord("0")


class RecordChoices(StrEnum):
    Up = "Up"
    Down = "Down"
    Left = "Left"
    Right = "Right"
    Select = "Select"
    Rest = "Rest"


@dataclass
class ExperimentConfig:
    """
    Config For Experiment Instance
    RecordLength and BreakLength are in milliseconds
    """

    ExperimentOrder: list[RecordChoices]
    RecordLength: int = 5000
    BreakLength: int = 10000
    HeadsetConfig: Optional[Unicorn.UnicornAmplifierConfiguration] = None
    SubjectID: Optional[str] = None
    AudioFile: str = "RecorderApp/Signal.mp3"
    HeadsetSerial: str = "UN-2021.12.19"


@dataclass
class Run:
    run: bool


class ExperimentInstance:
    def __init__(self, config: ExperimentConfig, promptViewer: PromptViewer):
        self.config = config
        self.PromptViewer = promptViewer
        self.OutputQueue: Queue[tuple[list[float], str]] = Queue()

    def Config(self):
        mixer.init()
        mixer.music.load(self.config.AudioFile)
        # mixer.music.play()
        self.Unicorn = Unicorn
        try:
            OpenDeviceOut = self.Unicorn.OpenDevice(self.config.HeadsetSerial)
            if OpenDeviceOut[2] != Unicorn.UnicornReturnStatus.Success:
                raise Exception("Device Not Found")
            self.HandleRef = OpenDeviceOut[0]
            self.HandleVal = OpenDeviceOut[1]

            CurrentConfig = self.Unicorn.GetConfiguration(self.HandleVal)
            print("Current Config: ", CurrentConfig[1])
            if self.config.HeadsetConfig is not None:
                setConfigOut = self.Unicorn.SetConfiguration(
                    self.HandleVal, self.config.HeadsetConfig
                )
                if setConfigOut != Unicorn.UnicornReturnStatus.Success:
                    raise Exception("Failed to set Configuration")
            else:
                self.config.HeadsetConfig = CurrentConfig[0]
            mixer.music.play()
        except Exception as e:
            print(e)

    async def StartExperiment(self):
        self.ReadTask = self.ExperimentThread()
        await self.ReadTask

    async def RecordForLength(self, length: int):
        if self.config.HeadsetConfig is None:
            raise Exception("Headset Config Not Set")

        for i in range(length):
            getDataOutput = self.Unicorn.GetData(
                self.HandleVal, 1, len(self.config.HeadsetConfig.channels)
            )
            if getDataOutput[1] == Unicorn.UnicornReturnStatus.Success:
                # print("GOT DATA")
                # print(getDataOutput)
                yield getDataOutput[0]
            else:
                print("Failed to get Data", getDataOutput[1])

    def RecordContinuously(self, run: Optional[Run] = None):
        if self.config.HeadsetConfig is None:
            raise Exception("Headset Config Not Set")
        while (True and run is None) or run.run:
            getDataOutput = self.Unicorn.GetData(
                self.HandleVal, 1, len(self.config.HeadsetConfig.channels)
            )
            if getDataOutput[1] == Unicorn.UnicornReturnStatus.Success:
                # print("GOT DATA")
                # print(getDataOutput)
                self.OutputQueue.put((getDataOutput[0], self.CurrentState))
            else:
                print("Failed to get Data", getDataOutput[1])

    def WriteThread(self):
        # file: aiofiles.threadpool.text.AsyncTextIOWrapper
        if self.config.HeadsetConfig is None:
            raise Exception("Headset Config Not Set")
        print("WRITE STARTED")
        fileName = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
        if self.config.SubjectID is not None:
            fileName = self.config.SubjectID + "-" + fileName
        if not os.path.exists("RecordedSessions"):
            os.mkdir("RecordedSessions")

        with open(os.path.join("RecordedSessions", f"{fileName}"), "a") as file:
            headers = [x.name for x in self.config.HeadsetConfig.channels]
            headers.append("State")
            file.write(",".join(headers) + "\n")
            while True:
                try:
                    try:
                        data = self.OutputQueue.get()
                    except Exception as e:
                        continue
                    if data[1] == "DONE":
                        return
                    dataCSV = ",".join(str(i) for i in data[0])
                    file.write(f"{dataCSV},{data[1]}\n")
                except Exception as e:
                    print(e)

    async def ExperimentThread(self):
        if self.config is None:
            raise Exception("Device not connected")
        else:
            print(self.config)
        try:
            self.CurrentState = "Wait"
            self.Unicorn.StartAcquisition(self.HandleVal, False)
            run = Run(run=True)
            writeThread = Thread(target=self.WriteThread)
            writeThread.start()
            print("Started Acquisition")
            ReadTask = Thread(target=self.RecordContinuously, args=(run,))
            ReadTask.start()
            self.CurrentState = "WarmUp"
            for i in range(60):
                cv2.waitKey(1000)
            for choice in self.config.ExperimentOrder:
                print("Rest")
                self.CurrentState = "Rest"
                print(self.config.BreakLength / 1000)
                self.PromptViewer.displayNamedPrompt("Rest")
                cv2.waitKey(self.config.BreakLength)
                await asyncio.sleep(0.01)
                # self.Unicorn.StopAcquisition(self.HandleVal)
                self.CurrentState = "Wait"
                mixer.music.play()
                cv2.waitKey(0)
                self.CurrentState = str(choice.value)
                print(choice.value)
                print(self.config.RecordLength / 1000)
                self.PromptViewer.displayNamedPrompt(choice.value)
                cv2.waitKey(self.config.RecordLength)
                await asyncio.sleep(0.01)
                self.CurrentState = "Wait"
                mixer.music.play()
                cv2.waitKey(0)
        except Exception as e:
            print(e)
            exit()
        try:
            print("END")
            run.run = False
            self.OutputQueue.put(([], "DONE"))
            self.Unicorn.StopAcquisition(self.HandleVal)
            self.Unicorn.CloseDevice(self.HandleVal)
        except:
            pass
        finally:
            writeThread.join()
            ReadTask.join()
        return


if __name__ == "__main__":

    recordSets = [
        RecordChoices.Up,
        RecordChoices.Up,
        RecordChoices.Up,
        RecordChoices.Up,
        RecordChoices.Up,
        RecordChoices.Left,
        RecordChoices.Left,
        RecordChoices.Left,
        RecordChoices.Left,
        RecordChoices.Left,
        RecordChoices.Right,
        RecordChoices.Right,
        RecordChoices.Right,
        RecordChoices.Right,
        RecordChoices.Right,
        RecordChoices.Down,
        RecordChoices.Down,
        RecordChoices.Down,
        RecordChoices.Down,
        RecordChoices.Down,
        RecordChoices.Select,
        RecordChoices.Select,
        RecordChoices.Select,
        RecordChoices.Select,
        RecordChoices.Select,
    ]
    np.random.seed(seed=int(time.time()))
    np.random.shuffle(recordSets)
    print(recordSets)

    exp = ExperimentConfig(
        ExperimentOrder=recordSets, SubjectID="NouraKhaled/NouraKhaled"
    )
    
    viewer = PromptViewer(
        [],
        {
            "Up": cv2.imread("RecorderApp/Pictures/Up.png"),
            "Down": cv2.imread("RecorderApp/Pictures/Down.png"),
            "Left": cv2.imread("RecorderApp/Pictures/Left.png"),
            "Right": cv2.imread("RecorderApp/Pictures/Right.png"),
            "Select": cv2.imread("RecorderApp/Pictures/Select.png"),
            "Rest": cv2.imread("RecorderApp/Pictures/Rest.png"),
        },
    )
    expInstance = ExperimentInstance(exp, viewer)
    expInstance.Config()
    cv2.waitKey(0)
    try:
        asyncio.run(expInstance.StartExperiment())
    finally:
        try:
            if hasattr(expInstance, "HandleVal"):
                Unicorn.CloseDevice(expInstance.HandleRef)
        except Exception as e:
            print(e)
