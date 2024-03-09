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
from random import shuffle
from pygame import mixer

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
    BreakLength: int = 5000
    HeadsetConfig: Optional[Unicorn.UnicornAmplifierConfiguration] = None
    SubjectID: Optional[str] = None
    AudioFile: str = "RecorderApp/Signal.mp3"
    HeadsetSerial: str = "UN-2021.12.19"


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
                raise Exception("Failed to get Data", getDataOutput[1])

    def WriteThread(self):
        # file: aiofiles.threadpool.text.AsyncTextIOWrapper
        print("WRITE STARTED")
        fileName = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
        if self.config.SubjectID is not None:
            fileName = self.config.SubjectID + "-" + fileName
        if not os.path.exists("RecordedSessions"):
            os.mkdir("RecordedSessions")
        with open(f"RecordedSessions/{fileName}", "a") as file:
            headers = [x.name for x in self.config.HeadsetConfig.channels]
            file.write(",".join(headers) + "\n")
            while True:
                try:
                    try:
                        data = self.OutputQueue.get()
                    except Exception as e:
                        continue
                    if data == "DONE":
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
        self.Unicorn.StartAcquisition(self.HandleVal, True)
        writeThread = Thread(target=self.WriteThread)
        writeThread.start()
        print("Started Acquisition")
        RecordDataCalls = int(
            self.config.RecordLength / 1000 * Unicorn.unicorn.UNICORN_SAMPLING_RATE
        )
        RestDataCalls = int(
            self.config.BreakLength / 1000 * Unicorn.unicorn.UNICORN_SAMPLING_RATE
        )

        for choice in self.config.ExperimentOrder:
            print("Rest")
            print(self.config.BreakLength / 1000)
            self.PromptViewer.displayNamedPrompt("Rest")
            async for data in self.RecordForLength(RestDataCalls):
                self.OutputQueue.put((data, "Rest"))
            self.Unicorn.StopAcquisition(self.HandleVal)
            mixer.music.play()
            cv2.waitKey(0)
            self.Unicorn.StartAcquisition(self.HandleVal, True)

            print(choice.value)
            print(self.config.RecordLength / 1000)
            self.PromptViewer.displayNamedPrompt(choice.value)
            async for data in self.RecordForLength(RecordDataCalls):
                self.OutputQueue.put((data, choice.value))
            mixer.music.play()
            self.Unicorn.StopAcquisition(self.HandleVal)
            cv2.waitKey(0)
            self.Unicorn.StartAcquisition(self.HandleVal, True)

        try:
            print("END")
            self.OutputQueue.put("DONE")
            self.Unicorn.StopAcquisition(self.HandleVal)
            self.Unicorn.CloseDevice(self.HandleVal)
        except:
            pass
        finally:
            writeThread.join()
        return


if __name__ == "__main__":

    recordSets = [
        RecordChoices.Up,
        RecordChoices.Left,
        RecordChoices.Right,
        RecordChoices.Down,
        RecordChoices.Select,
        RecordChoices.Down,
        RecordChoices.Down,
        RecordChoices.Left,
        RecordChoices.Left,
        RecordChoices.Select,
        RecordChoices.Select,
        RecordChoices.Up,
        RecordChoices.Up,
        RecordChoices.Right,
        RecordChoices.Right,
    ]
    shuffle(recordSets)

    exp = ExperimentConfig(ExperimentOrder=recordSets, SubjectID="Antony")
    viewer = PromptViewer(
        [],
        {
            "Up": cv2.imread("RecorderApp/Up.png"),
            "Down": cv2.imread("RecorderApp/Down.png"),
            "Left": cv2.imread("RecorderApp/Left.png"),
            "Right": cv2.imread("RecorderApp/Right.png"),
            "Select": cv2.imread("RecorderApp/Select.png"),
            "Rest": cv2.imread("RecorderApp/Rest.png"),
        },
    )
    expInstance = ExperimentInstance(exp, viewer)
    expInstance.Config()
    try:
        asyncio.run(expInstance.StartExperiment())
    finally:
        if hasattr(expInstance, "HandleVal"):
            Unicorn.CloseDevice(expInstance.HandleRef)
