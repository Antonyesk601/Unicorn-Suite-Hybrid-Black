import cv2
import numpy as np
from types import NoneType
from typing import Optional
from enum import Enum, StrEnum
from dataclasses import dataclass, asdict
import sys
import os
import asyncio
from asyncio.queues import Queue
import aiofiles
import uvloop
import datetime
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
        _ =cv2.waitKey(1) & 0xFF == ord("0")

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
    RecordLength: int = 3000
    BreakLength: int = 5000
    HeadsetConfig: Optional[Unicorn.UnicornAmplifierConfiguration] = None
    AudioQueuePath: Optional[str] = None
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
        mixer.music.play()
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
        except:
            pass

    async def StartExperiment(self):
        self.ReadTask = asyncio.create_task(self.ExperimentThread())
        await self.ReadTask

    async def RecordForLength(self, length: int):
        if self.config.HeadsetConfig is None:
            raise Exception("Headset Config Not Set")
        
        for i in range(length):
            getDataOutput = self.Unicorn.GetData(
                self.HandleVal, 1, len(self.config.HeadsetConfig.channels)
            )
            if getDataOutput[1] == Unicorn.UnicornReturnStatus.Success:
                yield getDataOutput[0]
            else:
                raise Exception("Failed to get Data", getDataOutput[1])

    async def WriteThread(self):
        file: aiofiles.threadpool.text.AsyncTextIOWrapper
        fileName = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv"
        async with aiofiles.open(fileName, mode="a") as file:
            while True:
                data = await self.OutputQueue.get()
                dataCSV = ",".join(str(i) for i in data[0])
                await file.write(f"{dataCSV},{data[1]}\n")

    async def ExperimentThread(self):
        self.Unicorn.StartAcquisition(self.HandleVal, True)
        writeThread = asyncio.create_task(self.WriteThread())
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
                await self.OutputQueue.put((data, "Rest"))
            mixer.music.play()
            self.Unicorn.StopAcquisition(self.HandleVal)
            cv2.waitKey(0)
            self.Unicorn.StartAcquisition(self.HandleVal, True)

            print(choice.value)
            print(self.config.RecordLength / 1000)
            self.PromptViewer.displayNamedPrompt(choice.value)
            async for data in self.RecordForLength(RecordDataCalls):
                await self.OutputQueue.put((data, choice.value))
            mixer.music.play()
            self.Unicorn.StopAcquisition(self.HandleVal)
            cv2.waitKey(0)
            self.Unicorn.StartAcquisition(self.HandleVal, True)


        try:
            print("END")
            self.Unicorn.StopAcquisition(self.HandleVal)
            self.Unicorn.CloseDevice(self.HandleVal)
            await writeThread
        except:
            pass


if __name__ == "__main__":

    exp = ExperimentConfig(
        ExperimentOrder=[
            RecordChoices.Up,
            RecordChoices.Down,
            RecordChoices.Left,
            RecordChoices.Right,
            RecordChoices.Select,
            RecordChoices.Rest,
        ]
    )
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
    asyncio.run(expInstance.StartExperiment())
