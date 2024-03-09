from enum import Enum

import DLLImporter as unicorn

from dataclasses import dataclass

import ctypes

import time


class UnicornReturnStatus(Enum):
    Success = unicorn.UNICORN_ERROR_SUCCESS
    InvalidParameter = unicorn.UNICORN_ERROR_INVALID_PARAMETER
    BluetoothInitFailed = unicorn.UNICORN_ERROR_BLUETOOTH_INIT_FAILED
    BluetoothSocketFailed = unicorn.UNICORN_ERROR_BLUETOOTH_SOCKET_FAILED
    OpenDeviceFailed = unicorn.UNICORN_ERROR_OPEN_DEVICE_FAILED
    InvalidConfiguration = unicorn.UNICORN_ERROR_INVALID_CONFIGURATION
    BufferOverflow = unicorn.UNICORN_ERROR_BUFFER_OVERFLOW
    BufferUnderflow = unicorn.UNICORN_ERROR_BUFFER_UNDERFLOW
    OperationNotAllowed = unicorn.UNICORN_ERROR_OPERATION_NOT_ALLOWED
    ConnectionProblem = unicorn.UNICORN_ERROR_CONNECTION_PROBLEM
    UnsupportedDevice = unicorn.UNICORN_ERROR_UNSUPPORTED_DEVICE
    InvalidHandle = unicorn.UNICORN_ERROR_INVALID_HANDLE
    GeneralError = unicorn.UNICORN_ERROR_GENERAL_ERROR


@dataclass
class UnicornAmplifierChannel:

    name: str
    unit: str
    range: tuple[float, float]
    enabled: bool


@dataclass
class UnicornAmplifierConfiguration:
    channels: list[UnicornAmplifierChannel]


@dataclass
class UnicornDeviceInformation:
    numberOfEegChannels: int
    serial: str
    firmwareVersion: str
    deviceVersion: str
    pcbVersion: int
    enclosureVersion: int


def GetApiVersion() -> float:
    return unicorn.lib.UNICORN_GetApiVersion()


def GetLastErrorText() -> str:
    return unicorn.lib.UNICORN_GetLastErrorText()


def GetAvailableDevices(rescan: bool) -> tuple[list[str], UnicornReturnStatus]:
    rescanInt = 1 if rescan else 0
    DeviceCount = ctypes.c_uint32(0)
    output = UnicornReturnStatus(
        unicorn.lib.UNICORN_GetAvailableDevices(
            None, ctypes.byref(DeviceCount), rescanInt
        )
    )
    print(output)
    AvailableDevicesCtypes = (unicorn.UNICORN_DEVICE_SERIAL * DeviceCount.value)()
    print(AvailableDevicesCtypes)
    output = UnicornReturnStatus(
        unicorn.lib.UNICORN_GetAvailableDevices(
            AvailableDevicesCtypes, ctypes.byref(DeviceCount), rescan
        )
    )
    return [device.value for device in AvailableDevicesCtypes], output


def OpenDevice(serial: str) -> tuple[int, int, UnicornReturnStatus]:
    handle = unicorn.UNICORN_HANDLE()
    output = UnicornReturnStatus(
        unicorn.lib.UNICORN_OpenDevice(serial.encode(), ctypes.byref(handle))
    )
    return ctypes.byref(handle), handle.value, output


def CloseDevice(handle: int) -> UnicornReturnStatus:
    # handle = unicorn.UNICORN_HANDLE(handle)
    return UnicornReturnStatus(unicorn.lib.UNICORN_CloseDevice(handle))


def StartAcquisition(handle: int, testSignalEnabled: bool) -> UnicornReturnStatus:
    testSignalEnabledInt = 1 if testSignalEnabled else 0
    return UnicornReturnStatus(
        unicorn.lib.UNICORN_StartAcquisition(handle, testSignalEnabledInt)
    )


def StopAcquisition(handle: int) -> UnicornReturnStatus:
    return UnicornReturnStatus(unicorn.lib.UNICORN_StopAcquisition(handle))


def SetConfiguration(
    handle: int, configuration: UnicornAmplifierConfiguration
) -> UnicornReturnStatus:
    configurationCtypes = unicorn.UNICORN_AMPLIFIER_CONFIGURATION()
    for i, channel in enumerate(configuration.channels):
        configurationCtypes.Channels[i].name = channel.name.encode()
        configurationCtypes.Channels[i].unit = channel.unit.encode()
        configurationCtypes.Channels[i].range[0] = channel.range[0]
        configurationCtypes.Channels[i].range[1] = channel.range[1]
        configurationCtypes.Channels[i].enabled = 1 if channel.enabled else 0
    return UnicornReturnStatus(
        unicorn.lib.UNICORN_SetConfiguration(handle, ctypes.byref(configurationCtypes))
    )


def GetConfiguration(
    handle: int,
) -> tuple[UnicornAmplifierConfiguration, UnicornReturnStatus]:
    configuration = UnicornAmplifierConfiguration(
        [
            UnicornAmplifierChannel("", "", (0, 0), False)
            for i in range(unicorn.UNICORN_TOTAL_CHANNELS_COUNT)
        ]
    )
    configurationCtypes = unicorn.UNICORN_AMPLIFIER_CONFIGURATION()
    result = UnicornReturnStatus(
        unicorn.lib.UNICORN_GetConfiguration(handle, ctypes.byref(configurationCtypes))
    )
    for i, channel in enumerate(configurationCtypes.Channels):
        configuration.channels[i].name = channel.name.decode()
        configuration.channels[i].unit = channel.unit.decode()
        configuration.channels[i].range = (channel.range[0], channel.range[1])
        configuration.channels[i].enabled = True if channel.enabled == 1 else False
    return configuration, result


def GetData(
    handle: int, ScanCount: int, OutputLength: int
) -> tuple[list[float], UnicornReturnStatus]:
    Destination = (ctypes.c_float * OutputLength)()
    output = UnicornReturnStatus(
        unicorn.lib.UNICORN_GetData(handle, ScanCount, Destination, OutputLength)
    )
    return list(Destination), output


def GetNumberOfAcquiredChannels(handle: int) -> tuple[int, UnicornReturnStatus]:
    NumberOfAcquiredChannels = ctypes.c_uint32()
    output = UnicornReturnStatus(
        unicorn.lib.UNICORN_GetNumberOfAcquiredChannels(
            handle, ctypes.byref(NumberOfAcquiredChannels)
        )
    )
    return NumberOfAcquiredChannels.value, output


def GetChannelIndex(handle: int, ChannelName: str) -> tuple[int, UnicornReturnStatus]:
    ChannelIndex = ctypes.c_uint32()
    output = UnicornReturnStatus(
        unicorn.lib.UNICORN_GetChannelIndex(
            handle, ChannelName.encode(), ctypes.byref(ChannelIndex)
        )
    )
    return ChannelIndex.value, output


def GetDeviceInformation(
    handle: int,
) -> tuple[UnicornDeviceInformation, UnicornReturnStatus]:
    DeviceInformation = unicorn.UNICORN_DEVICE_INFORMATION()
    output = UnicornReturnStatus(
        unicorn.lib.UNICORN_GetDeviceInformation(
            handle, ctypes.byref(DeviceInformation)
        )
    )
    return (
        UnicornDeviceInformation(
            DeviceInformation.numberOfEegChannels,
            DeviceInformation.serial.decode(),
            DeviceInformation.firmwareVersion.decode(),
            DeviceInformation.deviceVersion.decode(),
            DeviceInformation.pcbVersion,
            DeviceInformation.enclosureVersion,
        ),
        output,
    )


def SetDigitalOutput(handle: int) -> tuple[int, UnicornReturnStatus]:
    DigitalOutput = ctypes.c_uint8()
    output = UnicornReturnStatus(
        unicorn.lib.UNICORN_SetDigitalOutputs(handle, ctypes.byref(DigitalOutput))
    )
    return DigitalOutput.value, output


if __name__ == "__main__":
    output = OpenDevice("UN-2021.12.19")
    print(output)
    try:
        ChannelConfig = GetConfiguration(output[1])
        print(ChannelConfig)
        startOutput = StartAcquisition(output[1], True)
        print(startOutput)
        numberOfGetDataCalls = 10 * unicorn.UNICORN_SAMPLING_RATE
        for i in range(numberOfGetDataCalls):
            getDataOutput = GetData(output[1], 1, len(ChannelConfig[0].channels))
            print(getDataOutput)
        stopOutput = StopAcquisition(output[1])
        print(stopOutput)
    except Exception as e:
        print(e)
    finally:
        close = CloseDevice(output[1])
        print(close)
