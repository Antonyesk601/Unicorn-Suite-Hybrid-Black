import ctypes

lib = ctypes.CDLL("PythonWrapper/libunicorn.so")

BOOL = ctypes.c_uint32

FALSE = 0

TRUE = 1

UNICORN_ERROR_SUCCESS = 0

UNICORN_ERROR_INVALID_PARAMETER = 1
UNICORN_ERROR_BLUETOOTH_INIT_FAILED = 2
UNICORN_ERROR_BLUETOOTH_SOCKET_FAILED = 3
UNICORN_ERROR_OPEN_DEVICE_FAILED = 4
UNICORN_ERROR_INVALID_CONFIGURATION = 5
UNICORN_ERROR_BUFFER_OVERFLOW = 6
UNICORN_ERROR_BUFFER_UNDERFLOW = 7
UNICORN_ERROR_OPERATION_NOT_ALLOWED = 8
UNICORN_ERROR_CONNECTION_PROBLEM = 9
UNICORN_ERROR_UNSUPPORTED_DEVICE = 10
UNICORN_ERROR_INVALID_HANDLE = 0xFFFFFFFE
UNICORN_ERROR_GENERAL_ERROR = 0xFFFFFFFF
UNICORN_SERIAL_LENGTH_MAX = 14
UNICORN_DEVICE_VERSION_LENGTH_MAX = 6
UNICORN_FIRMWARE_VERSION_LENGTH_MAX = 12
UNICORN_SAMPLING_RATE = 250
UNICORN_STRING_LENGTH_MAX = 255
UNICORN_EEG_CHANNELS_COUNT = 8
UNICORN_ACCELEROMETER_CHANNELS_COUNT = 3
UNICORN_GYROSCOPE_CHANNELS_COUNT = 3
UNICORN_TOTAL_CHANNELS_COUNT = 17
UNICORN_EEG_CONFIG_INDEX = 0
UNICORN_ACCELEROMETER_CONFIG_INDEX = 8
UNICORN_GYROSCOPE_CONFIG_INDEX = 11
UNICORN_BATTERY_CONFIG_INDEX = 14
UNICORN_COUNTER_CONFIG_INDEX = 15
UNICORN_VALIDATION_CONFIG_INDEX = 16
UNICORN_NUMBER_OF_DIGITAL_OUTPUTS = 8

UNICORN_HANDLE = ctypes.c_uint64

UNICORN_DEVICE_SERIAL = ctypes.c_char * UNICORN_SERIAL_LENGTH_MAX

UNICORN_DEVICE_VERSION = ctypes.c_char * UNICORN_DEVICE_VERSION_LENGTH_MAX

UNICORN_FIRMWARE_VERSION = ctypes.c_char * UNICORN_FIRMWARE_VERSION_LENGTH_MAX


class UNICORN_AMPLIFIER_CHANNEL(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 32),
        ("unit", ctypes.c_char * 32),
        ("range", ctypes.c_float * 2),
        ("enabled", BOOL),
    ]


class UNICORN_AMPLIFIER_CONFIGURATION(ctypes.Structure):
    _fields_ = [
        ("Channels", UNICORN_AMPLIFIER_CHANNEL * UNICORN_TOTAL_CHANNELS_COUNT),
    ]


class UNICORN_DEVICE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("numberOfEegChannels", ctypes.c_uint16),
        ("serial", UNICORN_DEVICE_SERIAL),
        ("firmwareVersion", UNICORN_FIRMWARE_VERSION),
        ("deviceVersion", UNICORN_DEVICE_VERSION),
        ("pcbVersion", ctypes.c_uint8),
        ("enclosureVersion", ctypes.c_uint8),
    ]


lib.UNICORN_GetApiVersion.restype = ctypes.c_float
lib.UNICORN_GetApiVersion.argtypes = []

lib.UNICORN_GetLastErrorText.restype = ctypes.c_char_p
lib.UNICORN_GetLastErrorText.argtypes = []

lib.UNICORN_GetAvailableDevices.restype = ctypes.c_int
lib.UNICORN_GetAvailableDevices.argtypes = [
    ctypes.POINTER(UNICORN_DEVICE_SERIAL),
    ctypes.POINTER(ctypes.c_uint32),
    BOOL,
]

lib.UNICORN_OpenDevice.restype = ctypes.c_int
lib.UNICORN_OpenDevice.argtypes = [ctypes.c_char_p, ctypes.POINTER(UNICORN_HANDLE)]

lib.UNICORN_CloseDevice.restype = ctypes.c_int
lib.UNICORN_CloseDevice.argtypes = [ctypes.POINTER(UNICORN_HANDLE)]

lib.UNICORN_StartAcquisition.restype = ctypes.c_int
lib.UNICORN_StartAcquisition.argtypes = [UNICORN_HANDLE, BOOL]

lib.UNICORN_StopAcquisition.restype = ctypes.c_int
lib.UNICORN_StopAcquisition.argtypes = [UNICORN_HANDLE]

lib.UNICORN_SetConfiguration.restype = ctypes.c_int
lib.UNICORN_SetConfiguration.argtypes = [
    UNICORN_HANDLE,
    ctypes.POINTER(UNICORN_AMPLIFIER_CONFIGURATION),
]

lib.UNICORN_GetConfiguration.restype = ctypes.c_int
lib.UNICORN_GetConfiguration.argtypes = [
    UNICORN_HANDLE,
    ctypes.POINTER(UNICORN_AMPLIFIER_CONFIGURATION),
]

lib.UNICORN_GetData.restype = ctypes.c_int
lib.UNICORN_GetData.argtypes = [
    UNICORN_HANDLE,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint32,
]

lib.UNICORN_GetNumberOfAcquiredChannels.restype = ctypes.c_int
lib.UNICORN_GetNumberOfAcquiredChannels.argtypes = [
    UNICORN_HANDLE,
    ctypes.POINTER(ctypes.c_uint32),
]

lib.UNICORN_GetChannelIndex.restype = ctypes.c_int
lib.UNICORN_GetChannelIndex.argtypes = [
    UNICORN_HANDLE,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint32),
]

lib.UNICORN_GetDeviceInformation.restype = ctypes.c_int
lib.UNICORN_GetDeviceInformation.argtypes = [
    UNICORN_HANDLE,
    ctypes.POINTER(UNICORN_DEVICE_INFORMATION),
]

lib.UNICORN_SetDigitalOutputs.restype = ctypes.c_int
lib.UNICORN_SetDigitalOutputs.argtypes = [
    UNICORN_HANDLE,
    ctypes.POINTER(ctypes.c_uint8),
]
