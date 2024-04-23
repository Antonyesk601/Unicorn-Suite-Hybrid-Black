import Unicorn

def HandleError(status:Unicorn.UnicornReturnStatus):
    if status != Unicorn.UnicornReturnStatus.Success:
        print("Error: " + Unicorn.Unicorn.GetLastErrorText())
        exit()

def printErrorMessage(status:Unicorn.UnicornReturnStatus):
    if status != Unicorn.UnicornReturnStatus.Success:
        print("Error: " + Unicorn.Unicorn.GetLastErrorText())
    
def Main(): 
    print("Unicorn API version: " + str(Unicorn.Unicorn.GetApiVersion()))
    print("-"*10)
    programErrorStatus = Unicorn.UnicornReturnStatus.Success
    deviceHandle = 0
    try:
        availableDevices, programErrorStatus = Unicorn.Unicorn.GetAvailableDevices(True)
        if len(availableDevices) < 1 :
            print("No device available. Please pair with a Unicorn device first.")
            programErrorStatus = Unicorn.UnicornReturnStatus.GeneralError
            return
        availableDevices, programErrorStatus : tuple[list[str],Unicorn.UnicornReturnStatus] = Unicorn.Unicorn.GetAvailableDevices(True)
        HandleError(programErrorStatus)
        print("Available devices: " )
        print(*availableDevices,sep="\n")
        print("Select Device By ID: ")
        deviceID = int(input())
        if deviceID < 0 or deviceID >= len(availableDevices):
            print("Invalid device ID.")
            programErrorStatus = Unicorn.UnicornReturnStatus.GeneralError
            HandleError(programErrorStatus)
            return
        with open("out.csv", "w") as outputFile:
            try:
                numOfchannelsToAcquire, programErrorStatus = Unicorn.Unicorn.GetNumberOfAcquiredChannels(deviceHandle)
                HandleError(programErrorStatus)

                ampConfig, programErrorStatus = Unicorn.Unicorn.GetConfiguration(deviceHandle)
                HandleError(programErrorStatus)

                print("\n Acquisiton Configuration: ")
                print("Frame Length: " , 1)
                print("Number of Channels: " , numOfchannelsToAcquire)
                print("Data Acquisition Length: ", 10.0, "s")

                acquisitionBuffer = [0.0] * numOfchannelsToAcquire
                programErrorStatus = Unicorn.Unicorn.StartAcquisition(deviceHandle, False)
                HandleError(programErrorStatus)
                
                print("Acquisition started.")
                
            except:
                pass
    except Exception as e:
        print(e)
