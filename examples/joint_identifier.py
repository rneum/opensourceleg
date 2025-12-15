import glob

from opensourceleg.actuators.dephy import DephyActuator

# get list of all usb connections
ports = glob.glob("/dev/ttyACM*")

# for now, expect exactly two dephy actuators on ttyACM
if len(ports) > 2:
    print("more than two devices detected")
else:
    for i in ports:
        actuatorTemp = DephyActuator(port=i)
        if actuatorTemp.id == 1439:
            knee = actuatorTemp
        elif actuatorTemp.id == 1423:
            ankle = actuatorTemp
        else:
            print(actuatorTemp.id)
            print("I've never seen that actuator in my life")
