import airsim

client = airsim.VehicleClient()
client.confirmConnection()

client.simSetCameraOrientation(1, airsim.to_quaternion(0,0,1)); #radians	
client.simSetCameraOrientation(2, airsim.to_quaternion(0,0,(44/7)-1)); #radians	