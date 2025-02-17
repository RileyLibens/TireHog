from pymodbus.client.sync import ModbusSerialClient

client = ModbusSerialClient(method="rtu", port="COM1", baudrate = 9600, timeout=3)
client.connect()

read = client.read_holding_registers(address=0, unit=1)
read.registers

if(read.isError()):
    raise Exception("Error reading the registers")
else:
    data = read

client.close()
