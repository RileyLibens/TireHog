from pymodbus.client.sync import ModbusTcpClient
client = ModbusTcpClient(" ", port=, timeout=3)
client.connect()
read=client.read_holding_registers(address = NEED TO BE FILLED)
read.registers
data = read
