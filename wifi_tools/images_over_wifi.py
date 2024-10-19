
import socket
# Configura il server UDP
server_ip = '192.168.8.201'  # Ascolta su tutte le interfacce di rete
server_port = 8080
buffer_size = 1024
image_size = 640 * 480  # 307200 byte

# Crea un socket UDP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((server_ip, server_port))

print("Server UDP in attesa dei pacchetti immagine...")

# Ricevi i dati dell'immagine
received_data = bytearray()
while len(received_data) < image_size:
    packet, addr = server_socket.recvfrom(buffer_size)
    received_data.extend(packet)

# Salva l'immagine ricevuta in un file
with open("received_image_udp.raw", "wb") as image_file:
    image_file.write(received_data)

print("Immagine ricevuta e salvata.")

# Chiudi il socket UDP
server_socket.close()
