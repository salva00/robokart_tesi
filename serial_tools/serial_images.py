import serial
from PIL import Image
import io

# Configurazione della porta seriale
SERIAL_PORT = '/dev/cu.usbserial-A9MPHFJJ'  # Modifica con il nome della tua porta seriale
BAUD_RATE = 115200  # Modifica con il baud rate corretto
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
BYTES_PER_PIXEL = 1  # Ad esempio, 1 se l'immagine è in formato grayscale (8-bit)

# Calcola la dimensione totale dell'immagine
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL

# Apri la connessione seriale
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)


def receive_image():
    print("In attesa dell'immagine...")
    # Leggi tutti i byte necessari
    image_data = ser.read(IMAGE_SIZE)
    print("Immagine ricevuta!")

    # Crea un oggetto immagine da PIL
    image = Image.frombytes('L', (IMAGE_WIDTH, IMAGE_HEIGHT), image_data)
    return image



try:
    # Ricevi l'immagine e mostrala
    image = receive_image()
    image.show()
except Exception as e:
    print(f"Errore: {e}")
finally:
    # Chiudi la connessione seriale
    ser.close()