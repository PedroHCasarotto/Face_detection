import cv2

# Carregando o classificador pré-treinado para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializando a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    # Lendo o quadro da captura de vídeo
    ret, frame = cap.read()

    # Convertendo para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectando rostos na imagem em escala de cinza
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Desenhando retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Exibindo o resultado em uma janela
    cv2.imshow('Detecção de Rosto', frame)

    # Encerrando o loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberando os recursos
cap.release()
cv2.destroyAllWindows()
