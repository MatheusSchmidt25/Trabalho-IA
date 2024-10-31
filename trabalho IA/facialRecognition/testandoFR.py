import cv2

# Carrega as imagens
imgElon = cv2.imread('Elon.jpg')
imgElonTest = cv2.imread('ElonTest.jpg')
imgTony = cv2.imread('Tony.jpg')
imgPessoas = cv2.imread('a.jpg')

# Converte para escala de cinza (necessário para detecção de rosto com Haar Cascade)
grayElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2GRAY)
grayElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2GRAY)
grayTony = cv2.cvtColor(imgTony, cv2.COLOR_BGR2GRAY)
grayPessoas = cv2.cvtColor(imgPessoas, cv2.COLOR_BGR2GRAY)

# Carrega o classificador de rosto Haar Cascade (pré-treinado do OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detecta rostos nas duas imagens
facesElon = face_cascade.detectMultiScale(grayElon, scaleFactor=1.1, minNeighbors=5)
facesElonTest = face_cascade.detectMultiScale(grayElonTest, scaleFactor=1.1, minNeighbors=5)
facesTonyy = face_cascade.detectMultiScale(grayTony, scaleFactor=1.1, minNeighbors=5)
facesPessoas = face_cascade.detectMultiScale(grayPessoas, scaleFactor=1.1, minNeighbors=5)


# Desenha retângulos ao redor dos rostos detectados
for (x, y, w, h) in facesElon:
    cv2.rectangle(imgElon, (x, y), (x+w, y+h), (255, 0, 0), 2)

for (x, y, w, h) in facesElonTest:
    cv2.rectangle(imgElonTest, (x, y), (x+w, y+h), (255, 0, 0), 2)

for (x, y, w, h) in facesTonyy:
    cv2.rectangle(imgTony, (x, y), (x+w, y+h), (255, 0, 0), 2)  

for (x, y, w, h) in facesPessoas:
    cv2.rectangle(imgPessoas, (x, y), (x+w, y+h), (255, 0, 0), 2)  

# Exibe as imagens com os rostos detectados
cv2.imshow('Elon', imgElon)
cv2.imshow('Elon Test', imgElonTest)
cv2.imshow('Tony',imgTony)
cv2.imshow('Pessoas',imgPessoas)
cv2.waitKey(0)
cv2.destroyAllWindows()
