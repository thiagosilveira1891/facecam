
import cv2
from fer.fer import FER
import threading
import time

EMOCIONES_ES = {
    "angry": "Enojado",
    "disgust": "Asco",
    "fear": "Miedo",
    "happy": "Feliz",
    "sad": "Triste",
    "surprise": "Sorprendido",
    "neutral": "Neutral",
}

COLORES_EMOCION = {
    "angry": (0, 0, 255),
    "disgust": (0, 140, 255),
    "fear": (255, 0, 255),
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "surprise": (0, 255, 255),
    "neutral": (200, 200, 200),
}

frame_to_process = None
ultimos_resultados = []
thread_running = True

def emotion_worker():
    global frame_to_process, ultimos_resultados, thread_running
    
    print("Hilo de IA: Cargando modelo neuronal (MTCNN)...")
    detector = FER(mtcnn=True)
    print("Hilo de IA: Modelo listo.")

    while thread_running:
        if frame_to_process is not None:
            frame = frame_to_process.copy()
            frame_to_process = None
            
            try:
                resultados = detector.detect_emotions(frame)
                ultimos_resultados = resultados
            except Exception:
                pass
        else:
            time.sleep(0.01)

def main():
    global frame_to_process, ultimos_resultados, thread_running

    print("Iniciando cámara...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("¡Cámara lista! Escáner facial de emociones iniciado.")
    
    ia_thread = threading.Thread(target=emotion_worker)
    ia_thread.start()

    print("Aguarde un momento para que aparezca la ventana...")
    print("Presiona 'q' o 'ESC' en la ventana del video para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        frame = cv2.flip(frame, 1)

        if frame_to_process is None:
            frame_to_process = frame

        for rostro in ultimos_resultados:
            (x, y, w, h) = rostro["box"]
            emociones = rostro["emotions"]

            emocion_dominante = max(emociones, key=emociones.get)
            confianza = emociones[emocion_dominante]

            if confianza < 0.10:
                continue

            color = COLORES_EMOCION.get(emocion_dominante, (255, 255, 255))
            nombre_es = EMOCIONES_ES.get(emocion_dominante, emocion_dominante)
            porcentaje = int(confianza * 100)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            texto = f"{nombre_es} ({porcentaje}%)"

            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
            cv2.putText(frame, texto, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            barra_y = y + h + 5
            for i, (emo, valor) in enumerate(emociones.items()):
                nombre = EMOCIONES_ES.get(emo, emo)[:3]
                barra_ancho = int(valor * 100)
                emo_color = COLORES_EMOCION.get(emo, (200, 200, 200))

                cv2.rectangle(frame, (x, barra_y + i * 18),
                              (x + barra_ancho, barra_y + i * 18 + 14),
                              emo_color, -1)
                cv2.putText(frame, nombre, (x + barra_ancho + 5, barra_y + i * 18 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(frame, "Escaner de Emociones", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Presiona 'q' para salir", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Escaner Facial de Emociones", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == 27:
            thread_running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    ia_thread.join()
    print("Escáner cerrado.")

if __name__ == "__main__":
    main()
