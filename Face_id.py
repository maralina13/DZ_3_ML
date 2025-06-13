import cv2
import mediapipe as mp
import os
import numpy as np
from scipy.spatial import distance

# Настройки
USER_IMAGE_PATH = "my_face.jpg"  # Ваше фото в той же папке
YOUR_NAME = "Alina"
YOUR_SURNAME = "Martynova"

# Инициализация MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class EnhancedFaceRecognizer:
    def __init__(self):
        self.user_landmarks = None
        self.check_user_photo()
        self.setup_detectors()

    def check_user_photo(self):
        """Проверка и загрузка фото пользователя"""
        print("\n=== ПРОВЕРКА ФОТО ===")

        if not os.path.exists(USER_IMAGE_PATH):
            print(f"❌ Ошибка: Файл '{USER_IMAGE_PATH}' не найден!")
            print("Поместите ваше фото в ту же папку, что и программу")
            exit()

        user_photo = cv2.imread(USER_IMAGE_PATH)
        if user_photo is None:
            print(f"❌ Ошибка: Не удалось загрузить '{USER_IMAGE_PATH}'")
            print("Убедитесь, что это изображение в формате JPG/PNG")
            exit()

        # Извлекаем ключевые точки с фото
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(user_photo, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                print("❌ Ошибка: На фото не найдено лицо!")
                exit()

            self.user_landmarks = results.multi_face_landmarks[0]
            print(f"✅ На фото найдено лицо с {len(self.user_landmarks.landmark)} точками")

    def setup_detectors(self):
        """Инициализация детекторов"""
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )

        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8
        )

    def compare_faces(self, current_landmarks):
        """Сравнение лиц по ключевым точкам"""
        if not self.user_landmarks or not current_landmarks:
            return False

        # Используем только основные надежные точки
        key_points = [
            10, 152,  # Подбородок и лоб
            234, 454,  # Виски
            33, 263,  # Нос
            133, 362,  # Глаза
            61, 291,  # Щеки
            0, 17,  # Границы лица
        ]

        total_distance = 0
        valid_points = 0

        for i in key_points:
            if i < len(self.user_landmarks.landmark) and i < len(current_landmarks.landmark):
                user_point = np.array([self.user_landmarks.landmark[i].x,
                                       self.user_landmarks.landmark[i].y])
                current_point = np.array([current_landmarks.landmark[i].x,
                                          current_landmarks.landmark[i].y])

                total_distance += distance.euclidean(user_point, current_point)
                valid_points += 1

        if valid_points == 0:
            return False

        avg_distance = total_distance / valid_points
        return avg_distance < 0.05

    def detect_emotion(self, landmarks):
        """Определение эмоций по ключевым точкам"""
        try:
            # Индексы ключевых точек
            mouth_top = landmarks.landmark[13].y
            mouth_bottom = landmarks.landmark[14].y
            left_eyebrow = landmarks.landmark[65].y
            right_eyebrow = landmarks.landmark[295].y

            mouth_openness = mouth_bottom - mouth_top
            eyebrow_avg = (left_eyebrow + right_eyebrow) / 2

            if mouth_openness > 0.06:
                return "SURPRISED"
            elif mouth_openness > 0.03 and eyebrow_avg < 0.35:
                return "HAPPY"
            elif eyebrow_avg > 0.4:
                return "SAD"
            else:
                return "NEUTRAL"
        except:
            return "UNKNOWN"

    def count_fingers(self, landmarks):
        """Точный подсчет пальцев"""
        tips = [8, 12, 16, 20]  # Кончики пальцев
        mcp = [6, 10, 14, 18]  # Основания пальцев
        raised = 0

        for tip, base in zip(tips, mcp):
            if landmarks.landmark[tip].y < landmarks.landmark[base].y:
                raised += 1

        # Большой палец
        if landmarks.landmark[4].x < landmarks.landmark[3].x:
            raised += 1

        return raised


def main():
    print("\n=== ЗАПУСК СИСТЕМЫ РАСПОЗНАВАНИЯ ===")
    print("1. Убедитесь, что освещение хорошее")
    print("2. Смотрите прямо в камеру")
    print("3. Для выхода нажмите Q\n")

    recognizer = EnhancedFaceRecognizer()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Детекция рук
        finger_count = 0
        hand_results = recognizer.hands_detector.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                finger_count = recognizer.count_fingers(hand)

        # Детекция лиц и эмоций
        face_results = recognizer.face_mesh.process(rgb_frame)
        emotion = ""

        if face_results.multi_face_landmarks:
            current_landmarks = face_results.multi_face_landmarks[0]
            emotion = recognizer.detect_emotion(current_landmarks)

            # Получаем ограничивающий прямоугольник
            h, w = frame.shape[:2]
            landmarks_array = np.array([(lm.x * w, lm.y * h)
                                        for lm in current_landmarks.landmark])
            x, y = landmarks_array.min(axis=0).astype(int)
            x2, y2 = landmarks_array.max(axis=0).astype(int)

            # Сравнение лиц
            is_you = recognizer.compare_faces(current_landmarks)

            if is_you:
                color = (0, 255, 0)  # Зеленый
                if finger_count == 1:
                    text = YOUR_NAME  # Показываем имя при 1 пальце
                elif finger_count == 2:
                    text = YOUR_SURNAME  # Показываем фамилию при 2 пальцах
                elif finger_count == 3:
                    text = f"Emotion: {emotion}"  # Показываем эмоцию при 3 пальцах
                else:
                    text = "ME"  # Базовый текст
            else:
                color = (0, 0, 255)  # Красный
                text = "UNKNOWN"

            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Статус (только для отладки)
        debug_info = f"Fingers: {finger_count} | Press Q to quit"
        cv2.putText(frame, debug_info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Face Recognition System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== ПРОГРАММА ЗАВЕРШЕНА ===")


if __name__ == "__main__":
    main()