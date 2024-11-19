import math
import cv2
import mediapipe as mp
import pyautogui
import time
from googlesearch import search
import speech_recognition as sr
# Tentar esse comando para rodar o pyaoutgui:
# export DISPLAY=:0
# xhost +SI:localuser:$(whoami)


class Ponto:
    def __init__(self,x,y):
        self.x = x
        self.y = y
# Classe para calcular distâncias
class CalculadoraDistancia:
    @staticmethod
    def calcular_2d(ponto1, ponto2):
        """
        Calcula a distância 2D entre dois pontos.
        :param ponto1: Coordenadas (x, y) do primeiro ponto.
        :param ponto2: Coordenadas (x, y) do segundo ponto.
        :return: Distância em unidades normalizadas.
        """
        ponto1 = Ponto(ponto1[0],ponto1[1])
        ponto2 = Ponto(ponto2[0],ponto2[1])
        
        distancia_x = abs(ponto1.x-ponto2.x)
        distancia_y = abs(ponto1.y-ponto2.y)
        
        pontoCentral = Ponto(((ponto1.x + ponto2.x) / 2),((ponto1.y + ponto2.y) / 2))
        
        angulo = math.atan2((ponto1.y - ponto2.y),(ponto1.x-ponto2.x))
        angulo_oposto = angulo + math.pi
        
        distancia_adicional = 100
        # Calculando as novas coordenadas (ponto oposto)
        novo_ponto_x = abs(pontoCentral.x * math.cos(angulo_oposto))
        novo_ponto_y = abs(pontoCentral.y * math.sin(angulo_oposto))
        ponto_referencia = Ponto(novo_ponto_x,novo_ponto_y)
        # Vetor diretor do segmento
        # dx = x2 - x1
        # dy = y2 - y1
        
        # Vetor perpendicular
        # perp_dx = -dy
        # perp_dy = dx
        
        return math.sqrt((ponto2.x - ponto1.x)**2 + (ponto2.y - ponto1.y)**2),ponto_referencia

# Classe para representar um dedo
class Dedo:
    def __init__(self, nome, ponta, dobradiça, metacarpo):
        self.nome = nome
        self.ponta = ponta
        self.dobradiça = dobradiça
        self.metacarpo = metacarpo

    def distancia_para(self, outro_ponto, ponto_referido="ponta"):
        """
        Calcula a distância 2D entre o ponto referido do dedo e outro ponto.
        :param outro_ponto: Coordenadas (x, y) do outro ponto.
        :param ponto_referido: O ponto do dedo a ser usado para o cálculo ("ponta", "dobradiça" ou "metacarpo").
        :return: Distância em unidades normalizadas.
        """
        ponto_destino = (outro_ponto.x,outro_ponto.y)
        pontos = {
            "ponta": (self.ponta.x, self.ponta.y),
            "dobradiça": (self.dobradiça.x, self.dobradiça.y),
            "metacarpo": (self.metacarpo.x, self.metacarpo.y)
        }
        
        if ponto_referido not in pontos:
            raise ValueError(f"Ponto referido '{ponto_referido}' não é válido. Use 'ponta', 'dobradiça' ou 'metacarpo'.")

        ponto_atual = pontos[ponto_referido]
        return CalculadoraDistancia.calcular_2d(ponto_atual, ponto_destino)

# Classe para representar uma mão com seus dedos
class Mao:
    def __init__(self,hand_landmarks): # como deve ser
        self.polegar = Dedo(
            "Polegar",
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP]
        )
        self.indicador = Dedo(
            "Indicador",
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
        )
        self.medio = Dedo(
            "Médio",
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
        )
        self.anelar = Dedo(
            "Anelar",
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
        )
        self.mindinho = Dedo(
            "Mindinho",
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]
        )

#Funcao responsavel por ouvir e reconhecer a fala
def ouvir_microfone():
    #Habilita o microfone para ouvir o usuario
    microfone = sr.Recognizer()
    with sr.Microphone() as source:
        #Chama a funcao de reducao de ruido disponivel na speech_recognition
        microfone.adjust_for_ambient_noise(source)
        #Avisa ao usuario que esta pronto para ouvir
        print("Diga o que quer pesquisar: ")
        #Armazena a informacao de audio na variavel
        audio = microfone.listen(source,timeout=10,phrase_time_limit=None)
        try:
            #Passa o audio para o reconhecedor de padroes do speech_recognition
            frase = microfone.recognize_google(audio,language='pt-BR')
            #Após alguns segundos, retorna a frase falada
            print("Você disse: " + frase)
        #Caso nao tenha reconhecido o padrao de fala, exibe esta mensagem
        except sr.exceptions.UnknownValueError:
            print("tendi nada")
            frase = ""
        return frase

# Configurações do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Config padrões
cap = cv2.VideoCapture(0)
firstTime = True

inicio = time.perf_counter()
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Cria uma instância da mão com os dedos
                mao = Mao(hand_landmarks)
              
                # Calculando a distância entre a ponta do indicador e a ponta do polegar
                # distancia_1 = mao.indicador.distancia_para(mao.polegar.ponta, "ponta")
                # distancia_1 = round(distancia_1*100,1)

                # distancia_2 = mao.medio.distancia_para(mao.polegar.ponta, "ponta")
                # distancia_2 = round(distancia_2*100,1)
                
                distancia,ponto = mao.indicador.distancia_para(mao.mindinho.metacarpo, "metacarpo")
                winWidth,winHeight = (image.shape[1],image.shape[0])
                ponto.x = int(round(ponto.x*winWidth))
                ponto.y = int(round(ponto.y*winHeight))
                cv2.circle(image, (ponto.x, ponto.y), 5, (0, 255, 0), -1)
                # ref_metacarpo = (/2)
                # y - ref_metacarpo
                # referencial_metacarpo = mao.indicador.ponta.x + referencial_metacarpo
                
                
                
                
                ref = 4.5
                # Exibe a distância no console
                # print(f"\nDistância entre a ponta do indicador e do polegar: {distancia_1} unidades.")
                # print(f"\nDistância entre a ponta do dedo médio e do polegar: {distancia_2} unidades.")
                # if escolhendoSite:
                    
                # if distancia_1 < ref and (firstTime or tempo_passado > 5):
                #     print("Pinça fechada!")
                #     frasePesquisa = ouvir_microfone()
                #     if frasePesquisa != "":
                #         resultados = search(frasePesquisa,num_results=5)
                #         for idx, resultado in enumerate(resultados):
                #             print(f"{idx+1}. {resultado}")
                #         print("escolha um site: ")
                #     firstTime = False
                #     escolhendoSite = True
                #     inicio = time.perf_counter()
                  
                # elif distancia_1 < pow(ref,2) and (firstTime or tempo_passado > 5):
                #   pyautogui.hotkey('ctrl', 'super', 't')
                #   time.sleep(1)
                #   firstTime = False
                #   inicio = time.perf_counter()
                #   print("Pinça simples fechada")
                  
                # Mostrar distâncias das pontas dos dedos

                # h, w, _ = image.shape
                # cv2.line(image, (int(mao.polegar.ponta.x*w), int(mao.polegar.ponta.y*h)), 
                # (int(mao.indicador.ponta.x*w) ,int(mao.indicador.ponta.y*h)), (0, 0, 255), 2)

                # cv2.line(image, (int(mao.indicador.ponta.x*w), int(mao.indicador.ponta.y*h)), 
                # (int(mao.medio.ponta.x*w) ,int(mao.medio.ponta.y*h)), (0, 255, 0), 2)

                # cv2.line(image, (int(mao.medio.ponta.x*w), int(mao.medio.ponta.y*h)), 
                # (int(mao.anelar.ponta.x*w) ,int(mao.anelar.ponta.y*h)), (0, 0, 0), 2)
        
        cv2.imshow('MediaPipe Hands', image)
        cv2.moveWindow('MediaPipe Hands', 0, 0)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        fim = time.perf_counter()
        tempo_passado = fim-inicio
        # print(f"Tempo após comando: {tempo_passado:.3}")


cap.release()