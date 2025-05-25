import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from pynput.keyboard import Controller
import math
import pygame
import sys
import random
import sounddevice as sd
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
keyboard = Controller()

# --- Button Class for Keyboard (Used in Virtual Keyboard)
class Button:
def __init__(self, pos, text, size=[70, 70]):
self.pos = pos
self.size = size
self.text = text

# --- Draw Buttons for Virtual Keyboard
def drawAll(img, buttonList):
for button in buttonList:
x, y = button.pos
w, h = button.size
cv2.rectangle(img, button.pos, (x + w, y + h), (96, 96, 96), cv2.FILLED)
cv2.putText(img, button.text, (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
return img

# --- Virtual Keyboard Function
def start_virtual_keyboard():
cap = cv2.VideoCapture(0)
cap.set(3, 1000)
cap.set(4, 580)
text = ""
delay = 0
app = 0

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "CL"],
["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "SP"],
["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "APR"]]

keys1 = [[k.lower() for k in row] for row in keys]

buttonList = [Button([80 * j + 10, 80 * i + 10], key)
for i in range(len(keys)) for j, key in enumerate(keys[i])]
buttonList1 = [Button([80 * j + 10, 80 * i + 10], key)
for i in range(len(keys1)) for j, key in enumerate(keys1[i])]

def calculate_distance(x1, y1, x2, y2):
return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)

while True:
success, frame = cap.read()
frame = cv2.resize(frame, (1000, 580))
frame = cv2.flip(frame, 1)
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(img)
lanmark = []

list_buttons = buttonList if app == 0 else buttonList1
frame = drawAll(frame, list_buttons)
r = "up" if app == 0 else "down"

if results.multi_hand_landmarks:
for hn in results.multi_hand_landmarks:
for id, lm in enumerate(hn.landmark):
h, w, _ = frame.shape
cx, cy = int(lm.x * w), int(lm.y * h)
lanmark.append([id, cx, cy])

try:
x5, y5 = lanmark[5][1], lanmark[5][2]
x17, y17 = lanmark[17][1], lanmark[17][2]
dis = calculate_distance(x5, y5, x17, y17)
A, B, C = coff
distanceCM = A * dis**2 + B * dis + C

if 20 < distanceCM < 50:
x, y = lanmark[8][1], lanmark[8][2]
x2, y2 = lanmark[6][1], lanmark[6][2]
x3, y3 = lanmark[12][1], lanmark[12][2]
cv2.circle(frame, (x, y), 20, (255, 0, 255), cv2.FILLED)
cv2.circle(frame, (x3, y3), 20, (255, 0, 255), cv2.FILLED)

if y2 > y:
for button in list_buttons:
xb, yb = button.pos
wb, hb = button.size
if xb < x < xb + wb and yb < y < yb + hb:
cv2.rectangle(frame, (xb - 5, yb - 5),
(xb + wb + 5, yb + hb + 5), (160, 160, 160), cv2.FILLED)
cv2.putText(frame, button.text, (xb + 20, yb + 65),
cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
dis = calculate_distance(x, y, x3, y3)
if dis < 50 and delay == 0:
k = button.text
cv2.rectangle(frame, (xb - 5, yb - 5),
(xb + wb + 5, yb + hb + 5), (255, 255, 255), cv2.FILLED)
cv2.putText(frame, k, (xb + 20, yb + 65),
cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
if k == "SP":
text += ' '
keyboard.press(' ')
elif k == "CL":
text = text[:-1]
keyboard.press('\b')
elif k == "APR":
app = 1 if r == "up" else 0
else:
text += k
keyboard.press(k)
delay = 1

except Exception as e:
print("Error:", e)

if delay != 0:
delay += 1
if delay > 10:
delay = 0

cv2.rectangle(frame, (20, 250), (850, 400), (255, 255, 255), cv2.FILLED)
cv2.putText(frame, text, (30, 300), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
cv2.imshow('Virtual Keyboard', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break

cap.release()
cv2.destroyAllWindows()

# --- Virtual Mouse
def start_virtual_mouse():
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

while True:
ret, frame = cap.read()
if not ret:
break
frame = cv2.flip(frame, 1)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
output = hands.process(rgb_frame)

if output.multi_hand_landmarks:
for hand in output.multi_hand_landmarks:
mp_draw.draw_landmarks(frame, hand)
landmarks = hand.landmark
index_x = int(landmarks[8].x * screen_width)
index_y = int(landmarks[8].y * screen_height)
thumb_y = int(landmarks[4].y * screen_height)

if abs(index_y - thumb_y) < 20:
pyautogui.click()
else:
pyautogui.moveTo(index_x, index_y)

cv2.imshow('Virtual Mouse', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break

cap.release()
cv2.destroyAllWindows()

# --- Volume Control
def start_volume_control():
cap = cv2.VideoCapture(0)
while True:
ret, frame = cap.read()
if not ret:
break
frame = cv2.flip(frame, 1)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(rgb_frame)

if results.multi_hand_landmarks:
for hand_landmarks in results.multi_hand_landmarks:
mp_draw.draw_landmarks(frame, hand_landmarks)
index_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

if index_finger_y < thumb_y:
pyautogui.press('volumeup')
elif index_finger_y > thumb_y:
pyautogui.press('volumedown')

cv2.imshow('Volume Control', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
break

cap.release()
cv2.destroyAllWindows()

# --- Updated Drawing Application
def start_drawing():
cap = cv2.VideoCapture(0)
colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
color_names = ['CLEAR', 'BLUE', 'GREEN', 'RED', 'YELLOW']
current_color = (255, 0, 0)
canvas = None
paint_window = None
xp, yp = 0, 0

def draw_buttons(img):
for i, (color, name) in enumerate(zip(colors, color_names)):
x1 = i * 100
x2 = x1 + 100
cv2.rectangle(img, (x1, 0), (x2, 100), color, cv2.FILLED)
cv2.rectangle(img, (x1, 0), (x2, 100), (255, 255, 255), 2)
cv2.putText(img, name, (x1 + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
(0, 0, 0) if color != (0, 0, 0) else (255, 255, 255), 2)

while True:
ret, frame = cap.read()
if not ret:
break
frame = cv2.flip(frame, 1)
h, w, _ = frame.shape

if canvas is None:
canvas = np.zeros_like(frame)
if paint_window is None:
paint_window = 255 * np.ones_like(frame)

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(rgb)

draw_buttons(frame)

if results.multi_hand_landmarks:
for hand_landmarks in results.multi_hand_landmarks:
mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
lm = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]

if len(lm) > 12:
x1, y1 = lm[8]
x2, y2 = lm[12]

if abs(y2 - y1) < 40:
xp, yp = 0, 0
if y1 < 100:
color_index = x1 // 100
if color_index < len(colors):
current_color = colors[color_index]
if current_color == (0, 0, 0): # CLEAR
canvas = np.zeros_like(frame)
paint_window = 255 * np.ones_like(frame)
else:
if xp == 0 and yp == 0:
xp, yp = x1, y1
cv2.line(canvas, (xp, yp), (x1, y1), current_color, 5)
cv2.line(paint_window, (xp, yp), (x1, y1), current_color, 5)
xp, yp = x1, y1
else:
xp, yp = 0, 0

output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
cv2.imshow("Output", output)
cv2.imshow("Paint", paint_window)

if cv2.waitKey(1) & 0xFF == ord('q'):
break

cap.release()
cv2.destroyAllWindows()

# --- Car Game (Hand Control)
def start_car_game():
# Pygame settings
pygame.init()
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand-Controlled Car Game")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Load images
try:
car_image_path = 'car.png'
road_image_path = 'road.png'
obstacle_car_images_paths = ['obstacle_car1.png', 'obstacle_car2.png', 'obstacle_car3.png']

car_image = pygame.image.load(car_image_path)
road_image = pygame.image.load(road_image_path)
obstacle_car_images = [pygame.image.load(img_path) for img_path in obstacle_car_images_paths]

desired_car_width, desired_car_height = 50, 80

car_image = pygame.transform.scale(car_image, (desired_car_width, desired_car_height))
road_image = pygame.transform.scale(road_image, (WIDTH, HEIGHT))
obstacle_car_images = [pygame.transform.scale(img, (desired_car_width, desired_car_height)) for img in obstacle_car_images]

except Exception as e:
print("Error loading images:", e)
pygame.quit()
return

CAR_WIDTH, CAR_HEIGHT = car_image.get_size()
car_x = WIDTH // 2 - CAR_WIDTH // 2
car_y = HEIGHT - CAR_HEIGHT - 20

road_left_border = 304
road_right_border = 440

scroll_y = 0
distance = 0
font = pygame.font.SysFont("comicsans", 30)

clock = pygame.time.Clock()

# Settings for obstacle cars
NUM_OBSTACLE_CARS = 3

class ObstacleCar:
def __init__(self, x, y, speed, image):
self.x = x
self.y = y
self.speed = speed
self.image = image

obstacle_cars = []

def spawn_obstacle_cars(num_cars):
for _ in range(num_cars):
x_position = random.randint(road_left_border, road_right_border - CAR_WIDTH)
y_position = random.randint(-1000, 0)
speed = random.randint(1, 3)
image = random.choice(obstacle_car_images)
obstacle_car = ObstacleCar(x_position, y_position, speed, image)
obstacle_cars.append(obstacle_car)

spawn_obstacle_cars(NUM_OBSTACLE_CARS)

def calculate_speed(hand_y, min_speed=2, max_speed=100, sensitivity=1.5):
adjusted_y = 1 - hand_y
speed = min_speed + adjusted_y ** sensitivity * (max_speed - min_speed)
return int(speed)

def draw_window(car_x, car_y, scroll_y, distance, current_speed, lives):
win.fill(WHITE)
win.blit(road_image, (0, scroll_y))
win.blit(road_image, (0, scroll_y - HEIGHT))
win.blit(car_image, (car_x, car_y))

for obstacle_car in obstacle_cars:
win.blit(obstacle_car.image, (obstacle_car.x, obstacle_car.y))

distance_text = font.render(f"Distance: {distance} km", True, BLACK)
win.blit(distance_text, (WIDTH - distance_text.get_width() - 10, 10))

speed_text = font.render(f"Speed: {current_speed} km/h", True, BLACK)
win.blit(speed_text, (10, 10))

lives_text = font.render(f"Lives: {lives}", True, BLACK)
win.blit(lives_text, (10, 40))

pygame.display.update()

def is_fist_closed(hand_landmarks):
thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

thumb_index_distance = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
index_middle_distance = abs(index_tip.x - middle_tip.x) + abs(index_tip.y - middle_tip.y)

return thumb_index_distance < 0.05 and index_middle_distance < 0.05

def update_obstacles(current_speed):
for obs in obstacle_cars:
obs.y += obs.speed + current_speed // 10
if obs.y > HEIGHT:
obs.y = random.randint(-500, -100)
obs.x = random.randint(road_left_border, road_right_border - CAR_WIDTH)

def check_and_handle_collisions(car_x, car_y):
nonlocal lives
# FIXED: Changed 'game_y' to 'car_y' to use the correct y-position of the car
car_rect = pygame.Rect(car_x, car_y, CAR_WIDTH, CAR_HEIGHT)

for obs in obstacle_cars:
obstacle_rect = pygame.Rect(obs.x, obs.y, CAR_WIDTH, CAR_HEIGHT)

if car_rect.colliderect(obstacle_rect):
obs.y = random.randint(-500, -100)
obs.x = random.randint(road_left_border, road_right_border - CAR_WIDTH)
lives -= 1
print(f"Lives left: {lives}")

if lives <= 0:
print("Game Over!")
return False
return True

cap = cv2.VideoCapture(0)

lives = 6
running = True
current_speed = 0
while running:
ret, frame = cap.read()
if not ret:
print("Error capturing frame")
break

img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(img_rgb)

if results.multi_hand_landmarks:
for hand_landmarks in results.multi_hand_landmarks:
mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

if is_fist_closed(hand_landmarks):
current_speed = 0
else:
finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
current_speed = calculate_speed(finger_tip.y)
new_car_x = WIDTH - int(finger_tip.x * WIDTH) - CAR_WIDTH // 2

if new_car_x < road_left_border:
new_car_x = road_left_border
elif new_car_x > road_right_border:
new_car_x = road_right_border

car_x = new_car_x

scroll_y += current_speed
if scroll_y >= HEIGHT:
scroll_y = 0

distance += current_speed / 100
distance = round(distance, 1)

cv2.imshow("Hand Tracking", frame)

for event in pygame.event.get():
if event.type == pygame.QUIT:
running = False

update_obstacles(current_speed)

if not check_and_handle_collisions(car_x, car_y):
break

draw_window(car_x, car_y, scroll_y, distance, current_speed, lives)

if cv2.waitKey(1) & 0xFF == ord('q'):
break

clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()

# --- Snake Game (Voice Control)
def start_snake_game():
# Game settings
WIDTH, HEIGHT = 600, 600
CELL_SIZE = 20
FPS = 10
ANIMATION_SPEED = 5 # Higher = slower animation

# Colors
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
BG_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)
GRAY = (50, 50, 50)
WHITE = (255, 255, 255)

# Pygame Setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Voice-Controlled Snake")
clock = pygame.time.Clock()
font = pygame.font.SysFont("comicsansms", 48)
small_font = pygame.font.SysFont("comicsansms", 24)

# Sound Thresholds
LOW_THRESHOLD = 10 # Left
MEDIUM_THRESHOLD = 25 # Right
HIGH_THRESHOLD = 40 # Up
U_TURN_THRESHOLD = 60 # Down (U-turn)

# Timer Settings
NO_POINT_TIMEOUT = 30 # 30 seconds if no point
MAX_GAME_TIME = 180 # 3 minutes if point scored
start_time = time.time()

def get_microphone_volume():
duration = 0.1
sample_rate = 44100
try:
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
sd.wait()
volume = np.linalg.norm(audio) * 10
return volume
except Exception as e:
print("Mic error:", e)
return 0

def draw_cell(x, y, color, alpha=255):
cell_surface = pygame.Surface((CELL_SIZE, CELL_SIZE))
cell_surface.set_alpha(alpha)
cell_surface.fill(color)
screen.blit(cell_surface, (x, y))

def draw_snake(snake, alpha=255):
for i, segment in enumerate(snake):
shade = max(0, 200 - i * 20)
body_color = (0, shade, 0) if shade > 0 else DARK_GREEN
draw_cell(segment[0], segment[1], body_color, alpha)
if segment == snake[0]:
pygame.draw.circle(screen, WHITE, (segment[0] + CELL_SIZE // 4, segment[1] + CELL_SIZE // 4), 3)

def draw_food(food, pulse_size):
size = CELL_SIZE + abs(pulse_size)
x = food[0] + CELL_SIZE // 2 - size // 2
y = food[1] + CELL_SIZE // 2 - size // 2
for glow in range(5, 0, -1):
pygame.draw.rect(screen, (255, 0, 0, 50 // glow), (x - glow, y - glow, size + 2 * glow, size + 2 * glow))
pygame.draw.rect(screen, RED, (x, y, size, size))

def draw_background():
for x in range(0, WIDTH, CELL_SIZE):
for y in range(0, HEIGHT, CELL_SIZE):
color = LIGHT_GREEN if (x // CELL_SIZE + y // CELL_SIZE) % 2 == 0 else BG_GREEN
draw_cell(x, y, color)
for x in range(0, WIDTH, CELL_SIZE):
pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT), 1)
for y in range(0, HEIGHT, CELL_SIZE):
pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y), 1)
pygame.draw.rect(screen, GRAY, (0, 0, WIDTH, HEIGHT), 2)

def get_random_food_position():
x = random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
y = random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE
return (x, y)

def update_direction(current_direction, volume):
if volume < LOW_THRESHOLD:
return (-CELL_SIZE, 0) # Left
elif volume < MEDIUM_THRESHOLD:
return (CELL_SIZE, 0) # Right
elif volume < HIGH_THRESHOLD:
return (0, -CELL_SIZE) # Up
else:
return (0, CELL_SIZE) # Down

# Initialize snake and food
snake = [(100, 100), (80, 100), (60, 100)]
direction = (CELL_SIZE, 0)
food = get_random_food_position()
score = 0
running = True
game_over = False
pulse_size = 0
pulse_dir = 1
point_scored = False

# Main game loop
while running:
screen.fill(BG_GREEN)
draw_background()

for event in pygame.event.get():
if event.type == pygame.QUIT:
running = False

if not game_over:
volume = get_microphone_volume()
new_direction = update_direction(direction, volume)

if (new_direction[0] != -direction[0] or new_direction[1] != -direction[1]):
direction = new_direction

head = snake[0]
new_head = (head[0] + direction[0], head[1] + direction[1])

new_head = (new_head[0] % WIDTH, new_head[1] % HEIGHT)

for i in range(1, ANIMATION_SPEED + 1):
intermediate_x = head[0] + direction[0] * i / ANIMATION_SPEED
intermediate_y = head[1] + direction[1] * i / ANIMATION_SPEED
intermediate_x = intermediate_x % WIDTH
intermediate_y = intermediate_y % HEIGHT
screen.fill(BG_GREEN)
draw_background()
draw_snake([(int(intermediate_x), int(intermediate_y))] + snake[1:])
draw_food(food, pulse_size)
score_surface = pygame.Surface((200, 50))
score_surface.set_alpha(128)
score_surface.fill(BLACK)
score_text = font.render(f"Score: {score}", True, WHITE)
screen.blit(score_surface, (10, 10))
screen.blit(score_text, (20, 20))
elapsed_time = int(time.time() - start_time)
time_surface = pygame.Surface((150, 50))
time_surface.set_alpha(128)
time_surface.fill(BLACK)
time_text = font.render(f"Time: {elapsed_time}s", True, WHITE)
screen.blit(time_surface, (WIDTH - 160, 10))
screen.blit(time_text, (WIDTH - 150, 20))
pygame.display.flip()
clock.tick(FPS * 2)

snake.insert(0, new_head)

if new_head == food:
score += 1
point_scored = True
food = get_random_food_position()
else:
snake.pop()

if new_head in snake[1:]:
game_over = True
death_time = pygame.time.get_ticks()

elapsed_time = int(time.time() - start_time)
if not point_scored and elapsed_time >= NO_POINT_TIMEOUT:
game_over = True
elif point_scored and elapsed_time >= MAX_GAME_TIME:
game_over = True

if pulse_size >= 6 or pulse_size <= -6:
pulse_dir *= -1
pulse_size += pulse_dir * 0.5

draw_snake(snake)
draw_food(food, pulse_size)
score_surface = pygame.Surface((200, 50))
score_surface.set_alpha(128)
score_surface.fill(BLACK)
score_text = font.render(f"Score: {score}", True, WHITE)
screen.blit(score_surface, (10, 10))
score_text = font.render(f"Score: {score}", True, WHITE)
screen.blit(score_text, (20, 20))
time_surface = pygame.Surface((150, 50))
time_surface.set_alpha(128)
time_surface.fill(BLACK)
elapsed_time = int(time.time() - start_time)
time_text = font.render(f"Time: {elapsed_time}s", True, WHITE)
screen.blit(time_surface, (WIDTH - 160, 10))
screen.blit(time_text, (WIDTH - 150, 20))

if game_over:
fade_surface = pygame.Surface((WIDTH, HEIGHT))
for alpha in range(0, 255, 5):
fade_surface.set_alpha(alpha)
fade_surface.fill(BLACK)
screen.blit(fade_surface, (0, 0))
draw_snake(snake, 255 - alpha)
draw_food(food, pulse_size)
msg = font.render("GAME OVER", True, RED)
msg_shadow = small_font.render("GAME OVER", True, BLACK)
screen.blit(msg_shadow, (WIDTH // 2 - 85, HEIGHT // 2 - 30))
screen.blit(msg, (WIDTH // 2 - 80, HEIGHT // 2 - 25))
final_score = font.render(f"Score: {score}", True, WHITE)
final_time = small_font.render(f"Time: {elapsed_time}s", True, WHITE)
screen.blit(final_score, (WIDTH // 2 - 50, HEIGHT // 2 + 20))
screen.blit(final_time, (WIDTH // 2 - 50, HEIGHT // 2 + 60))
pygame.display.flip()
pygame.time.delay(50)
pygame.time.delay(2000)
running = False

pygame.display.flip()
clock.tick(FPS)

pygame.quit()

# --- Submenu for Virtual Tools
def virtual_tools_submenu():
while True:
print("\nVirtual Tools Submenu:")
print("1. Virtual Keyboard")
print("2. Virtual Mouse")
print("3. Volume Control")
print("4. Drawing Application")
print("5. Back to Main Menu")
choice = input("Enter your choice: ")

if choice == '1':
start_virtual_keyboard()
elif choice == '2':
start_virtual_mouse()
elif choice == '3':
start_volume_control()
elif choice == '4':
start_drawing()
elif choice == '5':
break
else:
print("Invalid choice, please try again.")

# --- Main Menu
def main_menu():
while True:
print("\nMain Menu:")
print("1. Virtual Mouse + Keyboard + Drawing + Volume")
print("2. Car Game (Hand Control)")
print("3. Snake Game")
print("4. Exit")
choice = input("Enter your choice: ")

if choice == '1':
virtual_tools_submenu()
elif choice == '2':
start_car_game()
elif choice == '3':
start_snake_game()
elif choice == '4':
print("Exiting...")
break
else:
print("Invalid choice, please try again.")

if __name__ == "__main__":
main_menu()
