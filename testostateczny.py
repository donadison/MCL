import pygame
import random
import math
import numpy as np



# Kolory
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Parametry symulacji
maxread=30
WIDTH = 1000
HEIGHT = 800
FPS = 60
ROBOT_SIZE = 30
SENSOR_LENGTH = 1000
SENSOR_ANGLE = 0  # Kąt w stopniach, 0 to w prawo, 90 to w dół, -90 to w górę
SENSOR_OFFSET = 20  # Odległość czujnika od centrum robota
MOVEMENT_SPEED = 5
TURN_ANGLE = 5  # Kąt obrotu robota podczas skręcania
NUM_PARTICLES = 100

# Przeszkody
OBSTACLES = [pygame.Rect(200, 200, 100, 300), pygame.Rect(500, 100, 200, 100), pygame.Rect(700, 100, 100, 300), pygame.Rect(600, 500, 100, 100),
             pygame.Rect(200, 640, 700, 100)]



def initialize_particles():
    particles = []
    for _ in range(NUM_PARTICLES):
        x = np.random.uniform(0, WIDTH)
        y = np.random.uniform(0, HEIGHT)
        r = np.random.uniform(0,360)
        particles.append((x, y, r))
    return particles


def initialize_weights():
    return np.ones(NUM_PARTICLES) / NUM_PARTICLES


def draw_particles(screen, particles):
    for particle in particles:
        pygame.draw.circle(screen, BLUE, (int(particle[0]), int(particle[1])), 2)
def draw_obstacles(screen):
    for obstacle in OBSTACLES:
        pygame.draw.rect(screen, BLACK, obstacle)


def is_collision(robot_rect):
    for obstacle in OBSTACLES:
        if robot_rect.colliderect(obstacle):
            return True
    return False



def draw_robot(screen, pos):
    pygame.draw.rect(screen, RED, pygame.Rect(pos[0] - ROBOT_SIZE / 2, pos[1] - ROBOT_SIZE / 2, ROBOT_SIZE, ROBOT_SIZE))


def draw_sensor(screen,robot_pos,angle):
    sensor_end_x = robot_pos[0] + math.cos(math.radians(angle)) * 100
    sensor_end_y = robot_pos[1] - math.sin(math.radians(angle)) * 100
    pygame.draw.line(screen, BLACK, robot_pos, (sensor_end_x, sensor_end_y), 1)


def sensor(screen, robot_pos, angle):
    sensor_length = 0
    sensor_x = robot_pos[0]
    sensor_y = robot_pos[1]

    # Przesuwamy sensor w kierunku jego kąta, dopóki nie natrafi na przeszkodę
    while sensor_length < SENSOR_LENGTH:
        sensor_x = robot_pos[0] + int(sensor_length * math.cos(math.radians(angle)))
        sensor_y = robot_pos[1] - int(sensor_length * math.sin(math.radians(angle)))

        # Sprawdzamy, czy nowa pozycja sensora znajduje się w granicach ekranu
        if not (0 <= sensor_x < WIDTH and 0 <= sensor_y < HEIGHT):
            break

        # Sprawdzamy, czy sensor natrafił na przeszkodę
        for obstacle in OBSTACLES:
            if obstacle.collidepoint(sensor_x, sensor_y):
                # Jeśli tak, zwracamy odległość do punktu na przeszkodzie
                return math.sqrt((sensor_x - robot_pos[0]) ** 2 + (sensor_y - robot_pos[1]) ** 2) - 16

        sensor_length += 1

    # Jeśli sensor nie natrafił na przeszkodę, zwracamy jego maksymalną długość
    return SENSOR_LENGTH


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Particle Filter TEST")
    clock = pygame.time.Clock()
    particles = initialize_particles()
    weights = initialize_weights()  # Inicjalizacja wag cząsteczek

    robot_pos = [WIDTH / 2, HEIGHT / 2]
    robot_angle = 0  # Początkowy kąt robota

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            new_x = robot_pos[0] + MOVEMENT_SPEED * math.cos(math.radians(robot_angle))
            new_y = robot_pos[1] - MOVEMENT_SPEED * math.sin(math.radians(robot_angle))
            new_rect = pygame.Rect(new_x - ROBOT_SIZE / 2, new_y - ROBOT_SIZE / 2, ROBOT_SIZE, ROBOT_SIZE)
            if not is_collision(new_rect) and 0 <= new_x <= WIDTH and 0 <= new_y <= HEIGHT:
                robot_pos = [new_x, new_y]

                # Predykcja: Aktualizacja pozycji cząsteczek z losowym szumem
                for i in range(len(particles)):
                    noise = np.random.normal(0, 10, 2)  # Losowy szum
                    particles[i] = (particles[i][0] + MOVEMENT_SPEED * math.cos(math.radians(robot_angle)) + noise[0],
                                    particles[i][1] - MOVEMENT_SPEED * math.sin(math.radians(robot_angle)) + noise[1],
                                    particles[i][2])

        if keys[pygame.K_LEFT]:
            robot_angle += TURN_ANGLE
        if keys[pygame.K_RIGHT]:
            robot_angle -= TURN_ANGLE

        draw_obstacles(screen)
        draw_robot(screen, robot_pos)
        draw_sensor(screen, robot_pos, robot_angle + SENSOR_ANGLE)

        # Pomiar: Uzyskanie pomiaru z sensora
        closest_distance = sensor(screen, robot_pos, robot_angle + SENSOR_ANGLE)
        print("Odległość od najbliższej przeszkody:", closest_distance)

        # Aktualizacja wag: Aktualizacja wag cząsteczek na podstawie zgodności ich pozycji z pomiarem
        for i in range(NUM_PARTICLES):
            particle_pos = particles[i][:2]
            particle_angle = particles[i][2]
            particle_distance = sensor(screen, particle_pos, particle_angle + SENSOR_ANGLE)
            weights[i] = 1.0 / (abs(closest_distance - particle_distance) + 1e-10)  # Aktualizacja wag
        weights /= np.sum(weights)  # Normalizacja wag

        # Resampling: Wybór nowego zbioru cząsteczek na podstawie ich wag
        indices = np.random.choice(range(NUM_PARTICLES), size=NUM_PARTICLES, p=weights)
        particles = [particles[idx] for idx in indices]

        # Rysowanie cząsteczek
        draw_particles(screen, particles)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
