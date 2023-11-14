import pygame
import sys
import random

from enum import Enum

from brains import StraightDirectionSimpleBrain, SimpleBrain
from utils import NeuronInput

# Initialize Pygame
pygame.init()

# Set up the display
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Mouse Tracker with Pygame')

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Set the starting position of the circle
circle_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

# Setup the font for displaying the speed
font = pygame.font.Font(None, 36)

# Clock to control the frame rate
clock = pygame.time.Clock()

# Enable relative mouse mode (uncomment if you want to use relative mode)
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)

class GameStates(Enum):
    TITLE = 0
    GAME = 1
    END = 2
    STARTING = 3

class GameState():
    def __init__(self):
        self.state = GameStates.TITLE
    

# Common event handling
def handle_common_events(event):
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()
    elif event.type == pygame.KEYUP:
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
            pygame.quit()
            sys.exit()

def handle_events_title_screen(game_state: GameState):
    for event in pygame.event.get():
        handle_common_events(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                game_state.state = GameStates.STARTING

def handle_events_game_screen(game_state: GameState):
    for event in pygame.event.get():
        handle_common_events(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                game_state.state = GameStates.TITLE
    mouse_pos = pygame.mouse.get_pos()
    mouse_speed = pygame.mouse.get_rel()
    return mouse_pos, mouse_speed

def draw_title_screen(screen, font):
    screen.fill(BLACK)
    start_text = font.render("Press SPACE to start or ESC to quit", True, WHITE)
    text_rect = start_text.get_rect(center=(screen_width/2, screen_height/2))
    screen.blit(start_text, text_rect)

def draw_game_screen(screen, font, mouse_pos, mouse_speed):
    screen.fill(BLACK)
    text = font.render(f"Pos: {mouse_pos} Speed: {mouse_speed}", True, WHITE)
    screen.blit(text, (10, 10))
    pygame.draw.circle(screen, WHITE, mouse_pos, 20)

# The spike train data, for example, 100 time points
spike_train_data = [0] * 100

def draw_spike_train(screen, spike_train, ypos, line_height, line_spacing):
    for i, value in enumerate(spike_train):
        if value == 1:
            pygame.draw.line(screen, RED, 
                             (i * line_spacing, ypos - line_height), 
                             (i * line_spacing, ypos))

def main():
    game_state = GameState()
    brain = None
    
   
    while True:
        # Screen Events
        match game_state.state:
            case GameStates.TITLE:
                handle_events_title_screen(game_state)
                draw_title_screen(screen, font)
            case GameStates.STARTING:
                game_state.state = GameStates.GAME
                # brain = StupidBrain()
                brain = StraightDirectionSimpleBrain()
            case GameStates.GAME:
                mouse_pos, mouse_speed = handle_events_game_screen(game_state)
                draw_game_screen(screen, font, mouse_pos, mouse_speed)


                # Update the brain
                neuron_input = NeuronInput(mouse_speed) 
                brain.update(neuron_input)
                # Get the spikes
                spikes = brain.get_n_recent_spikes(100)
                # Draw the spikes

                for num, spike_train in enumerate(spikes):
                    draw_spike_train(screen, spike_train, screen_height - 20 - num * 20, 10, screen_width / len(spike_train_data))

                # draw_spike_train(screen, spikes[0], screen_height - 20, 10, screen_width / len(spike_train_data))
                # draw_spike_train(screen, spikes[1], screen_height - 40, 10, screen_width / len(spike_train_data))

            case GameStates.END:
                ...
            
        # # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

if __name__ == '__main__':
    main()