import pygame
import sys
import random

from enum import Enum

from brains import StraightDirectionSimpleBrain, SimpleBrain, Brain
from utils import NeuronInput
import time

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

class GameOutput():
    def __init__(self):
        self.mouse_pos = None
        self.rel_speed = None

class GameStates(Enum):
    TITLE = 0
    GAME = 1
    END = 2
    STARTING = 3

class RecordingStates(Enum):
    RECORDING = 0
    NOT_RECORDING = 1

class GameState():
    def __init__(self):
        self.state = GameStates.TITLE
    
class RecordingState():
    def __init__(self):
        self.state = RecordingStates.NOT_RECORDING
        self.recording_file_buffer = None
    
    def start_recording(self, game_output: GameOutput, brain: Brain):
        self.state = RecordingStates.RECORDING
        filename = f"./recordings/recording_{time.time()}.csv"
        self.recording_file_buffer = open(filename, "w")
        # draw header according to game_output
        self.recording_file_buffer.write("time,mouse_x,mouse_y,mouse_speed_x,mouse_speed_y")
        # for each neuron in the brain add a column
        for num, neuron in enumerate(brain.neurons):
            self.recording_file_buffer.write(f",neuron_{num}")
        self.recording_file_buffer.write("\n")
        self.add_recording(game_output, brain)
    
    def add_recording(self, game_output: GameOutput, brain: Brain):
        self.recording_file_buffer.write(
            f"{time.time()}, {game_output.mouse_pos[0]}, {game_output.mouse_pos[1]}, {game_output.rel_speed[0]}, {game_output.rel_speed[1]}")
        for neuron in brain.neurons:
            self.recording_file_buffer.write(f", {int(neuron.spike_train.data[-1][1])}")
        self.recording_file_buffer.write("\n")

    def stop_recording(self):
        self.state = RecordingStates.NOT_RECORDING
        self.recording_file_buffer.close()
        self.recording_file_buffer = None
    

# Common event handling
def handle_common_events(event):
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()
    elif event.type == pygame.KEYUP:
        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
            pygame.quit()
            sys.exit()

# Event handling for the different screens
def handle_events_title_screen(game_state: GameState):
    for event in pygame.event.get():
        handle_common_events(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                game_state.state = GameStates.STARTING

def handle_events_game_screen(game_state: GameState, recording_state: RecordingState, game_output: GameOutput, brain: Brain):
    for event in pygame.event.get():
        handle_common_events(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                game_state.state = GameStates.TITLE
            if event.key == pygame.K_r:
                if recording_state.state == RecordingStates.RECORDING:
                    recording_state.stop_recording()
                else:
                    print("Start recording")
                    recording_state.start_recording(game_output, brain)
    mouse_pos = pygame.mouse.get_pos()
    mouse_speed = pygame.mouse.get_rel()
    game_output.mouse_pos = mouse_pos
    game_output.rel_speed = mouse_speed

# Drawing the different screens
def draw_title_screen(screen, font):
    screen.fill(BLACK)
    start_text = font.render("Press SPACE to start or ESC to quit", True, WHITE)
    text_rect = start_text.get_rect(center=(screen_width/2, screen_height/2))
    screen.blit(start_text, text_rect)

def draw_game_screen(screen, font, recording_state, game_output):
    mouse_pos = game_output.mouse_pos
    mouse_speed = game_output.rel_speed
    screen.fill(BLACK)
    text = font.render(f"Pos: {mouse_pos} Speed: {mouse_speed}", True, WHITE)
    screen.blit(text, (10, 10))
    pygame.draw.circle(screen, WHITE, mouse_pos, 20)
    # draw a little flashing red circle on the top right if recording_state is recording
    if recording_state.state == RecordingStates.RECORDING:
        if time.time() % 1 < 0.5:
            pygame.draw.circle(screen, RED, (screen_width - 20, 20), 10)

def draw_spike_train(screen, spike_train, ypos, line_height, line_spacing):
    for i, value in enumerate(spike_train):
        if value == 1:
            pygame.draw.line(screen, RED, 
                             (i * line_spacing, ypos - line_height), 
                             (i * line_spacing, ypos))
            
# Logic for the game
def game_logic(recording_state: RecordingState, game_output: GameOutput, brain: Brain):
    if recording_state.state == RecordingStates.RECORDING:
        recording_state.add_recording(game_output, brain)

def main():
    game_state = GameState()
    brain = None
    recording_state = RecordingState()
    game_output = GameOutput()

   
    while True:
        dt = time.time()
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
                handle_events_game_screen(game_state, recording_state, game_output, brain)
                draw_game_screen(screen, font, recording_state, game_output)

                # Update the brain
                neuron_input = NeuronInput(game_output.rel_speed) 
                brain.update(neuron_input)
                # Get the spikes
                show_n_spikes = 100
                spikes = brain.get_n_recent_spikes(show_n_spikes)
                # Draw the spikes

                for num, spike_train in enumerate(spikes):
                    draw_spike_train(screen, spike_train, screen_height - 20 - num * 20, 10, screen_width / show_n_spikes)

                game_logic(recording_state, game_output, brain)
            case GameStates.END:
                ...
            
        # # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

if __name__ == '__main__':
    main()