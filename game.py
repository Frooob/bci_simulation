import pygame
import sys
import random

from enum import Enum
import numpy as np

from brains import StraightDirectionSimpleBrain, SimpleBrain, Brain
from utils import NeuronInput
from filters import SimpleKalmanFilter
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
MIDDLE = np.array((screen.get_width() / 2, screen.get_height() / 2))

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
    
    def state_for_kalman_filter(self):
        return [self.mouse_pos[0], self.mouse_pos[1], self.rel_speed[0], self.rel_speed[1],1] # Add 1 in the end for the possibility of noise correction

    def state_for_recording(self):
        return [self.mouse_pos[0], self.mouse_pos[1], self.rel_speed[0], self.rel_speed[1]]

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
        posx, posy, velx, vely = game_output.state_for_recording()
        self.recording_file_buffer.write(
            f"{time.time()}, {posx}, {posy}, {velx}, {vely}")
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

def handle_events_game_screen(game_state: GameState, recording_state: RecordingState, game_output: GameOutput, brain: Brain, kalman_filter: SimpleKalmanFilter):
    for event in pygame.event.get():
        handle_common_events(event)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                game_state.state = GameStates.TITLE
            if event.key == pygame.K_r:
                if recording_state.state == RecordingStates.RECORDING:
                    recording_state.stop_recording()
                else:
                    recording_state.start_recording(game_output, brain)
            if event.key == pygame.K_s:
                set_filter_and_mouse_to_middle(kalman_filter)
            if event.key == pygame.K_t:
                kalman_filter.train_mode = not kalman_filter.train_mode
                kalman_filter.aquiring_traing_data = not kalman_filter.aquiring_traing_data
            if event.key == pygame.K_p:
                kalman_filter.prediction_mode = not kalman_filter.prediction_mode

    mouse_pos = pygame.mouse.get_pos()
    mouse_speed = pygame.mouse.get_rel()
    mouse_pos = np.array(mouse_pos) - MIDDLE
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
    mouse_pos_display = mouse_pos + MIDDLE # to draw the mouse on the correct position
    pygame.draw.circle(screen, WHITE, mouse_pos_display, 20)
    # draw a little flashing red circle on the top right if recording_state is recording
    if recording_state.state == RecordingStates.RECORDING:
        if time.time() % 1 < 0.5:
            pygame.draw.circle(screen, RED, (screen_width - 20, 20), 10)

def draw_predicted_pos(screen, predicted_pos):
    pygame.draw.circle(screen, RED, predicted_pos + MIDDLE, 20)

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

def set_filter_and_mouse_to_middle(kalman_filter: SimpleKalmanFilter):
    pygame.mouse.set_pos(*MIDDLE)
    kalman_filter.current_state = np.array((0,0,0,0,1), dtype=np.float64)

def main():
    game_state = GameState()
    brain = None
    recording_state = RecordingState()
    game_output = GameOutput()
    kalman_filter = SimpleKalmanFilter()

   
    while True:
        t = time.time()
        # Screen Events
        match game_state.state:
            case GameStates.TITLE:
                handle_events_title_screen(game_state)
                draw_title_screen(screen, font)
            case GameStates.STARTING:
                game_state.state = GameStates.GAME
                # brain = StupidBrain()
                brain = StraightDirectionSimpleBrain()
                set_filter_and_mouse_to_middle(kalman_filter)
                kalman_filter.get_sensible_defaults() # TODO: remove this
            case GameStates.GAME:
                ## Handle events and get new data
                handle_events_game_screen(game_state, recording_state, game_output, brain, kalman_filter)  # Writes to game_output new position

                ## Logic of the game and brain
                neuron_input = NeuronInput(game_output.rel_speed) 
                brain.update(neuron_input)
                game_logic(recording_state, game_output, brain)
                new_measurement = brain.get_most_recent_measurement()
                if kalman_filter.aquiring_traing_data:
                    kalman_filter.add_training_data(game_output.state_for_kalman_filter(), new_measurement)
                if kalman_filter.train_mode:
                    kalman_filter.train()
                if kalman_filter.prediction_mode:
                    predicted_state = kalman_filter.predict(new_measurement, t)
                    if predicted_state is not None:
                        predicted_pos = predicted_state[:2]
                else: 
                    predicted_pos = None
                    
                # Optional: Make this thing kinda frustrating
                if predicted_pos is not None:
                    pygame.mouse.set_pos(*predicted_pos)
                    game_output.mouse_pos = predicted_pos
                ## Draw everything on screen
                draw_game_screen(screen, font, recording_state, game_output)
                if predicted_pos is not None:
                    draw_predicted_pos(screen, predicted_pos)
                show_n_spikes = 100
                spikes = brain.get_n_recent_spikes(show_n_spikes)
                # Draw the spikes
                for num, spike_train in enumerate(spikes):
                    draw_spike_train(screen, spike_train, screen_height - 20 - num * 20, 10, screen_width / show_n_spikes)   
            case GameStates.END:
                ...
            
        # # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

if __name__ == '__main__':
    main()