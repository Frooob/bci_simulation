import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

class SimpleKalmanFilter():
    def __init__(self) -> None:
        self.train_X = []  # states
        self.train_Y = []  # neuron measurements
        self.train_mode = False
        self.aquiring_traing_data = False
        self.prediction_mode = False
        self.acqw = None
        self.current_state = None
        self.last_time = None
    
    def get_sensible_defaults(self):
        self.acqw = default_acqw()
    
    def add_training_data(self, new_state, new_measurement):
        self.train_X.append(new_state)
        self.train_Y.append(new_measurement)

    def train(self):
        X = np.array(self.train_X).T
        Y = np.array(self.train_Y).T
        try:
            self.acqw = compute_A_C_Q_W(X, Y)
        except:
            print(f"[{datetime.now()}]: Could not compute A, C, Q, W.")
            return
    
    def predict(self, new_measurement, t):
        if self.last_time is None:
            self.last_time = t
            return
        dt = t - self.last_time
        self.last_time = t
        if self.acqw is None:
            print(f"[{datetime.now()}]: Could not predict, no training data available.")
            return
        if self.current_state is None:
            print(f"[{datetime.now()}]: Could not predict, no current position available.")
            return        
        predicted_state, _ = compute_filter([new_measurement], self.current_state, *self.acqw)

        predicted_velocity = predicted_state[0,2:4]
        pos_change = compute_position_change(predicted_velocity, dt)
        pos_change *= 61  # multiply by 60 to account for spikes per second
        self.current_state[:2] += pos_change

        return self.current_state


def compute_filter(measurements, initial_state, A, C, Q, W):
    sigma_mask=[[0,0,0,0,0],
                [0,0,0,0,1],
                [0,0,1,1,1],
                [0,0,1,1,1],
                [1,1,1,1,1]]
    predicted_states = []
    x_t_minus_1 = initial_state
    sigma_t_minus_1 = np.zeros([initial_state.shape[0], initial_state.shape[0]])
    sigmas = [sigma_t_minus_1]
    for y_t in measurements:
        x_hat_t_given_t_minus_1 = A @ x_t_minus_1
        sigma_t_given_t_minus_1 = A @ sigma_t_minus_1 @ A.T + W
        sigma_t_given_t_minus_1 *= sigma_mask
        y_hat_t = y_t - C @ x_hat_t_given_t_minus_1
        S_t = C @ sigma_t_given_t_minus_1 @ C.T + Q
        K_t = sigma_t_given_t_minus_1 @ C.T @ np.linalg.inv(S_t)
        
        x_hat_t = x_hat_t_given_t_minus_1 + K_t @ y_hat_t
        sigma_t = (np.identity(5) - K_t @ C) @ sigma_t_given_t_minus_1
        
        predicted_states.append(x_hat_t)
        x_t_minus_1 = x_hat_t

        sigmas.append(sigma_t)
        sigma_t_minus_1 = sigma_t
    predicted_states = np.array(predicted_states)
        
    return predicted_states, sigmas

def compute_A_C_Q_W(X,Y, dt=1):
    # if dt is set to 1, the automatically calculated position won't make a lot of sense. Refrain to computing the trajectory from the velocities then. 
    X1 = X[:,:-1]
    X2 = X[:,1:]
    D = X.shape[1]
    A = X2@X1.T @ np.linalg.inv((X1@X1.T)) 
    A_mult_mask = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,1,0],
        [0,0,1,1,0],
        [0,0,0,0,0],])
    A_add_mask = np.array([
        [1,0,60*dt,0, 0],  # multiply dt by 60 to account for spikes per second
        [0,1,0, 60*dt,0],
        [0,0,0, 0, 0],
        [0,0,0, 0, 0],
        [0,0,0, 0, 1],])
    A = A * A_mult_mask + A_add_mask
    W = (1/(D-1)) * (X2 - A@X1)@(X2 - A@X1).T
    W_mult_mask = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,1,0],
        [0,0,1,1,0],
        [0,0,0,0,0],])
    W = W * W_mult_mask
    C = Y @ X.T @ np.linalg.inv(X@X.T)
    Q = (1/(D-1)) * (Y - C@X)@(Y - C@X).T
    return A, C, Q, W

def compute_position_change(velocity, dt):
    return velocity * dt

def compute_trajectory_from_velocities(initial_state, velocities, dt):
    # Computes the trajectory from the velocities
    trajectory = [initial_state]
    for velocity in velocities:
        trajectory.append(trajectory[-1] + velocity * dt )
    return np.array(trajectory)


def default_acqw(recording_path='./recordings/recording_no_noise_nice_noorigin.csv'):
    recording_df, state_cols, neuron_cols, velocity_cols_copied, X, Y, A, C, Q, W, dt = analyse_recording_df(recording_path)
    return A, C, Q, W

def analyse_recording_df(recording_path):
    recording_df = pd.read_csv(recording_path)
    state_cols = ['mouse_x', 'mouse_y', 'mouse_speed_x', 'mouse_speed_y']
    neuron_cols = [col for col in recording_df.columns if col.startswith('neuron_')]
    velocity_cols = [c for c in recording_df.columns if c.startswith('mouse_speed_')]
    velocity_cols_copied = [c + "_vel" for c in velocity_cols]

    dt_first = recording_df["time"].diff().mean()
    recording_df["dt"] = dt_first
    # subtract the first position from all the positions
    recording_df['mouse_x'] -= recording_df['mouse_x'].iloc[0]
    recording_df['mouse_y'] -= recording_df['mouse_y'].iloc[0]

    # Create a copy of the velocity columns and multiply them with 59
    recording_df[velocity_cols_copied] = recording_df[velocity_cols]
    # recording_df[velocity_cols] *= 0.05

    # resample the data, by taking the mean of every 3 rows but the sum of the time column

    aggregator = {"dt": ("dt", "sum"), 
                **{sc: (sc, "mean") for sc in state_cols}, 
                **{nc: (nc, "sum") for nc in neuron_cols},
                    **{vc: (vc, "mean") for vc in velocity_cols_copied}
                }

    recording_df = recording_df.groupby(np.arange(len(recording_df))//1).agg(**aggregator)

    dt = recording_df["dt"].mean()

    X = recording_df[state_cols].values.T
    X = np.vstack((X, np.ones(X.shape[1])))  # add a 1 column to the end of X
    Y = recording_df[neuron_cols].values.T
    A, C, Q, W = compute_A_C_Q_W(X,Y,dt)
    return recording_df, state_cols, neuron_cols, velocity_cols_copied, X, Y, A, C, Q, W, dt


if __name__ == '__main__':
    # recording_df = pd.read_csv('./recordings/recording_no_base_spikes.csv')
    # recording_df = pd.read_csv('./recordings/recording_no_noise_horizontal_vertical.csv')
    recording_path = './recordings/recording_no_noise_nice_noorigin.csv'
    recording_df = pd.read_csv(recording_path)

    recording_df, state_cols, neuron_cols, velocity_cols_copied, X, Y, A, C, Q, W, dt = analyse_recording_df(recording_path)

    # Plot all the real positions
    states, measurements = recording_df[state_cols].values, recording_df[neuron_cols].values
    # add a 1 column to the end of the states
    states = np.hstack((states, np.ones((states.shape[0], 1))))
    initial_pos = states[0]
    predicted_states, sigmas = compute_filter(measurements, states[0], A, C, Q, W)


    def plotting():
        # predicted
        # Create a colormap
        # add the initial position back to the predicted states
        # predicted_states += initial_pos
        cmap = plt.get_cmap('viridis')
        x = predicted_states[:,0]
        y = predicted_states[:,1]

        print(f"Mean position of predicted trajectory: {np.mean(predicted_states, axis=0)}")

        # See the convergence of the trajectory with 0 input for each time step (no spikes)
        input = np.ones((len(neuron_cols), 50)).T
        print(input.shape)
        # add zeros
        input_zeros = np.ones((len(neuron_cols), 1000)).T
        input = np.concatenate((input, input_zeros), axis=0)
        print(input.shape)
        predicted_states_convergence, sigmas = compute_filter(input, np.array([0,0,0,0,0]), A, C, Q, W)
        trajectory = compute_trajectory_from_velocities([0,0], predicted_states_convergence[:,2:4], dt) * 60
        print(predicted_states_convergence)
        x_convergence = predicted_states_convergence[:,0]
        y_convergence = predicted_states_convergence[:,1]
        plt.plot(x_convergence, y_convergence, label='convergence')
        plt.title(f'predicted convergence')
        plt.legend()



        # actual
        x = states[:,0]
        y = states[:,1]
        fig, ax = plt.subplots()
        for i in range(len(x)-1):
            ax.plot(x[i:i+2], y[i:i+2], color=cmap(i / len(x)))
            if i == 0:
                ax.scatter(x[i], y[i], color="red", label='start')
            elif i == len(x) - 2:
                ax.scatter(x[i+1], y[i+1], color="blue", label='end')
        ax.set_title('Predicted trajectory fancy colormap')
        ax.legend()
        plt.show()


        pred_color = "tab:orange"
        real_color = "tab:blue"
        plt.plot(X[0,:], X[1,:], label='real', color=real_color)
        plt.plot(predicted_states[:,0], predicted_states[:,1], label='predicted', color=pred_color)
        # indicate the start and end of the trajectory
        plt.scatter(X[0,0], X[1,0], color="black", label='start', zorder=5)
        plt.scatter(X[0,-1], X[1,-1], color=real_color,zorder=5)
        plt.scatter(predicted_states[-1,0], predicted_states[-1,1], color=pred_color, zorder=5)
        plt.title(f'predicted vs real trajectory')
        plt.legend()
        plt.show()

        # x direction
        neurons_to_plot = [0,1]
        plt.plot(recording_df["mouse_speed_x_vel"], label='real')
        plt.plot(predicted_states[:,2], label='predicted')
        # for i in neurons_to_plot:
            # plt.plot(Y[i,:], label=f'neuron {i}')
        plt.title(f'x direction speed predicted vs real')
        plt.legend()
        plt.show()

        # y direction
        neurons_to_plot = [2,3]
        plt.plot(recording_df["mouse_speed_y_vel"], label='real')
        plt.plot(predicted_states[:,3], label='predicted')
        # for i in neurons_to_plot:
            # plt.plot(Y[i,:], label=f'neuron {i}')
        plt.title(f'y direction speed predicted vs real')
        plt.legend()
        plt.show()

        # plot the trajectory computed from the real velocities, should trivially be overlapping
        trajectory = compute_trajectory_from_velocities([0,0], recording_df[velocity_cols_copied].values, dt) * 60
        plt.plot(trajectory[:,0], trajectory[:,1], label='real from vel')
        plt.plot(X[0,:], X[1,:], label='real')
        plt.title(f'trajectory computed from real velocities (trivial overlap)')
        plt.legend()
        plt.show()

    plotting()
