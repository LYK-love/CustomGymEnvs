import numpy as np
import pygame
import time

import gym
from gym import spaces
from gym.utils import seeding

class BouncingBallEnv(gym.Env):
    """
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, size=2, velocity_scale = 1.0, ball_radius_ratio = 0.005, wall_thickness_ratio = 0.01, energy_loss_factor=0.9, apply_action=True, log=False):
        """
        Initialize the BouncingBallEnv.

        """
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.seed()
        
        # Define observation space (ball's position and velocity in 2D)
        self.observation_space = spaces.Box(low=0, high=size, shape=(2,), dtype=np.float32)
        # Define action space (2D continuous: move in any direction)
        self.action_space = spaces.Box(low=-velocity_scale, high=velocity_scale, shape=(2,), dtype=np.float32)
        # Initialize state (position of the ball)
        self.state = None
        
        self.energy_loss_factor = energy_loss_factor  # Control how much energy is lost on collision

        if apply_action:
            self.velocity_change_factor = 0.1 # Control how much the action affects the velocity
        else:
            self.velocity_change_factor = 0 
        
        self.min_velocity = 0.01  # Threshold for considering the velocity to be effectively zero
        self.velocity_initial_size = velocity_scale
        # Example of adjusting units to the world size
        self.wall_thickness_ratio = wall_thickness_ratio  # Wall thickness as a proportion of world size
        self.ball_radius_ratio = ball_radius_ratio  # Ball radius as a proportion of world size
        
        self.safe_margin = self.size * self.wall_thickness_ratio  + self.size * self.ball_radius_ratio # The size of the wall + the size of the (radius) if the ball 
            
        # Adjust for ball's radius in collision detection
        self.left_bound = 0 + self.safe_margin 
        self.right_bound = self.size  - self.safe_margin
        self.bottom_bound = 0 + self.safe_margin
        self.top_bound = self.size  - self.safe_margin
        
        
        self.log = log
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def _get_obs(self):
        """
        Get the current observation of the environment.

        Returns:
        - obs (np.ndarray): The current observation of the environment as an RGB array.
        """
        # Temporarily set render_mode to 'rgb_array' for this operation, if necessary
        original_render_mode = self.render_mode
        self.render_mode = 'rgb_array'
        
        # Generate the observation as an RGB array
        obs = self._render_frame()

        # Reset the render mode to its original state, if needed
        self.render_mode = original_render_mode
        
        return obs
    
    def _get_info(self):
        """
        Get additional information about the environment.

        Returns:
        - info (dict): Additional information about the environment.
        """
        return {
            
        }
    
    def get_keys_to_action(self):
        """
        Get the mapping of keys to actions.

        Returns:
        - keys_to_action (dict): The mapping of keys to actions.
        """
        return self.keys_to_action
    
    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.

        Returns:
        - observation (np.ndarray): The initial observation of the environment.
        """
        

        # Randomly initialize the ball's position within the safe bounds
        initial_x = self.np_random.uniform(low=self.left_bound, high=self.right_bound)
        initial_y = self.np_random.uniform(low=self.bottom_bound, high=self.top_bound)
        
        # Randomly initialize the ball's velocity
        
        initial_velocity_x = self.np_random.uniform(low=-self.velocity_initial_size, high=self.velocity_initial_size)
        initial_velocity_y = self.np_random.uniform(low=-self.velocity_initial_size, high=self.velocity_initial_size)

        # Set the initial state
        self.state = np.array([initial_x, initial_y, initial_velocity_x, initial_velocity_y], dtype=np.float32)
        
        if self.render_mode == "human":
            self._render_frame()
        
        observation = self._get_obs()
        return observation


    def step(self, action):
        """
        Take a step in the environment based on the given action.
        First, use the current velocity to move. If touches the wall, reverse the velocity and reduce the energy. But the position is still on the wall.
        After that, at next call to `step()`, the agent will begin at the wall and has a new, reversed velocity.
        Args:
        - action (np.ndarray): The action to take in the environment.

        Returns:
        - observation (np.ndarray): The new observation of the environment.
        - reward (float): The reward for the current step.
        - done (bool): Whether the episode is done or not.
        - info (dict): Additional information about the environment.
        """
        # Normalize the action to ensure it's a unit vector
        action_direction = action / np.linalg.norm(action)
        
        # Adjust the ball's velocity based on the action
        self.state[2] += action_direction[0] * self.velocity_change_factor
        self.state[3] += action_direction[1] * self.velocity_change_factor
        
        # print(f"bottom bound: {self.bottom_bound}, left bound: {self.left_bound}, top bound: {self.top_bound}, right bound: {self.right_bound}")
        # print(f"Current velocity: {self.state[2:]} \nCurrent position: {self.state[:2]} \nCurrent energy: {np.linalg.norm(self.state[2:])}")
        
        
        # Apply old velocity to calculate the new position
        # Here we assume each step has unit time duration. So s = v * t.
        self.state[:2] += self.state[2:]
        next_position = (self.state[:2]).copy() # The idea next position (unclipped), used for collision detection
        # print(f"next_position: {next_position}")
        
        # Ensure the ball's position is within the environment bounds
        self.state[:2] = np.clip(self.state[:2], self.left_bound, self.top_bound)
        

        # Initialize reward and done flag
        reward = 0
        done = False

        
        # Check collisions with adjustments for ball radius
        collision = False
        if next_position[0] <= self.left_bound or next_position[0] >= self.right_bound:
            self.state[2] = -self.state[2] * self.energy_loss_factor  # Reverse X velocity
            collision = True
        if next_position[1] <= self.bottom_bound or next_position[1] >= self.top_bound:
            self.state[3] = -self.state[3] * self.energy_loss_factor  # Reverse Y velocity
            collision = True

        
        if self.log:
            print(f"New velocity: {self.state[2:]}\nNew position: {self.state[:2]}\nNew energy: {np.linalg.norm(self.state[2:])}")
            if collision:
                reward = 10  # Reward for hitting a wall
                print(f"Collision detected!")
                print(f"=====> reward: {reward}")
            else:
                print("No collision detected")
        
        # Check if the ball's velocity is effectively zero
        if np.linalg.norm(self.state[2:]) < self.min_velocity:
            done = True  # End the episode if the ball has stopped moving
        
        # Update the observation with the current state
        observation = self.state

        info = {
            "message": "Ball has stopped" if done else "In motion",
            "Energy": {np.linalg.norm(self.state[2:])}
            }

        if self.render_mode == "human":
            self._render_frame()
            
        # if self.log:
            # if collision:
                # print(f"Collision detected! =====> reward: {reward}")
            # print(f"Velocity: {self.state[2:]}, Position: {self.state[:2]}, Energy: {np.linalg.norm(self.state[2:])}")
        return observation, reward, done, info

    
    def render(self, mode="human"):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background

    
        # Convert these ratios to pixel dimensions for rendering
        ball_radius_pixels = self.ball_radius_ratio * self.window_size
        wall_thickness_pixels = self.wall_thickness_ratio * self.window_size

        # Draw walls around the perimeter using the pixel dimensions
        wall_color = (0, 0, 0)

        # Top wall
        pygame.draw.rect(canvas, wall_color, pygame.Rect(0, 0, self.window_size, wall_thickness_pixels))
        # Bottom wall
        pygame.draw.rect(canvas, wall_color, pygame.Rect(0, self.window_size - wall_thickness_pixels, self.window_size, wall_thickness_pixels))
        # Left wall
        pygame.draw.rect(canvas, wall_color, pygame.Rect(0, 0, wall_thickness_pixels, self.window_size))
        # Right wall
        pygame.draw.rect(canvas, wall_color, pygame.Rect(self.window_size - wall_thickness_pixels, 0, wall_thickness_pixels, self.window_size))

        # Draw the agent (ball)
        agent_location = self.state[:2]
        agent_color = (0, 0, 255)  # Blue for the agent
        # Convert agent location to pixel coordinates, adjusting for ball radius to keep it within bounds
        agent_center_x = (agent_location[0] * self.window_size / self.size)
        agent_center_y = (agent_location[1] * self.window_size / self.size)
        agent_center = (int(agent_center_x), int(agent_center_y))
        
        radius = int(ball_radius_pixels)
        
        # print("==========================")
        # print("Radius: ", radius)
        # print("Agent center: ", agent_center)
        
        
        # print("Radius (size): ", self.ball_radius_ratio * self.size)
        # print("Agent center (size): ", (agent_location[0], agent_location[1]))
        # print("Wall thickness(size): ", self.wall_thickness_ratio * self.size)
        # print("==========================")
        
        
        pygame.draw.circle(canvas, agent_color, agent_center, radius)
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            returned_img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            return returned_img
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    