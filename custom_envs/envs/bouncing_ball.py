import numpy as np
import pygame
import time

import gym
from gym import spaces
from gym.utils import seeding

class BouncingBallEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=40):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.seed()
        
        # Define observation space (ball's position and velocity in 2D)
        self.observation_space = spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.float32)
        # Define action space (2D continuous: move in any direction)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Initialize state (position of the ball)
        self.state = None
        
        
        self.wall_thickness = 10 # Fixed thickness for walls
        self.ball_diameter = self.window_size / 30  # Adjust ball size as needed
        
        self.keys_to_action = {
            # Direction vectors represented as [x, y]
            "w": [0, -1],  # Up
            "a": [-1, 0],  # Left
            "s": [0, 1],   # Down
            "d": [1, 0],   # Right
        }

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
        # Temporarily set render_mode to 'rgb_array' for this operation, if necessary
        original_render_mode = self.render_mode
        self.render_mode = 'rgb_array'
        
        # Generate the observation as an RGB array
        obs = self._render_frame()

        # Reset the render mode to its original state, if needed
        self.render_mode = original_render_mode
        
        return obs
    
    def _get_info(self):
        return {
            
        }
    def get_keys_to_action(self):
        return self.keys_to_action
    
    def reset(self):
        # Define the safe area for ball initialization, considering wall thickness
        wall_thickness_ratio = 10 / self.window_size  # Assuming '10' is the wall thickness in pixels
        safe_margin = self.size * wall_thickness_ratio  # Convert wall thickness to environment scale
        
        # Define the safe bounds for ball initialization
        lower_bound = 0 + safe_margin
        upper_bound = self.size - 1 - safe_margin

        # Randomly initialize the ball's position within the safe bounds
        initial_x = self.np_random.uniform(low=lower_bound, high=upper_bound)
        initial_y = self.np_random.uniform(low=lower_bound, high=upper_bound)
        
        # Initialize the ball's velocity to 0 (if your model includes velocity)
        initial_velocity_x = 0
        initial_velocity_y = 0

        
        # Set the initial state
        self.state = np.array([initial_x, initial_y, initial_velocity_x, initial_velocity_y], dtype=np.float32)
        
        if self.render_mode == "human":
            self._render_frame()
        # Return the initial observation (ball's position and velocity)
        
        observation = self._get_obs()
        return observation


    def step(self, action):
        # The `step()` method must return four values: obs, reward, done, info
        
        # Normalize the action to get the direction vector with magnitude 1
        direction = action / np.linalg.norm(action)
        next_position = self.state[:2] + direction  # Update position based on the direction

    
        # Initialize variables
        reward = 0
        done = False

        # Check for wall collisions and adjust the position if necessary
        if np.any(next_position <= 0) or np.any(next_position >= self.size - 1):
            # Collision detected: Bounce back by reversing the direction
            # Note: This simple model assumes a perfect elastic collision without energy loss
            inverse_direction = -direction
            # Ensure the ball stays within bounds after the bounce
            # The ball moves back 1 unit in the inverse direction after collision
            self.state[:2] = np.clip(self.state[:2] + inverse_direction, 0, self.size - 1)
            reward = 1  # Assign reward for hitting and bouncing off the wall
            print(f"Collision! ======> reward: {reward}")
        else:
            # No collision: Update position normally
            self.state[:2] = np.clip(next_position, 0, self.size - 1)
            

        # Assuming 'done' remains False unless a specific termination condition is met
        observation = self._get_obs()
        info = self._get_info()  # Add any additional info if necessary

        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, done, info
    
    def render(self):
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

        # Calculate the size of the ball and wall thickness
        ball_diameter = self.window_size / 30  # Adjust ball size as needed
        wall_thickness = self.wall_thickness  # Fixed thickness for walls

        # Draw walls as thick lines around the perimeter
        wall_color = (0, 0, 0)  # Black for walls
        pygame.draw.line(canvas, wall_color, (0, 0), (self.window_size, 0), wall_thickness)  # Top wall
        pygame.draw.line(canvas, wall_color, (0, self.window_size), (self.window_size, self.window_size), wall_thickness)  # Bottom wall
        pygame.draw.line(canvas, wall_color, (0, 0), (0, self.window_size), wall_thickness)  # Left wall
        pygame.draw.line(canvas, wall_color, (self.window_size, 0), (self.window_size, self.window_size), wall_thickness)  # Right wall

        # Now we draw the agent (ball)
        agent_location = self.state[:2]
        agent_color = (0, 0, 255)  # Blue for the agent
        agent_center = (int(agent_location[0] * self.window_size / self.size), 
                        int(agent_location[1] * self.window_size / self.size))  # Convert agent location to pixel coords
        pygame.draw.circle(canvas, agent_color, agent_center, ball_diameter // 2)
        
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
    