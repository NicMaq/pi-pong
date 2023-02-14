import gym
from gym import spaces
import math
import numpy as np
import cv2
from skimage.transform import resize
import imageio
import tensorflow as tf
from datetime import datetime

import logging

from sense_hat import SenseHat

log = logging.getLogger()
log.setLevel(logging.DEBUG)


N_DISCRETE_ACTIONS = 3 # Do nothing (0), Go Left (1), Right (2)
N_SIZE = 8
SAVE_DIR = '/home/nicolas/Workspace/ml/logs/PI_sense'


class PongEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}

    class Paddle():

        def __init__(self):

            self.size = 3
            self.location = 4
            self.span = (self.size//2)
            self.states = self.get_states()
        
        def move(self, action):

            # Do nothing (0), Go Left (1), Right (2)
            if action == 1 and self.location >= self.span: self.location -=1
            if action == 2 and self.location < N_SIZE - self.span: self.location +=1 

        def get_state(self,):

            _states = np.arange(8)
            _states = np.where(_states <  self.location + self.span and _states > self.location - self.span, 1, 0)

            return _states

    class Ball():

        def __init__(self, init_pos, init_vector, scale):

            self.location_init = init_pos
            self.location = self.location_init
            self.vector_init = init_vector
            self.vector = self.vector_init
            self.scale = scale
            _angle = 0.6
            self.rotation_left = [[math.cos(_angle), -math.sin(_angle)][math.sin(_angle), math.cos(_angle)]]
            self.rotation_right = [[math.cos(-_angle), -math.sin(-_angle)][math.sin(-_angle), math.cos(-_angle)]]
        
        def move(self, states):

            is_out, is_collided = False, False

            _move = self.scale * self.vector
            _slope = _move[1]/_move[0]

            #The ball is moving down
            if _move[1] > 0: 

                #The ball reflects on the wall
                if self.location[0] + _move[0] < 0:
                if self.location[0] + _move[0] > 7:

                #The ball moves beyond the paddle
                if self.location[1] + _move[1] >= 6:
                    #Does it collided with the Paddle
                    x_on_paddle_line_is = int((6-self.location[1]+_slope*self.location[0])/_slope)
                    is_collided = (states[6][x_on_paddle_line_is] == 1)
                    is_out = (int(_move[1] + self.location[1])>= 7)
                     
                    #If collided, the ball bounces on the padlle
                    if is_collided: 
                        #If the ball touches the paddle on the left or the right then the reflection is modified
                        _residual_move = 
                        self.location =
                        self.vector = 
                    #If not 
                    if is_out: 
                        self.location = self.location_init
                        self.vector = self.vector_init
                    else: 
                        self.location = self.location + _move

                if self.location[1] + _move[1] <= 1:

            #The ball is moving up
            else:
                pass




            return is_out, is_collided            


    def __init__(self, **kwargs):

        super(PongEnv, self).__init__()

        self.now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.nsteps = 0

        self.sense = SenseHat()
        self.sense.show_message("Welcome to Bonsai Pong!")

        #Set action and state spaces

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        
        # Number of leds on the PI Sense
        self.height = N_SIZE
        self.width = N_SIZE

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)    

        self.paddle_top = PongEnv.Paddle()
        self.paddle_bottom = PongEnv.Paddle()

        self.states = self.get_states()
        print("state is: \n %s",self.states )

        self.ball_direction = 0

        log.info(f'{"Observation type is: {}".format(type(self.observations_init[0]))}')
        log.info(f'{"Observation 0 shape is: {}".format(self.observations_init[0].shape)}')

        #Keep all states to generate anumated gif
        self.raw_images = []

        cv2.imwrite(f'{SAVE_DIR}{"/observation-{}-Steps_{}.jpg".format(self.now, self.num_forms_discovered, self.nsteps)}', self.observation)
                                           

    def step(self, action):

        self.nsteps += 1        

        isAlreadyVisited, isOnForm, isOnNewForm, isOnPerimeter, isEndForm = self.new_landing_desc(action)
        # Add new point to self.observation
        self.observation[self.last_point[0],self.last_point[1],0]=255
        #self.raw_images.append(self.observation.copy())
        self.raw_images.append(self.observation.copy())

        step_reward = self.calculate_reward(isAlreadyVisited, isOnForm, isOnNewForm, isOnPerimeter, isEndForm)
        self.episode_reward += step_reward
        #truncated = self.nsteps > 5000
        done = isOnPerimeter or isEndForm or self.nsteps > 5000

        #if isOnNewForm or isEndForm or isOnForm or isOnPerimeter: 
        #    self.generate_gif()        

        if done: 
            
            if self.max_forms_discovered[self.selected_obs] < self.num_forms_discovered:
                self.max_forms_discovered[self.selected_obs] = self.num_forms_discovered 
                self.generate_gif()

            log.info(f'{"Episode ended after {} steps - score is {} ".format(self.nsteps, self.episode_reward)}')
            log.info(f'{"Trajectory ended bc: isOnPerimeter={} isEndForm={} ".format(isOnPerimeter, isEndForm)}')

        info = {'steps': self.nsteps, 'selected_obs': self.selected_obs, 'discovered_forms': self.num_forms_discovered, 'success': isEndForm}

        #cv2.imwrite(f'{SAVE_DIR}{"/returned_Obs-{}-Forms-{}-Steps_{}.jpg".format(self.now, self.num_forms_discovered, self.nsteps)}', self.observation)

        return self.observation, step_reward, done, info


    def reset(self):
        self.nsteps = 0
        self.raw_images = []

        self.paddle_top.location = 4
        self.paddle_bottom.location = 4

        return self.states 

    def get_states(self,):

        # State is either 0 (led is turned off) or 1 (let is on)- init to all off   
        _states = np.zeros([self.height,self.width], dtype = int)    
        _pad_zeros = np.zeros([self.height-2,self.width], dtype = int)
        _paddle_layer = np.stack((self.paddle_top.states, _pad_zeros, self.paddle_bottom.states),axis=0)
        _states = self.states + _paddle_layer

        return _states





    def render(self, mode='human'):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Outsole", self.observation)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            image = self.observation.copy()
            return image

    
    def close(self):
        cv2.destroyAllWindows()


    def calculate_reward(self, isAlreadyVisited, isOnForm, isOnNewForm, isOnPerimeter, isEndForm):

        step_reward = 0

        #if new point already visited then penalty of -100
        if isAlreadyVisited: step_reward += -5        
        #if new point not on form then penalty of -1
        if not isOnForm: step_reward += -0.1
        #if new point not on form then penalty of -1
        if isOnForm: step_reward += 0.0
        #if new point on perimeter then penalty of -100
        if isOnPerimeter: step_reward += -20
        #if new point on new form then penalty of +10
        if isOnNewForm: step_reward += 10 #+ self.num_forms_discovered * 2
        #if new point end_point then penalty of +100
        if isEndForm: step_reward += 20     

        #print('step_reward', step_reward)

        return step_reward

    def generate_gif(self):
      
        self.raw_images = [image.astype('uint8') for image in self.raw_images]
        new_forms = np.expand_dims(self.new_forms, axis=-1)
        cv2.imwrite(f'{SAVE_DIR}{"/Nike-New_Forms-{}-{}-{}-Forms-{}.jpg".format(self.now, self.selected_obs, int(self.episode_reward), self.num_forms_discovered)}', new_forms)
        imageio.mimsave(f'{SAVE_DIR}{"/Nike-{}-{}-{}-Forms-{}.gif".format(self.now, self.selected_obs, int(self.episode_reward), self.num_forms_discovered)}', self.raw_images, duration=1/30)

