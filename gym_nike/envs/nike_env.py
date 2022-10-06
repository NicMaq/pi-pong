import gym
from gym import spaces

import numpy as np
import cv2
from skimage.transform import resize
import imageio
import tensorflow as tf
from datetime import datetime

import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)


N_DISCRETE_ACTIONS = 4 #Go up (0), right (1), down (2) and left (3)
N_CHANNELS = 3
SAVE_DIR = '~/Workspace/ml/openai_logs/Nike'


class NikeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):

        super(NikeEnv, self).__init__()

        self.now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.nsteps = 0
        print("kwargs received by Nike_env are:is: %s " % kwargs )
        assert 'config' in kwargs, "Invalid mode, must provide a config for this env"
        file = kwargs['config']
        config, extension = file.split(".")
        print(config)
        print(extension)
        self.nconfig = kwargs['nconfig']
        print(self.nconfig)
        log.debug(f'{"Config is: {}".format(config)}')
        print("Config is: %s " % config )

        # Define action and observation space
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Using a design as input:
        # the decoded images will have the channels stored in B G R order.
        self.observation_init = []
        
        #self.observation_init.append(cv2.imread(config))
        for i in range(self.nconfig):

            print(f'{"{}_{}.{}".format(config, i, extension)}')          

            #clean image on all channel
            read_image = cv2.imread(f'{"{}{}.{}".format(config, i, extension)}')
            cv2.imwrite(f'{SAVE_DIR}{"/read_image-{}-{}.jpg".format(self.now,i)}', read_image)
            #print('type(read_image)',read_image.dtype)

            self.observation_init.append(read_image)

        self.height = self.observation_init[0].shape[0]
        self.width = self.observation_init[0].shape[1]

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)            

        log.info(f'{"Observation type is: {}".format(type(self.observation_init[0]))}')
        log.info(f'{"Observation 0 shape is: {}".format(self.observation_init[0].shape)}')

        #Randomly select which observation to use
        self.selected_obs = np.random.randint(self.nconfig, size=1)[0]
        #print('self.selected_obs',self.selected_obs)

        #Extract and randomize the forms
        self.set_forms()
        self.observation = self.grey_obs[self.selected_obs].copy()
        self.new_forms = self.forms[self.selected_obs].copy()

        self.raw_images = []
        self.raw_images.append(self.observation.copy())

        # The start/stop points are the uppest form point for start the lowest for stop 
        #self.start_point, _ = self.find_points()
        self.start_point = (10, self.width//2)
        log.info(f'{"self.start_point is: {}".format(self.start_point)}')

        #print('self.end_point is ', self.end_point)
        self.last_point = self.start_point
        #print('self.start_point',self.start_point)

        self.previous_points = []
        self.previous_points.append(self.start_point)

        self.num_forms_discovered = 0
        self.max_forms_discovered = [0 for obs in self.observation_init]

        cv2.imwrite(f'{SAVE_DIR}{"/observation-{}-Forms-{}-Steps_{}.jpg".format(self.now, self.num_forms_discovered, self.nsteps)}', self.observation)
                                           

    def step(self, action):

        self.nsteps += 1        
        #print('action', action)
        #print('self.nsteps', self.nsteps)

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
            log.info(f'{"Episode ended after {} steps - score is {} ".format(self.nsteps, self.episode_reward)}')
            log.info(f'{"Trajectory ended bc: isOnPerimeter={} isEndForm={} ".format(isOnPerimeter, isEndForm)}')

        info = {'steps': self.nsteps, 'selected_obs': self.selected_obs, 'discovered_forms': self.num_forms_discovered, 'success': isEndForm}

        #cv2.imwrite(f'{SAVE_DIR}{"/returned_Obs-{}-Forms-{}-Steps_{}.jpg".format(self.now, self.num_forms_discovered, self.nsteps)}', self.observation)

        return self.observation, step_reward, done, info


    def reset(self):
        self.nsteps = 0
        self.selected_obs = np.random.randint(self.nconfig, size=1)[0]
        #print('self.selected_obs',self.selected_obs)

        #Extract and randomize the forms
        self.set_forms()
        
        self.observation = self.grey_obs[self.selected_obs].copy()
        self.new_forms = self.forms[self.selected_obs].copy()

        self.raw_images = []
        self.raw_images.append(self.observation.copy())

        self.num_forms_discovered = 0
        self.current_length = 0
        self.episode_reward = 0

        self.previous_points = []
        self.last_point = self.start_point
        self.previous_points.append(self.start_point)

        return self.observation  # reward, done, info can't be included


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
        imageio.mimsave(f'{SAVE_DIR}{"/Nike-{}-{}-{}-Forms-{}.gif".format(self.now, self.selected_obs, int(self.episode_reward), self.num_forms_discovered)}', self.raw_images, duration=1/30)


    def find_points(self):
        # The start/stop points are the uppest form point for start the lowest for stop 

        start_point = (0,0)
        end_point = (0,0)
        found_start = False
        last_coordinate = (0,0)

        it = np.nditer(self.forms, flags=['multi_index'])
        for x in it:
            if x==255 and not found_start:
                #print("%d <%s>" % (x, it.multi_index), end=' ')
                found_start = True
                start_point = it.multi_index
                last_coordinate = start_point
            
            if x==255:
                last_coordinate = it.multi_index

        end_point = last_coordinate
        start_point = (start_point[0]-5,start_point[1]-5)
        end_point = (end_point[0]+1,end_point[1])

        return start_point, end_point


    def new_landing_desc(self, action):
        #print('action is',action)

        # Go up (0), right (1), down (2) and left (3)
        #print('self.last_point is',self.last_point)
        if action == 0: self.last_point = (self.last_point[0]-1, self.last_point[1])
        if action == 1: self.last_point = (self.last_point[0], self.last_point[1]+1)
        if action == 2: self.last_point = (self.last_point[0]+1, self.last_point[1])
        if action == 3: self.last_point = (self.last_point[0], self.last_point[1]-1)
        #print('new last_point is',self.last_point) 

        isAlreadyVisited = self.last_point in self.previous_points
        #print('isAlreadyVisited is',isAlreadyVisited)
        #print('self.previous_points is',self.previous_points)

        if not isAlreadyVisited: 
            self.previous_points.append(self.last_point)
            self.current_length += 1

        isOnForm = (self.forms[self.selected_obs][self.last_point[0],self.last_point[1]]==255)
        #print('isOnForm',isOnForm)
        isOnNewForm =  (self.new_forms[self.last_point[0],self.last_point[1]]==255)
        #print('isOnNewForm',isOnNewForm)
        if isOnNewForm: 
            #cv2.imwrite(f'{SAVE_DIR}{"/landing_new_forms0-{}-Forms-{}.jpg".format(self.now, self.num_forms_discovered)}', self.new_forms)
            self.new_forms = self.remove_form(self.new_forms, self.last_point)
            #cv2.imwrite(f'{SAVE_DIR}{"/landing_new_forms1-{}-Forms-{}.jpg".format(self.now, self.num_forms_discovered)}', self.new_forms)
            self.num_forms_discovered += 1
            if self.max_forms_discovered[self.selected_obs] < self.num_forms_discovered:
                self.max_forms_discovered[self.selected_obs] = self.num_forms_discovered
                self.generate_gif()
            #name = "new_forms-{}.jpg".format(self.nsteps)
            #cv2.imwrite(name, self.new_forms)
        #print('isOnNewForm',isOnNewForm)

        isOnPerimeter =  (self.perimeters[self.selected_obs][self.last_point[0],self.last_point[1]]==255)
        #print('isOnPerimeter',isOnPerimeter)

        isEndForm =  self.IsItLastForm()
        #print('isEndPoint',isEndPoint)

        return isAlreadyVisited, isOnForm, isOnNewForm, isOnPerimeter, isEndForm  

    
    def set_forms(self):

        # The forms are on channel green 
        self.forms = [self.clean(np.squeeze(obs[:,:,1].copy())) for obs in self.observation_init]
        #print('type(self.forms[0])',self.forms[0].dtype)
        
        # Moving the forms randomly
        move_axis_0 = np.random.randint(7, size=1)[0]-3
        move_axis_1 = np.random.randint(7, size=1)[0]-3
        self.forms = [np.roll(form, (move_axis_0, move_axis_1), axis=(0, 1)) for form in self.forms]

        self.grey_forms = [np.expand_dims(np.where(a == 255, 80, 0), axis=-1) for a in self.forms]
        #print('type(self.grey_forms[0])',self.grey_forms[0].dtype)
        

        # The perimeter is on channel red 
        self.perimeters = [self.clean(np.squeeze(obs[:,:,2].copy())) for obs in self.observation_init]
        self.grey_perimeters = [np.expand_dims(np.where(a == 255, 160, 0), axis=-1) for a in self.perimeters]
        

        # The observations are build in gray scale
        self.grey_path = [np.zeros_like(a) for a in self.grey_forms]
        self.grey_obs = [ a+b+c for a,b,c in zip(self.grey_path, self.grey_forms, self.grey_perimeters)]
        #print('type(self.grey_obs[0])',self.grey_obs[0].dtype)

        #cv2.imwrite(f'{SAVE_DIR}{"/grey_forms-{}.jpg".format(self.nsteps)}', self.grey_forms[0])
        #cv2.imwrite(f'{SAVE_DIR}{"/grey_perimeter-{}.jpg".format(self.nsteps)}', self.grey_perimeters[0])
        #cv2.imwrite(f'{SAVE_DIR}{"/grey_obs-{}.jpg".format(self.nsteps)}', self.grey_obs[0])



    def remove_form(self, newforms, point):

            newforms[point[0],point[1]] = 0

            point_up = (point[0]-1,point[1])
            if newforms[point_up[0],point_up[1]] == 255:
                    newforms = self.remove_form(newforms, point_up)
            point_right = (point[0],point[1]+1)
            if newforms[point_right[0],point_right[1]] == 255:
                    newforms = self.remove_form(newforms, point_right)       
            point_down = (point[0]+1,point[1])
            if newforms[point_down[0],point_down[1]] == 255:
                    newforms = self.remove_form(newforms, point_down)    
            point_left = (point[0],point[1]-1)
            if newforms[point_left[0],point_left[1]] == 255:
                    newforms = self.remove_form(newforms, point_left)                                                 

            return newforms


    def IsItLastForm(self):  
        isLast = True
        it = np.nditer(self.new_forms, flags=['multi_index'])
        for x in it:
            if x==255:
                isLast = False
                break

        return isLast            


    def clean(self, image):
        clean = image

        it = np.nditer(image, flags=['multi_index'])
        for x in it:
            if x>100:
                #print("%d <%s>" % (x, it.multi_index), end=' ')
                clean[it.multi_index[0],it.multi_index[1]]=255
            if x<=100:
                #print("%d <%s>" % (x, it.multi_index), end=' ')
                clean[it.multi_index[0],it.multi_index[1]]=0                


        return clean            

    