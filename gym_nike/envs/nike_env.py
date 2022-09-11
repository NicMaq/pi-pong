import gym
from gym import spaces

import numpy as np
import cv2
from skimage.transform import resize
import imageio


N_DISCRETE_ACTIONS = 4 #Go up (0), right (1), down (2) and left (3)
N_CHANNELS = 3


class NikeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}

    def __init__(self, config):

        super(NikeEnv, self).__init__()

        self.nsteps = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Using a design as input:
        # the decoded images will have the channels stored in B G R order.
        self.observation_init = cv2.imread(config)
        print(type(self.observation_init))
        print(self.observation_init.shape)
        self.observation = self.observation_init.copy()

        self.height = self.observation_init.shape[0]
        self.width = self.observation_init.shape[1]

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, N_CHANNELS), dtype=np.uint8)
        
        # The forms are on channel green 
        self.forms = self.observation[:,:,1].copy()
        self.forms = np.squeeze(self.forms)
        self.forms = self.clean(self.forms)
        self.new_forms = self.forms.copy()

        # The perimeter is on channel red 
        self.perimeter = self.observation[:,:,2].copy() 
        self.perimeter = np.squeeze(self.perimeter)
        self.perimeter = self.clean(self.perimeter)   

        # The start/stop points are the uppest form point for start the lowest for stop 
        self.start_point, _ = self.find_points()
        print('self.start_point is ', self.start_point)
        #print('self.end_point is ', self.end_point)
        self.last_point = self.start_point

        #self.new_forms = self.remove_form(self.new_forms.copy(), self.last_point)
        #name = "new_forms-{}.jpg".format(self.nsteps)
        #cv2.imwrite(name, self.new_forms)

        #self.raw_images = []
        self.previous_points = []
        self.previous_points.append(self.start_point)

        self.num_forms_discovered = 0 
        

        print('End Init')
                                    

    def step(self, action):

        self.nsteps += 1

        isAlreadyVisited, isOnForm, isOnNewForm, isOnPerimeter, isEndForm = self.new_landing_desc(action)
        # Add new point to self.observation
        self.observation[self.last_point[0],self.last_point[1],0]=255
        #self.raw_images.append(self.observation.copy())

        step_reward = self.calculate_reward(isAlreadyVisited, isOnForm, isOnNewForm, isOnPerimeter, isEndForm)
        self.episode_reward += step_reward
        done = isOnPerimeter or isEndForm or self.nsteps > 5000
        info = {'steps': self.nsteps, 'discovered_forms': self.num_forms_discovered, 'success': isEndForm}

        return self.observation, step_reward, done, info


    def reset(self):
        self.nsteps = 0
        self.observation = self.observation_init.copy()       
        self.new_forms = self.forms.copy()
        #self.new_forms = self.remove_form(self.new_forms.copy(), self.last_point)
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

        #print('Raw_images #', len(self.raw_images))   
        #print('self.episode_reward', self.episode_reward )  
        #for idx, frame_idx in enumerate(self.raw_images): 
        #    self.raw_images[idx] = resize(frame_idx, (800, 340, 3), preserve_range=True, order=0).astype(np.uint8)
        #imageio.mimsave('GymNike.gif', self.raw_images, duration=1/30)


    def calculate_reward(self, isAlreadyVisited, isOnForm, isOnNewForm, isOnPerimeter, isEndForm):

        step_reward = 0

        #if new point already visited then penalty of -100
        if isAlreadyVisited: step_reward += -5        
        #if new point not on form then penalty of -1
        if not isOnForm: step_reward += -1
        #if new point not on form then penalty of -1
        if isOnForm: step_reward += 1
        #if new point on perimeter then penalty of -100
        if isOnPerimeter: step_reward += -100
        #if new point on new form then penalty of +10
        if isOnNewForm: step_reward += 5 + self.num_forms_discovered * 2
        #if new point end_point then penalty of +100
        if isEndForm: step_reward += 100        

        #print('step_reward', step_reward)

        return step_reward


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
        start_point = (start_point[0]-1,start_point[1])
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

        isOnForm = (self.forms[self.last_point[0],self.last_point[1]]==255)
        #print('isOnForm',isOnForm)
        #print('self.new_forms[self.last_point[0],self.last_point[1]]',self.new_forms[self.last_point[0],self.last_point[1]])
        isOnNewForm =  (self.new_forms[self.last_point[0],self.last_point[1]]==255)
        #print('isOnNewForm',isOnNewForm)
        if isOnNewForm: 
            #cv2.imwrite('landing_new_forms.jpg', self.new_forms)
            self.new_forms = self.remove_form(self.new_forms.copy(),self.last_point)
            self.num_forms_discovered += 1
            #name = "new_forms-{}.jpg".format(self.nsteps)
            #cv2.imwrite(name, self.new_forms)
        #print('isOnNewForm',isOnNewForm)

        isOnPerimeter =  (self.perimeter[self.last_point[0],self.last_point[1]]==255)
        #print('isOnPerimeter',isOnPerimeter)

        isEndForm =  self.IsItLastForm()
        #print('isEndPoint',isEndPoint)

        return isAlreadyVisited, isOnForm, isOnNewForm, isOnPerimeter, isEndForm  


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

    