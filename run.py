import numpy as np
import matplotlib.pyplot as plt
from pr2_utils import read_data_from_csv, mapCorrelation, bresenham2D, tic, toc, compute_stereo
from scipy.interpolate import interp1d
import os

# etimestamp, edata = read_data_from_csv('data/sensor_data/encoder.csv')
# ltimestamp, ldata = read_data_from_csv('data/sensor_data/lidar.csv')
# ftimestamp, fdata = read_data_from_csv('data/sensor_data/fog.csv')
# print('Done')

class PFSLAM:
    def __init__(self, epath='data/sensor_data/encoder.csv', fpath='data/sensor_data/fog.csv', lpath='data/sensor_data/lidar.csv', num_particle=1000) -> None:
        self.etime, self.edata = read_data_from_csv(epath)
        self.ftime, self.fdata = read_data_from_csv(fpath)
        self.ltime, self.ldata = read_data_from_csv(lpath)
        
        self.imgs_left = os.listdir(os.path.join('data/stereo_images/stereo_left')); self.imgs_left.sort()
        self.imgs_right = os.listdir(os.path.join('data/stereo_images/stereo_right')); self.imgs_right.sort()
        ctime = np.zeros(len(self.imgs_left)).astype(int)
        for i in range(len(self.imgs_left)):
            ctime[i] = int(self.imgs_left[i][:-4])
        ctime.sort()
        # discard the first timestamp
        self.ctime = ctime[1:]
        
        self.lwd = 0.623479
        self.rwd = 0.622806
        self.wb = 1.52439
        self.eres = 4096

        # self.fR = np.identity(3)
        # self.fT = np.array([-0.335, -0.035, 0.78])

        self.lR = np.array([0.00130201, 0.796097, 0.605167, 0.999999, -0.000419027, -0.00160026, -0.00102038, 0.605169, -0.796097]).reshape((3, 3))
        self.lT = np.array([[0.8349, -0.0126869, 1.76416]]).T

        self.cR = np.array([-0.00680499, -0.0153215, 0.99985, -0.999977, 0.000334627, -0.00680066, -0.000230383, -0.999883, -0.0153234]).reshape((3, 3))
        self.cT = np.array([[1.64239, 0.247401, 1.58411]]).T

        # time sychronization
        self.yaw = np.cumsum(self.fdata[:, 2])
        f = interp1d(self.ftime, self.yaw)
        self.sync_yaw = f(self.ltime)
        f = interp1d(self.etime, self.edata[:, 0])
        sync_lenc = (f(self.ltime[1:]) - self.edata[0, 0]) * np.pi * self.lwd / self.eres
        f = interp1d(self.etime, self.edata[:, 1])
        sync_renc = (f(self.ltime[1:]) - self.edata[0, 1]) * np.pi * self.rwd / self.eres
        self.sync_edata = np.stack((sync_lenc, sync_renc)).T
        
        # self.sync_yaw = self.sync_yaw[1:]
        # self.sync_yaw = self.sync_yaw[1:] - self.sync_yaw[1]
        self.sync_dyaw = self.sync_yaw[2:] - self.sync_yaw[1:-1]
        self.first_lidar = self.ldata[1, :]
        self.ldata = self.ldata[2:, :]
        self.ltime = self.ltime[2:]
        self.step = len(self.ldata)
        
        self.angles = np.linspace(-5, 185, 286) / 180 * np.pi
        self.sins = np.sin(self.angles)
        self.coss = np.cos(self.angles)

        # camera parameters
        P = np.zeros((4,4))
        P[:3,:] = np.array([ 7.7537235550066748e+02, 0., 6.1947309112548828e+02, 0., 
        0., 7.7537235550066748e+02, 2.5718049049377441e+02, 0., 
        0., 0., 1., 0. ]).reshape((3,4))
        P[2:,:] = np.array([[0, 0, 0, 3.6841758740842312e+02], [0, 0, 1, 0]])
        self.invP = np.linalg.inv(P)

        x = np.linspace(0, 1279, 1280)
        y = np.linspace(0, 559, 560)
        u, v = np.meshgrid(x, y)
        self.u = u.reshape(560*1280)
        self.v = v.reshape(560*1280)
        self.ones = np.ones((560*1280))

        # noise parameters
        self.nvmu = 0
        self.nvsigma = 1e-1
        self.ntmu = 0
        self.ntsigma = 5e-5

        self.num_particle = num_particle
        self.particles = np.zeros([num_particle, 3])
        self.pweights = np.ones([num_particle]) / num_particle
        self.Nth = self.num_particle / 10

        self.trajectory = np.zeros((self.step, 3))
        self.next_stereo_idx = 0

        # map
        MAP = {}
        MAP['res']   =  1 #meters
        MAP['xmin']  = -50  #meters
        MAP['ymin']  = -1100
        MAP['xmax']  =  1300
        MAP['ymax']  =  50 
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res']) + 1) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res']) + 1)
        MAP['logit'] = np.zeros((MAP['sizex'],MAP['sizey']), dtype=np.int8) #DATA TYPE: char or int8
        MAP['colored_map'] = np.zeros((MAP['sizex'],MAP['sizey'],3), dtype=np.uint8)
        MAP['occupancy'] = np.zeros((MAP['sizex'],MAP['sizey']), dtype=np.bool)
        MAP['x_im'] = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
        MAP['y_im'] = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
        # MAP['x_range'] = np.arange(-0.5, 0.5+0.1, 0.1)
        # MAP['y_range'] = np.arange(-0.5, 0.5+0.1, 0.1)
        MAP['x_range'] = np.arange(-0.5, 0.5+0.25, 0.25)
        MAP['y_range'] = np.arange(-0.5, 0.5+0.25, 0.25)
        MAP['lambda_max'] = 10
        self.map = MAP
  
    def getRM2D(self, theta):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        return R

    def deadreckoning(self):
        # trajectory = np.zeros((self.step, 3))
        yaw = 0
        # ds = np.mean(self.sync_edata[0,:])
        # trajectory[0,:] = np.array([ds*np.cos(self.sync_yaw[0]), ds*np.sin(self.sync_yaw[0])])
        for i in range(1, self.step):
            ds = np.mean(self.sync_edata[i,:]) - np.mean(self.sync_edata[i-1,:])
            yaw += self.sync_dyaw[i]
            self.trajectory[i,:2] = self.trajectory[i-1,:2] + ds * np.array([np.cos(yaw), np.sin(yaw)])
            self.trajectory[i,-1] = yaw
            
        # return np.cumsum(trajectory, axis=0)

    def predict(self, i):
        '''
        k: step index
        '''
        for p in range(self.num_particle):
            ds = np.mean(self.sync_edata[i+1,:]) - np.mean(self.sync_edata[i,:])
            ds += np.random.normal(self.nvmu, self.nvsigma)
            self.particles[p,-1] += (self.sync_dyaw[i] + np.random.normal(self.ntmu, self.ntsigma))
            self.particles[p,:2] += ds * np.array([np.cos(self.particles[p,-1]), np.sin(self.particles[p,-1])])
        
    def update(self, i):
        ranges = self.ldata[i, :]

        # take valid indices
        indValid = np.logical_and((ranges < 80),(ranges > 0.1))
        ranges = ranges[indValid]
        angles = self.angles[indValid]

        # xy position in the sensor frame
        xs0 = ranges*np.cos(angles)
        ys0 = ranges*np.sin(angles)
        # xy position in the world frame (needs adding current vehicle pos)
        points2 = np.matmul(self.lR[:2,:2], np.vstack((xs0,ys0)))
        # points2 = points2 + self.lT[:2]

        # timestamp = tic()

        c = np.zeros((self.num_particle))
        for p in range(self.num_particle):
            R = self.getRM2D(self.particles[p,-1])
            # convert from meters to cells
            scanned = points2 + self.particles[p:p+1,:2].T + R @ self.lT[:2]
            # xis = np.ceil((scanned[0,:] - self.map['xmin']) / self.map['res'] ).astype(np.int16) - 1
            # yis = np.ceil((scanned[1,:] - self.map['ymin']) / self.map['res'] ).astype(np.int16) - 1

            # indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < self.map['sizex'])), (yis < self.map['sizey']))
            # map = np.zeros((self.map['sizex'],self.map['sizey']),dtype=np.int8)
            # map[xis[indGood],yis[indGood]] = 1
            # compute map correlation
            map_corr = mapCorrelation(self.map['map'],self.map['x_im'],self.map['y_im'],scanned,self.map['x_range'],self.map['y_range'])
            c[p] = np.amax(map_corr)
            # ix, iy = np.unravel_index(np.argmax(map_corr), map_corr.shape)
            # self.particles[p,:2] += np.array([self.map['x_range'][ix], self.map['y_range'][iy]])


        # toc(timestamp)

        # update weights
        self.pweights = self.pweights * np.exp(c)
        self.pweights /= np.sum(self.pweights)

        # update map logits
        idx = np.argmax(self.pweights)
        scanned = points2 + self.particles[idx:idx+1,:2].T
        cu_x, cu_y = self.particles[idx,0], self.particles[idx,1]
        xis = np.ceil((cu_x - self.map['xmin']) / self.map['res'] ).astype(np.int16) - 1
        yis = np.ceil((cu_y - self.map['ymin']) / self.map['res'] ).astype(np.int16) - 1
        for i in range(scanned.shape[1]):
            xie = np.ceil((scanned[0,i] - self.map['xmin']) / self.map['res'] ).astype(np.int16) - 1
            yie = np.ceil((scanned[1,i] - self.map['ymin']) / self.map['res'] ).astype(np.int16) - 1
            trace = bresenham2D(xis, yis, xie, yie)
            xi = trace[0,:-1]
            yi = trace[1,:-1]
            indGood = np.logical_and(np.logical_and(np.logical_and((xi > 1), (yi > 1)), (xi < self.map['sizex'])), (yi < self.map['sizey']))
            self.map['logit'][xi[indGood],yi[indGood]] -= 1
            try:
                self.map['logit'][xie,yie] += 1
            except:
                pass
            # self.map['logit'][passed[0,:-1],passed[1,:-1]] -= 1
            # self.map['logit'][passed[0,-1],passed[1,-1]] +=1
        self.map['logit'][self.map['logit']>self.map['lambda_max']] = self.map['lambda_max']
        self.map['logit'][self.map['logit']<-self.map['lambda_max']] = -self.map['lambda_max']
        self.map['occupancy'] = (self.map['logit'] > 0)


    def resample(self):
        var = np.sum(self.pweights * self.pweights)
        if 1/var <= self.Nth:
            N = self.num_particle
            th = np.cumsum(self.pweights)
            r = np.random.uniform(0, 1/N, size=N) + np.arange(0, 1, 1/N)
            new_particles = np.zeros((N, 3))
            j = 0
            for i in range(N):
                while r[i]>th[j]:
                    j += 1
                new_particles[i,:] = self.particles[j,:]
            self.particles = new_particles
            self.pweights = np.ones([N]) / N

    def initial_mapping(self):
        ranges = self.ldata[0, :]
        
        # take valid indices
        indValid = np.logical_and((ranges < 80),(ranges > 0.1))
        ranges = ranges[indValid]
        angles = self.angles[indValid]

        # x = ranges * self.coss
        # y = ranges * self.sins

        # xy position in the sensor frame
        xs0 = ranges*np.cos(angles)
        ys0 = ranges*np.sin(angles)
        # z = np.zeros([286])
        # points = np.vstack((x,y,z))
        # points = np.matmul(self.lR, points)
        scanned = np.matmul(self.lR[:2,:2], np.vstack((xs0,ys0)))
        scanned = scanned + self.lT[:2]

        # convert from meters to cells
        xis = np.ceil((scanned[0,:] - self.map['xmin']) / self.map['res'] ).astype(np.int16) - 1
        yis = np.ceil((scanned[1,:] - self.map['ymin']) / self.map['res'] ).astype(np.int16) - 1

        # update map logits
        xis = np.ceil((0 - self.map['xmin']) / self.map['res'] ).astype(np.int16) - 1
        yis = np.ceil((0 - self.map['ymin']) / self.map['res'] ).astype(np.int16) - 1
        for i in range(scanned.shape[1]):
            xie = np.ceil((scanned[0,i] - self.map['xmin']) / self.map['res'] ).astype(np.int16) - 1
            yie = np.ceil((scanned[1,i] - self.map['ymin']) / self.map['res'] ).astype(np.int16) - 1
            trace = bresenham2D(xis, yis, xie, yie)
            # self.map['logit'][passed[0,:-1],passed[1,:-1]] -= 1
            # self.map['logit'][passed[0,-1],passed[1,-1]] += 1
            xi = trace[0,:-1]
            yi = trace[1,:-1]
            indGood = np.logical_and(np.logical_and(np.logical_and((xi > 1), (yi > 1)), (xi < self.map['sizex'])), (yi < self.map['sizey']))
            self.map['logit'][xi[indGood],yi[indGood]] -= 1
            try:
                self.map['logit'][xie,yie] += 1
            except:
                pass

        self.map['logit'][self.map['logit']>self.map['lambda_max']] = self.map['lambda_max']
        self.map['logit'][self.map['logit']<-self.map['lambda_max']] = -self.map['lambda_max']
        self.map['map'] = (self.map['logit'] > 0)

    def slam(self):
        self.initial_mapping()
        timestamp = tic()
        for step in range(0, self.step-1):
            # timestamp = tic()
            self.predict(step)
            # toc(timestamp)

            if step % 100 == 0:
                # t = tic()
                self.update(step)
                # toc(t, disp=True)

            # timestamp = tic()
            self.resample()
            # toc(timestamp)
            # t = tic()
            self.build_texture_map(step)
            # toc(t, disp=True)

            self.trajectory[step+1,:] = np.mean(self.particles, axis=0)
            if step % 100==0:
                dt = toc(timestamp)
                print('step = {}, remaining {} seconds'.format(step, (self.step-step)/100*dt))
                timestamp = tic()

    def show_trajectory_map(self, step):
        # trajectory = np.cumsum(self.trajectory[:,:2], axis=0)
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        extent = [self.map['ymax'], self.map['ymin'], self.map['xmin'], self.map['xmax']]
        ax0.imshow(np.flipud(np.fliplr(self.map['occupancy'])), extent=extent, cmap='gray')
        ax0.set_title('Occupancy Map'); ax0.set_xlabel('y'); ax0.set_ylabel('x')
        ax1.imshow(np.flipud(np.fliplr(pfslam.map['colored_map'])), extent=extent)
        ax1.set_title('Textured Map'); ax1.set_xlabel('y'); ax1.set_ylabel('x')
        ax2.plot(self.trajectory[:step+1,1], self.trajectory[:step+1,0])
        ax2.set_ylim([self.map['xmin'], self.map['xmax']])
        ax2.set_xlim([self.map['ymin'], self.map['ymax']])
        ax2.invert_xaxis()
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title('Trajectory'); ax2.set_xlabel('y'); ax2.set_ylabel('x')
        plt.show()

    def build_texture_map(self, step):
        lt = self.ltime[step]
        if self.next_stereo_idx >= len(self.ctime):
            return
        ct =self.ctime[self.next_stereo_idx]
        if lt >= ct:
            f = interp1d(self.ltime[step-1:step+1], self.trajectory[step-1:step+1, 0])
            sync_x = f(ct)
            f = interp1d(self.ltime[step-1:step+1], self.trajectory[step-1:step+1, 1])
            sync_y = f(ct)
            f = interp1d(self.ltime[step-1:step+1], self.trajectory[step-1:step+1, 2])
            sync_theta = f(ct)
            sync_traj_to_image = np.stack((sync_x, sync_y, sync_theta)).T
            
            # for i in range(len(self.ctime)):
            i = self.next_stereo_idx
            self.next_stereo_idx += 1
            img_l, disparity = compute_stereo(self.imgs_left[i], self.imgs_right[i])
            # img_l, disparity = compute_stereo(ctime[i])
            img_l = img_l.reshape(560*1280, 3)
            disparity = disparity.reshape(560*1280)
            indValid = (disparity > 0)
            validPixels = np.vstack((self.u[indValid], self.v[indValid], disparity[indValid], self.ones[indValid]))
            validRGB = img_l[indValid,:]
            coor_cam = self.invP @ validPixels
            coor_cam = coor_cam / coor_cam[3,:]
            coor_cam = coor_cam[:3,:]
            theta = sync_traj_to_image[-1]
            R = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            cam_pos = sync_traj_to_image.reshape(3,1) + R @ self.cT
            coor_world = R @ self.cR @ coor_cam + cam_pos
            indHorizon = np.logical_and(0 <= coor_world[2,:], coor_world[2,:] <= 1)
            coor_horizon = coor_world[:,indHorizon]
            rgb_horizon = validRGB[indHorizon,:]
            # update colored map
            xi = np.ceil((coor_horizon[0,:] - self.map['xmin']) / self.map['res'] ).astype(np.int16) - 1
            yi = np.ceil((coor_horizon[1,:] - self.map['ymin']) / self.map['res'] ).astype(np.int16) - 1
            indGood = np.logical_and(np.logical_and(np.logical_and((xi > 1), (yi > 1)), (xi < self.map['sizex'])), (yi < self.map['sizey']))
            self.map['colored_map'][xi[indGood], yi[indGood]]=rgb_horizon[indGood,:]
            
            if self.next_stereo_idx % 100 == 0:
                self.show_trajectory_map(step)
            # print()
            

pfslam = PFSLAM(num_particle=100)
# pfslam.deadreckoning()
# pfslam.build_texture_map()
# pfslam.show_trajectory_map()
pfslam.slam()
pfslam.show_trajectory_map(pfslam.step)

# trajectory = pfslam.deadreckoning()
# plt.plot(trajectory[:,0], trajectory[:,1]); plt.show()
# pfslam.show_trajectory()
# fig, ax1 = plt.subplots(1, 1)
# ax1.imshow(pfslam.map['colored_map'])
# ax1.set_title('Textured Map')
# plt.show()
# plt.imshow(pfslam.map['colored_map']); plt.show()
print('DONE')