import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.signal as sg
import pandas as pd

def get_lane_pertenence(Y, lanes_pose):
    
    min_pertenence = 0.1

    #Changing id's order because Y was transformed
    lanes_pose = sorted(lanes_pose)
    
    n_lanes = len(lanes_pose)
    
    if n_lanes == 4:
    
        l_w1 = (lanes_pose[1] - lanes_pose[0])
        l_w2 = (lanes_pose[3] - lanes_pose[2])

        if Y < lanes_pose[0] - l_w1/2.0:
            return [min_pertenence, 0.0, -1]
        elif Y < lanes_pose[1]:
            if Y < lanes_pose[0]:
                return [(l_w1 - np.abs(Y - lanes_pose[0]))/l_w1 , 0.0, -1]
            else:
                return [(l_w1 - np.abs(Y - lanes_pose[0]))/l_w1 , (l_w1 - np.abs(Y - lanes_pose[1]))/l_w1, -1]
        elif Y < (lanes_pose[1] + lanes_pose[2])/2:
            return [0.0, np.max([(l_w1 - np.abs(Y - lanes_pose[1]))/l_w1, min_pertenence]), -1]


        elif Y > lanes_pose[3] + l_w2/2.0:
            return [0.0, 0.0, -1]
        elif Y > lanes_pose[2]:
            if Y > lanes_pose[3]:
                return [(l_w2 - np.abs(Y - lanes_pose[3]))/l_w2 , 0.0, -1]
            else:
                return [(l_w2 - np.abs(Y - lanes_pose[3]))/l_w2 , (l_w2 - np.abs(Y - lanes_pose[2]))/l_w2, -1]
        else:
            return [0.0, np.max([(l_w2 - np.abs(Y - lanes_pose[2]))/l_w2, min_pertenence]), -1]

    else:

        l_w1 = (lanes_pose[1] - lanes_pose[0])
        l_w2 = (lanes_pose[2] - lanes_pose[1])
        l_w3 = (lanes_pose[4] - lanes_pose[3])
        l_w4 = (lanes_pose[5] - lanes_pose[4])

        if Y < lanes_pose[0]:
            return [np.max([(l_w1 - np.abs(Y - lanes_pose[0]))/l_w1, min_pertenence]) , 0.0, 0.0]
        elif Y < lanes_pose[1]:
            return [(l_w1 - np.abs(Y - lanes_pose[0]))/l_w1 , (l_w1 - np.abs(Y - lanes_pose[1]))/l_w1, 0.0]
        elif Y < lanes_pose[2]:
            return [0.0, (l_w2 - np.abs(Y - lanes_pose[1]))/l_w2 , (l_w2 - np.abs(Y - lanes_pose[2]))/l_w2]
        elif Y < (lanes_pose[2] + lanes_pose[3])/2:
            return [0.0, 0.0, np.max([(l_w2 - np.abs(Y - lanes_pose[2]))/l_w2, min_pertenence])]
            
        elif Y > lanes_pose[5]:
            return [np.max([(l_w4 - np.abs(Y - lanes_pose[5]))/l_w4, min_pertenence]) , 0.0, 0.0]
        elif Y > lanes_pose[4]:
            return [(l_w4 - np.abs(Y - lanes_pose[5]))/l_w4 , (l_w4 - np.abs(Y - lanes_pose[4]))/l_w4, 0.0]
        elif Y > lanes_pose[3]:
            return [0.0, (l_w3 - np.abs(Y - lanes_pose[4]))/l_w3 , (l_w3 - np.abs(Y - lanes_pose[3]))/l_w3]
        else:
            return [0.0, 0.0, np.max([(l_w3 - np.abs(Y - lanes_pose[3]))/l_w3, min_pertenence])]

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, reconstruction_loss=64, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss


    def train_step(self, data):
        data_0 = []
        if isinstance(data, tuple):
            data_0 = data[1]
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
           # if(data_0):
            reconstruction = self.decoder(z)
           # else:
                
            # reconstruction = self.decoder([z, data_0])
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.MAE(data, reconstruction)
            )
            reconstruction_loss_print = tf.reduce_mean(
                tf.keras.losses.MAE(data, reconstruction)
            )
            reconstruction_loss *= self.reconstruction_loss
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss_print,
            "kl_loss": kl_loss,
        }
    
    
    def test_step(self, data):
        if isinstance(data, tuple):
            data_0 = data[1]
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # reconstruction = self.decoder([z, data_0])
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.MSE(data, reconstruction)
            )
            reconstruction_loss_print = tf.reduce_mean(
                tf.keras.losses.MSE(data, reconstruction)
            )
            reconstruction_loss *= self.reconstruction_loss
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss_print,
            "kl_loss": kl_loss,
        }


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    
class HighDPredict():
    
    def __init__(self, mode='vel', encoder_model='encoder_model_1.h5', encoder_model_2='encoder_model_2.h5', 
                 lc_model='lc_model.h5', decoder_model='decoder_model.h5', norm_data='norm_data.csv'):
        self.mode = mode
        self.encoder_model = tf.keras.models.load_model('encoder_model_1.h5', custom_objects={'Sampling': Sampling})
        self.encoder_model_2 = tf.keras.models.load_model('encoder_model_2.h5', custom_objects={'Sampling': Sampling})
        self.lc_model = tf.keras.models.load_model('lc_model.h5')
        self.norm_data = pd.read_csv(norm_data, index_col=0)
        self.max_distance = 100.0
        self.T = 1.0/25.0
        self.vx0 = 0.0
        self.next_idx = 1
        self.latent_data = 0
        self.decoder_model = tf.keras.models.load_model('decoder_model.h5')
        
    def save_latent_data(self, name='latent'):

        np.save(name, self.latent_data)

    def set_init_x_vel(self, vel):

        self.vx0 = vel * self.norm_data.loc['xVelocity', 'std'] + self.norm_data.loc['xVelocity', 'mean']
    
    
    def lat_corr_predict(self, X_vel, X_surr):

        #print('*+*+*+*+*+*+*',np.expand_dims(X_surr, axis=(0, -1)).shape, np.expand_dims(X_vel, axis=(0, -1)).shape, '*+*+*+*+**+*')
        #print(np.expand_dims(X_surr, axis=(0, -1)))
        #print('+++++')

        xx_velacc_lat = self.encoder_model_2.predict(np.expand_dims(X_surr, axis=(0, -1)))[0]
        xx_sorr_lat = self.encoder_model.predict(np.expand_dims(X_vel, axis=(0, -1)))[0]
        xx_lat = np.concatenate((xx_velacc_lat, xx_sorr_lat), axis=1)



        # self.latent_data = np.append(self.latent_data, xx_lat, axis=0)

        lat_hat = self.lc_model.predict(xx_lat)
        # lat_hat = estimator.predict(xx_lat)


        yy = self.decoder_model.predict(lat_hat)
        b, a = sg.butter(2, 0.1)
        yy[0,:,0,0] = sg.filtfilt(b, a, yy[0,:,0,0])
        yy[0,:,1,0] = sg.filtfilt(b, a, yy[0,:,1,0])
        return yy
    
    def real_to_norm(self, vel, surr):
        
        # b, a = sg.butter(2, 0.1)
        # vel[:,0] = sg.filtfilt(b, a, vel[:,0])

        # b, a = sg.butter(2, 0.1)
        # vel[:,1] = sg.filtfilt(b, a, vel[:,1])     

        vel[:,0] = (vel[:,0] - self.norm_data.loc['xVelocity', 'mean'])/self.norm_data.loc['xVelocity', 'std']
        vel[:,1] = (vel[:,1])/self.norm_data.loc['yVelocity', 'std']  


        i = 0
        for h in ['f_d', 'b_d', 'bl_d', 'l_d', 'fl_d', 'br_d', 'r_d', 'fr_d']:
          
            if ((h != 'r_d') and (h != 'l_d')):

                surr[surr[:, i] == np.inf, i] = self.max_distance
                surr[surr[:, i] == 0, i] = self.max_distance
                surr[:, i] = self.max_distance - surr[:, i]
                surr[surr[:, i] < 0, i] = 0.0

            else:

                surr[:, i] = 0.0
            
            surr[:, i] = (surr[:, i])/self.norm_data.loc[h, 'std']

            i += 1
    
        return vel, surr
    
    
    def norm_to_real(self, y):
    
        return y
    
    
    def vel_acc_to_pose(self, y_hat, vel=0):
        
        x0 = np.array(0.0)
        y0 = np.array(0.0)
        # vx0 = (vel[-1, 0]*self.norm_data.loc['xVelocity', 'std'] + self.norm_data.loc['xVelocity', 'mean'])
        vx0 = float(self.vx0)
        # vy0 = vel[-1, 1]
        ax0 = 0.0
        x_arr = []
        y_arr = []
        
        # Check
        # x_arr.append(x0)
        # y_arr.append(y0)
        x_arr.append(0.0)
        y_arr.append(0.0)

        for i in range(32):
            y_arr.append(y_arr[-1] + y_hat[0, i, 1]*self.norm_data.loc['yVelocity', 'std']*self.T)

            #Velocity
            if self.mode == 'vel':
                x_arr.append(x_arr[-1] + y_hat[0, i, 0]*self.norm_data.loc['xVelocity', 'std']*self.T + 
                            self.norm_data.loc['xVelocity', 'mean']*self.T)

            # Acceleration
            elif self.mode == 'acc':
                x_arr.append(x_arr[-1] + (vx0)*self.T + 
                        0.5*y_hat[0, i, 0]*self.norm_data.loc['xAcceleration', 'std']*(self.T**2))

                vx0 = vx0 + y_hat[0, i, 0]*self.norm_data.loc['xAcceleration', 'std']*self.T

                if i == self.next_idx:
                    self.vx0 = float(vx0)
        
        if self.mode == 'vel':
            y_hat[0, :, 0] = y_hat[0, :, 0] * self.norm_data.loc['xVelocity', 'std'] + self.norm_data.loc['xVelocity', 'mean']
        
        elif self.mode == 'acc':
            y_hat[0, :, 0] = y_hat[0, :, 0] * self.norm_data.loc['xAcceleration', 'std']

        y_hat[0, :, 1] = y_hat[0, :, 1] * self.norm_data.loc['yVelocity', 'std']

        
        return x_arr, y_arr, y_hat
       
        
    def real_data_predict(self, vel, surr):

        vel, surr = self.real_to_norm(vel, surr)
        
        y = self.lat_corr_predict(vel, surr)
        
        y = self.norm_to_real(y)
        
        return self.vel_acc_to_pose(y, vel)