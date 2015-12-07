# Default constants for training the SVR
SVR_EPSILON = .0625
N_SEGMENTS = 200
SQUARE_SIZE = 10
C = .125   # C = .25 is also pretty good

# For evaluating different values of C and epsilon
C_LIST = [.015625, .03125, .0625, .125, .25, .5]
EPSILON_LIST = [.015625, .03125, .0625, .125]

WEIGHT_DIFF_LIST = []    #[1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
THRESHOLD_LIST = [0, 25] #[20, 25, 30, 35, 40]

# Constants for running ICM on the MRF
ICM_ITERATIONS = 10
ITER_EPSILON = .01
COVAR = 0.25       # Covariance of predicted chrominance from SVR and actual covariance
WEIGHT_DIFF = 2    # Relative importance of neighboring superpixels
THRESHOLD = 25     # Threshold for comparing adjacent superpixels.
                   # Setting a higher threshold reduces error, but causes the image to appear more uniform.
