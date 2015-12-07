# Default constants for training the SVR
SVR_EPSILON = .0625
N_SEGMENTS = 200
SQUARE_SIZE = 10
C = .125   # C = .25 is also pretty good

# For evaluating different values of C and epsilon
C_LIST = [.015625, .03125, .0625, .125, .25, .5]
EPSILON_LIST = [.015625, .03125, .0625, .125]

# Constants for running ICM on the MRF
ICM_ITERATIONS = 10
ITER_EPSILON = .01
COVAR = 0.25       # Covariance of predicted chrominance from SVR and actual covariance
WEIGHT_DIFF = 2    # Relative importance of neighboring superpixels
THRESHOLD = 25     # Threshold for comparing adjacent superpixels.
                   # Setting a higher threshold reduces error, but causes the image to appear more uniform.
