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
COVAR = 1          # Covariance of observed chrominance
WEIGHT_DIFF = 0.5  # Relative importance of neighboring superpixels
MAX_DIFF = 0.2     # Maximum contribution to potential
THRESHOLD = 25     # Threshold for comparing adjacent superpixels. 100 is too high to change anything.
