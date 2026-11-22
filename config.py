
"""
Central configuration for the Rocket Thermal Model.

Edit these values to change the baseline simulation.

All quantities use SI units unless otherwise specified.
"""

# --------------------------------------------------
# Coolant Properties
# --------------------------------------------------

T_C_IN = 300.0          # Coolant inlet temperature [K]
M_DOT_C = 2.0           # Coolant mass flow rate [kg/s]
C_P_C = 3000.0          # Coolant specific heat [J/(kg K)]

# --------------------------------------------------
# Heat Transfer Parameters
# --------------------------------------------------

H_C = 15000.0           # Coolant-side HTC [W/(m^2 K)]
L_W = 0.003             # Wall thickness [m]
K_W = 350.0             # Wall thermal conductivity [W/(m K)]
EPSILON_EFF = 0.20      # Effective radiation emissivity [-]

# --------------------------------------------------
# Illustrative Design Constraints
# --------------------------------------------------

T_WALL_ALLOW = 1000.0       # Maximum wall temperature [K]
T_COOLANT_ALLOW = 600.0     # Maximum coolant temperature [K]

# --------------------------------------------------
# Simplified Heat Transfer Geometry
# --------------------------------------------------

# Used by the original constant-perimeter model.
# The geometry-based model calculates wall areas
# directly from the nozzle geometry.

P_H = 0.10              # Effective heated perimeter [m]


# --------------------------------------------------
# Cooling Channel Geometry
# --------------------------------------------------

N_CHANNELS = 50         # Number of coolant channels
CHANNEL_WIDTH = 0.005  # Channel width [m]
CHANNEL_HEIGHT = 0.002 # Channel height [m]


# --------------------------------------------------
# Coolant Hydraulic Properties
# --------------------------------------------------

P_IN = 5.0e6            # Coolant inlet pressure [Pa]
RHO_C = 1000.0          # Coolant density [kg/m^3]
F_D = 0.020             # Darcy friction factor [-]
