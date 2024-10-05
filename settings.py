# -----------------------------------------------------------------------------
# Author: Ludwig Horvath
# Email: ludhor@kth.se
# Date: 2024-10-05
# -----------------------------------------------------------------------------


import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as lines

saved_features = {
    'figures': {
        'topview animation': {
            'figsize': (10, 10)
        }
    },
    'rectangles': {
        'ground': patches.Rectangle(**{
            'xy': (-1, -1),
            'width': 21,
            'height': 27,
            'fill': True,
            'edgecolor': 'white',
            'facecolor': 'white',
        }),
        'farm1': patches.Rectangle(**{
            'xy': (0, 1),
            'width': 5,
            'height': 16,
            'fill': True,
            'edgecolor': 'blue',
            'facecolor': 'blue',
            'alpha': 0.1
        }),
        'farm2': patches.Rectangle(**{
            'xy': (7, 1),
            'width': 5,
            'height': 16,
            'fill': True,
            'edgecolor': 'blue',
            'facecolor': 'blue',
            'alpha': 0.1
        }),
        'farm3': patches.Rectangle(**{
            'xy': (14, 1),
            'width': 5,
            'height': 16,
            'fill': True,
            'edgecolor': 'blue',
            'facecolor': 'blue',
            'alpha': 0.1
        })
    },
    'paths': {
        'charging_to_farm_1': {
            'M': 0,
            'Line': lines.Line2D(**{
                'xdata': [16.5, 0.5],
                'ydata': [23, 17],
                'color': 'lightgreen',
                'zorder': 2
            })
        },
        'farm_1': {
            'M': 1,
            'Line': lines.Line2D(**{
                'xdata': [0.5, 0.5, 1.5, 1.5, 3.5, 3.5, 4.5, 4.5],
                'ydata': [17, 0, 0, 18, 18, 0, 0, 17],
                'color': 'red',
                'zorder': 2
            })
        },
        'farm_transition_1': {
            'M': 0,
            'Line': lines.Line2D(**{
                'xdata': [4.5, 4.5, 7.5, 7.5],
                'ydata': [17, 18, 18, 17],
                'color': 'lightgreen',
                'zorder': 2
            })
        },
        'farm_2': {
            'M': 1,
            'Line': lines.Line2D(**{
                'xdata': [7 + 0.5, 7 + 0.5, 7 + 1.5, 7 + 1.5, 7 + 3.5, 7 + 3.5, 7 + 4.5, 7 + 4.5],
                'ydata': [17, 0, 0, 18, 18, 0, 0, 17],
                'color': 'red',
                'zorder': 2
            })
        },
        'farm_transition_2': {
            'M': 0,
            'Line': lines.Line2D(**{
                'xdata': [7 + 4.5, 7 + 4.5, 7 + 7.5, 7 + 7.5],
                'ydata': [17, 18, 18, 17],
                'color': 'lightgreen',
                'zorder': 2
            })
        },
        'farm_3': {
            'M': 1,
            'Line': lines.Line2D(**{
                'xdata': [2 * 7 + 0.5, 2 * 7 + 0.5, 2 * 7 + 1.5, 2 * 7 + 1.5, 2 * 7 + 3.5, 2 * 7 + 3.5, 2 * 7 + 4.5, 2 * 7 + 4.5],
                'ydata': [17, 0, 0, 18, 18, 0, 0, 17],
                'color': 'red',
                'zorder': 2
            })
        },
        'farm_3_to_charging': {
            'M': 0,
            'Line': lines.Line2D(**{
                'xdata': [18.501, 16.5],
                'ydata': [17, 23],
                'color': 'lightgreen',
                'zorder': 2
            })
        }
    },
    'environment': {
        'default': {
            'nr_workers': 2,
            'nr_visitors': 0
        }
    },
    'agents': {
        'robot': {
            'width': 1,
            'height': 1,
            'fill': True,
            'edgecolor': 'black',
            'facecolor': 'white',
            'alpha': 1,
            'zorder': 5,
        },
        'human': {
            'worker': {
                'P': [[0, 0.1, 0.1, 0.1, 0.7],
                      [0.003, 0.97, 0.006, 0, 0.021],
                      [0.0015, 0.0225, 0.97, 0.003, 0.003],
                      [0.0015, 0, 0.024, 0.97, 0.0045],
                      [0.006, 0.003, 0.003, 0.018, 0.97]],
                'p_react_to_alarm': 1,
                'color': 'blue'
            },
            'visitor': {
                'P': [[0.8, 0.07, 0.05, 0.04, 0.04],
                      [0.005, 0.95, 0.025, 0, 0.02],
                      [0.0075, 0, 0.95, 0.02, 0.0225],
                      [0.01, 0, 0.005, 0.95, 0.035],
                      [0.035, 0.014, 0.007, 0.014, 0.93]],
                'p_react_to_alarm': 0.8,
                'color': 'red'
            }
        }
    },
    'sensors': {
            'rgbd-cnn': {
                'P_u': 0,
                'P_f': 0.0007,
                'std': 0.02
            },
            'rgbd-lidar': {
                'P_u': 0,
                'P_f': 0.0006,
                'std': 0.01
            },
    },
    'informative': {
        'safety zones': {
            'red zone': {
                'radius': 3,
                'fill': True,
                'edgecolor': 'black',
                'facecolor': 'red',
                'alpha': 0.3,
                'zorder': 4
            },
            'yellow zone': {
                'radius': 7,
                'fill': True,
                'edgecolor': 'black',
                'facecolor': 'yellow',
                'alpha': 0.3,
                'zorder': 4
            },
            'green zone': {
                'radius': 10,
                'fill': True,
                'edgecolor': 'black',
                'facecolor': 'green',
                'alpha': 0.3,
                'zorder': 4
            }
        },
        'sensor': {
            'sensor circle': {
                'fill': False,
                'edgecolor': 'black',
                'alpha': 0.4,
                'zorder': 4
            }
        },
    },
    'checkpoints': {
        'farm1': {
            'x': 2.5,
            'y': 17,
            'color': 'orange',
        },
        'farm2': {
            'x': 9.5,
            'y': 17,
            'color': 'orange',
            'zorder': 3
        },
        'farm3': {
            'x': 16.5,
            'y': 17,
            'color': 'orange',
            'zorder': 3
        },
        'chargingstation': {
            'x': 16.5,
            'y': 23,
            'color': 'orange',
            'zorder': 3
        },
        'entrance/exit': {
            'x': 2.5,
            'y': 25,
            'color': 'orange',
            'zorder': 3
        }
    },
    'sensitivity_analysis': {
        'default': {
            'environment': {
                'nr_workers': 2,
                'nr_visitors': 0
            },
            'rgbd-cnn': {
                'P_u': 0,
                'P_f': 0.0007,
                'std': 0.02
            },
            'rgbd-lidar': {
                'P_u': 0,
                'P_f': 0.0006,
                'std': 0.01
            }
        },
        'ranges': {
            'environment': {
                'nr_workers': np.arange(3, 4).tolist(),  
                'nr_visitors': np.arange(3, 4).tolist()  
            },
            'sensor': {
                # Using linspace for ranges
                'P_u': np.linspace(0, 1, num=1).tolist(),  # Generates values from 0.1 to 0.5
                'P_f': np.linspace(0.0006, 0.0008, num=10).tolist(),  # Generates values from 0.1 to 0.2
                'std': np.linspace(0.005, 0.015, num=10).tolist()  # Generates values from 0.001 to 0.05
            }
        }
    },
    'plotstyle_settings': {
        'font': {
            'family': 'sans-serif',
            'name': 'Arial',
            'title_size': 16,
            'label_size': 12
        } 
    }
}
