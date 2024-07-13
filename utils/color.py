import numpy as np
import matplotlib
import matplotlib.cm as cm 

rwb_data = ['#ff2414', '#ff4704', '#ff5f00', '#ff7400', '#ff8700', '#ff9900', '#ffaa00', '#ffba00', '#ffca17', '#ffd044', '#ffd763', '#ffdd7e', '#ffe498', '#ffeab2', '#fff1cc', '#fff8e5', '#ffffff', '#ecfff4', '#d8ffe9', '#c4ffdd', '#aeffd2', '#96ffc7', '#7affbc', '#58ffb1', '#14ffa6', '#00f0c0', '#00dfe0', '#00ccff', '#00b8ff', '#00a1ff', '#0086ff', '#0062ff', '#1424ff']

rwb_thick_data = ['#ff2414', '#ff4704', '#ff5f00', '#ff7400', '#ff8700', '#ff9900', '#ffaa00', '#ffba00', '#ffca17', '#ffd044', '#ffd763', '#ffdd7e', '#ffe498', '#ffeab2', '#fff1cc', '#fff8e5', '#ffffff', '#ffffff', '#ffffff', '#ecfff4', '#d8ffe9', '#c4ffdd', '#aeffd2', '#96ffc7', '#7affbc', '#58ffb1', '#14ffa6', '#00f0c0', '#00dfe0', '#00ccff', '#00b8ff', '#00a1ff', '#0086ff', '#0062ff', '#1424ff']


rwb_map = matplotlib.colors.LinearSegmentedColormap.from_list('rwb', rwb_data)
rwb_thick_map = matplotlib.colors.LinearSegmentedColormap.from_list('rwb_thick', rwb_thick_data)
