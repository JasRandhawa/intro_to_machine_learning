import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pylab import plt, mpl
plt.style.use('seaborn')
import os

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

infile = open(data_path("mass16.dat"),'r')

mpl.rcParams['font.family'] = 'serif'

###################plots#########################
def MakePlot(x,y, styles, labels, axlabels):
    plt.figure(figsize=(10,6))
    for i in range(len(x)):
        plt.plot(x[i], y[i], styles[i], label = labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
    plt.legend(loc=0)
#####################################################

#################read data ############################
Masses = pd.read_fwf(infile, usecols=(2,3,4,6,11),names=('N', 'Z', 'A', 'Element', 'Ebinding'),widths=(1,3,5,5,5,1,3,4,1,13,11,11,9,1,2,11,9,1,3,1,12,11,1),header=39,index_col=False)
Masses['Ebinding'] = pd.to_numeric(Masses['Ebinding'], errors='coerce')
Masses=Masses.dropna()
Masses['Ebinding']/=1000
Masses= Masses.groupby('A')
#ele=Masses.get_group(2)
Masses = Masses.apply(lambda t: t[t.Ebinding==t.Ebinding.max()])
A= Masses['A']
Z= Masses['Z']
N= Masses['N']
Element=Masses['Element']
Energy=(Masses['Ebinding'])
print (Z, Element, Energy)
MakePlot([Z],[Energy],['b-'],['NvsZ'],['$N$','$Z$'])
#plt.show()

###################################################################
####setting up design matrix (very important)######################
###################################################################

#It is based on liquid drop model
### BE=  a1*A - a2*A^2/3 - a3*(Z^2)/(A^1/3)- a4*((N-Z)^2)/A
###################################################################
X= np.zeros((len(A),5))
X[:,0]=1
X[:,1]=A
X[:,2]=A**(2.0/3.0)
X[:,3]=A**(-1.0/3.0)
X[:,4]=A**(-1.0)

############# Using Linear Regression #######################
linreg = skl.LinearRegression()

clf = linreg.fit(X,Energy)
yy = clf.predict(X)
Masses['Eapprox']  = yy
# Generate a plot comparing the experimental with the fitted values values.
fig, ax = plt.subplots()
ax.set_xlabel(r'$A = N + Z$')
ax.set_ylabel(r'$E_\mathrm{bind}\,/\mathrm{MeV}$')
ax.plot(Masses['A'], Masses['Ebinding'], alpha=0.7, lw=2,
        label='Ame2016')
ax.plot(Masses['A'], Masses['Eapprox'], alpha=0.7, lw=2, c='m',
        label='Fit')
ax.legend()
save_fig("Masses2016")
plt.show()
