# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 22:24:31 2020

@author: Adam Linek
"""

import mdshare #package for downloading data
import numpy as np #package for math operations
import matplotlib.pyplot as plt #package for plotting data
import argparse #package for working in the command-line mode

from sklearn.manifold import TSNE #package for reducing dimensionality of the data

def loadData(): #function which downloads data from FTP server or loads it from file if already downloaded and then returns it as ndarray
    dataset = mdshare.fetch('alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
    with np.load(dataset) as f:
        X = np.vstack([f[key] for key in sorted(f.keys())])
    return X
    

def fit(dataToFit): #function which reduces dimensionality of the given ndarray
    print("Computing in progres...") #prints the message that reducing dimensionalty has been started (may take some time)
    Y = TSNE(n_components=2, perplexity=parsedPerplexity, early_exaggeration=parsedEarly_exaggeration, learning_rate=parsedLearning_rate, n_iter=parsedN_iter, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=parsedVerbose, random_state=None, method='barnes_hut', angle=parsedAngle, n_jobs=None).fit_transform(dataToFit[::parsedStep])                                                                                                                                                                                                                                                                                                                                                 
    #Reducing dimensionality is processed by t-SNE tool which takes several arguments. More: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    Y[:, 0] = np.interp(Y[:, 0], (Y[:, 0].min(), Y[:, 0].max()), (-np.pi, np.pi)) #scales 1st dim of the reduced 2-dim array to fit -π:π range (for better visualization purposes)
    Y[:, 1] = np.interp(Y[:, 1], (Y[:, 1].min(), Y[:, 1].max()), (-np.pi, np.pi)) #scales 2nd dim of the reduced 2-dim array to fit -π:π range (for better visualization purposes)
    print("Done.") #prints the message that reducing dimensionality is done
    return Y #Returns 2-dim array

def visualize(embeddingToVisualize): #function which generates up to two plots (scattered and contoured) of given 2-dim array
    print("Generating scatter visualization...") #prints the message that scatter plot is being generated
    plt.scatter(embeddingToVisualize[:, 0], embeddingToVisualize[:, 1] , s = 25, alpha = 0.5) #generates scatter plot for 1st dim of given array as xaxis and 2nd dim of the given array as yaxis with dots of size 25 and 0.5 for transparency
    plt.axis('scaled') #makes the plot to be squared
    plt.xlim(-4, 4) #sets the xaxis range from -4 to 4
    plt.ylim(-4,4) #sets the yaxis range from -4 to 4
    plt.xticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) #shows only -π, 0 and π on the xaxis label
    plt.yticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) #shows only -π, 0 and π on the yaxis label
    plt.show() #shows the plot
    print("Done.") #prints the message that generating scatter plot is done
    
    if contourEnable == True: #checks if user wants to generate contour plot with density estimation (make take some time)
        print("Generating contour visualization...") #prints the message that contour plot is being generated
        gridSize = 4 
        gridSpacing = 0.1
        meshx = np.arange(-gridSize, gridSize + gridSpacing, gridSpacing) 
        meshy = np.arange(-gridSize, gridSize + gridSpacing, gridSpacing)
        #to estimate density of the points, the grid array is generated; (-gridSize, gridSize) range for xaxis and (-gridSize, gridSize) range for yaxis
        Xmesh, Ymesh = np.meshgrid(meshx, meshy) #converts two 1-dim arrays to 2-dim arrays for plotting and calculations purposes
        Z = np.zeros([meshx.size, meshy.size]) #creates 2-dim array which corresponds to grid array; the value for any Z(x, y) determines how many data points is near that grid point(x,y)
        DATA = np.array([embeddingToVisualize[:, 0], embeddingToVisualize[:, 1]]) #creates temporary array for data points
        progress = 0 #variable to calculate current progress of density calculations (that takes some time and the user might be impatient without any information about the current progress of calculations)
        for i in meshx: #for every point of xaxis of the grid
            progress += 100/meshx.size #calculates current progress
            print(int(progress), "%") #prints the message with current progress
            for j in meshy: #for every point of yaxis of the grid
                for l in np.transpose(DATA): #for every (x, y) point of data
                    if np.sqrt(np.power(i-l[0], 2) + np.power(j-l[1], 2)) < gridSpacing: #check if the distance between given data point and given grid point is less than grid space
                        Z[int((j+gridSize)/gridSpacing), int((i+gridSize)/gridSpacing)] += 1 #if the point is close enough, add increase the value of grid point (x,y) by 1 
        Z = np.interp(Z, (Z.min(), Z.max()), (0, 1)) #scales the Z array to 0:1 range             
        plt.contourf(Xmesh, Ymesh, Z, cmap = 'plasma') #generate filled contour plot
        plt.axis('scaled') #makes the plot to be squared
        plt.xlim(-4, 4) #sets the xaxis range from -4 to 4
        plt.ylim(-4,4) #sets the yaxis range from -4 to 4
        plt.xticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) #shows only -π, 0 and π on the xaxis label
        plt.yticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) #shows only -π, 0 and π on the yaxis label
        cbar = plt.colorbar() #shows colorbar on the right side
        cbar.set_ticks([0, Z.max()]) ##shows only 0 and 1 on the colorbar label
        plt.show() #shows the plot
        print("Done.") #prints the message that generating contour plot is done
    
#Processing iportant flags for command-line mode
parser = argparse.ArgumentParser(description = "Adam Linek: Projection of high-dimensional tensor onto a two-dimensional space.")
parser.add_argument('-s','--step', metavar = '', type = int, default = 500, help = "Set the step of the samples. Higher number means more samples will be skiped (500 by default).")
#sets the step of data which are taken into account during reducing dimensionality
parser.add_argument('-p','--perplexity', metavar = '', type = float, default = 30., help = "The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Different values can result in significantly different results (30. by default).")
#sets the perplexity
parser.add_argument('-e','--early_exaggeration', metavar = '', type = float, default = 12., help = "Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. For larger values, the space between natural clusters will be larger in the embedded space. Again, the choice of this parameter is not very critical. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high (12. by default).")
#sets easrly exaggeration
parser.add_argument('-l','--learning_rate', metavar = '', type = float, default = 200., help = "The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate is too low, most points may look compressed in a dense cloud with few outliers. If the cost function gets stuck in a bad local minimum increasing the learning rate may help (200. by default).")
#sets learning rate
parser.add_argument('-n','--n_iter', metavar = '', type = int, default = 1000, help = "Maximum number of iterations for the optimization. Should be at least 250 (1000 by default).")
#sets number of iterations
parser.add_argument('-v','--verbose', metavar = '', type = int, default = 0, help = "Verbosity level (0 by default).")
#sets verbosity level
parser.add_argument('-a','--angle', metavar = '', type = float, default = 0.5, help = "The trade-off between speed and accuracy for Barnes-Hut T-SNE. ‘angle’ is the angular size of a distant node as measured from a point. If this size is below ‘angle’ then it is used as a summary node of all points contained within it. This method is not very sensitive to changes in this parameter in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing computation time and angle greater 0.8 has quickly increasing error (0.5 by default).")
#sets angle
parser.add_argument('-c','--contour', metavar = '', type = bool, default = False, help = "If set to True it will generate contour plot in addition to scatter plot. Warning: May take some time if step is small (False by default).")
#sets if contour plot is supposed to be generated or not

args = parser.parse_args() #parses command and assigns values into variables
parsedStep = args.step
parsedPerplexity = args.perplexity
parsedEarly_exaggeration = args.early_exaggeration
parsedLearning_rate = args.learning_rate
parsedN_iter = args.n_iter
parsedVerbose = args.verbose
parsedAngle = args.angle
contourEnable = args.contour

#main part of the script
visualize(fit(loadData())) #executing visualize() function onto the 2-dim array after being reduced by fit() function from high-dimensional tensor loaded by loadData() function

