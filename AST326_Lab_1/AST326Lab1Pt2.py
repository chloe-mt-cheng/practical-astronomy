import numpy as np
import math as m
import matplotlib.pyplot as plt

photon_counts =(13, 17, 18, 14, 11, 8, 21, 18, 9, 12, 9, 17, 14, 6, 10, 16, 16, 11, 10, 12, 8, 20, 14, 10, 14, 17, 13, 16, 12, 10) #Load photon count data from textfile

#Mean function:
def mean(x): #Where x is any list of numbers.  List x is meant to be the list of distance measurements.
	avg = np.sum(x)/len(x) #Mean formula.
	return avg #Return mean.

#Standard deviation function:
def std(x): #Where x is any list of numbers.  List x is meant to be the list of distance measurements.
	avg = mean(x) #Calculate the mean of all of the elements in x.
	stdvals = [] #Create an empty list to add to later.
	for measurement in x: #Iterate through the elements in list x.
		measurement = (measurement - avg)**2 #Each element in list x minus mean of all elements in list x, squared.  These will be the new values for the elements in list x.
		stdvals.append(measurement) #Add the new values for each element in list x to the list stdvals.
	stdev = np.sqrt((np.sum(stdvals))/((len(stdvals)) - 1)) #Standard deviation formula.
	return stdev #Return standard deviation.
	
bar_gamma = mean(photon_counts) #Calculate the mean of the photon measurements.
sigma_gamma = std(photon_counts) #Calculate the standard deviation of the photon measurements.
print("The average photon count per second is ", bar_gamma, "+/-", sigma_gamma, "counts/second.")

#Poisson function:
def Poisson(x): 
    y = [] #Create an empty list to append to later
    for i in x: #Iterate through the values in the list that is called by the Poisson function
        P = (((mu**i)/m.factorial(i)))*(np.e**(-mu)) #Poisson formula
        y.append(P) #Append these values to the empty list y
    return y

mu = 12
x = np.arange(5,25,1)
Poisson_gamma = Poisson(x)
array_gamma = np.asarray(Poisson_gamma) #Turn the list into an array
norm_gamma = array_gamma*30 #Normalize the Poisson distribution by multiplying it by the number of measurements

#To get the same bin widths as the histogram:
binned_gamma = np.zeros(19) 
for i in range(19):
    norm_gamma[i] = (norm_gamma[i] + norm_gamma[i + 1]) #Put values of the Poisson function into bins with width 2

#Histogram with Poisson function superimposed:
bin_count, bin_edges, boxes = plt.hist(photon_counts, bins = 8, range = (5,21)) #Plot the histogram
plt.plot(x,norm_gamma,'r--') #Plot the Poisson function on top
plt.xlabel("Number of Photons")
plt.ylabel("Number of Measurements")
plt.title('Histogram of Photon Count Rates with Poisson Function Superimposed')
plt.savefig('pt2_plot.pdf')