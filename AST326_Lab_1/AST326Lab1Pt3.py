import numpy as np
import matplotlib.pyplot as plt

large = np.loadtxt('Group_A_Large.txt',float) #Load the 'Large' data
small = np.loadtxt('Group_A_Small.txt',float) #Load the 'Small' data

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

#Function for weighted mean
def weightedmean(x,y): #Where x and y are lists of numbers.  List x is meant to be the list of measurements, while list y is meant to be the list of measurement errors. 
    a = [] #Create empty list to append values of the sum of the elements in list x times the elements in list y^-2.
    b = [] #Create empty list to append values of the sum of the elements in list y^-2 to.
    for measurement,error in zip(x,y): #Iterate through the elements in lists x and y, respectively.
        if error != 0: #Calculate numerator for values not equal to 0.  If don't distinguish between 0 values, will get an error because raising 0 to negative 2 is technically dividing by 0.
            num = measurement*((error)**(-2)) #In numerator, each element in list x times each element in list y^-2.
            a.append(num) #Append these values to the empty list "a" created above.
        else:
            a.append(error) #Append values of 0 as they are, since any operation on a 0 value won't change its value.
    numerator = np.sum(a) #Sum the values in list a.
    for error in y: 
        if error != 0: #Calculate denominator for values not equal to 0.  If don't distinguish between 0 values, will get an error because raising 0 to negative 2 is technically dividing by 0.
            denom = (error)**(-2) #In denominator, each element of y^-2.
            b.append(denom) #Append these values to the empty list "b" created above.
        else: 
            b.append(error) #Append values of 0 as they are, since any operation on a 0 value won't change its value.
    denominator = np.sum(b) #Sum the values in list b.
    wmean = numerator/denominator #Divide numerator by denominator, formula for weighted mean.
    return wmean #Return weighted mean.

#Function for uncertainty on weighted mean
def weightederror(y):
    c = [] #Create empty list to append values of the sum of the elements in list y to.
    for error in y: #Iterate through each of the elements in list y.
        if error != 0: #Calculate for values not equal to 0.  If don't distinguish between 0 values, will get an error because raising 0 to negative 2 is technically dividing by 0.
            denom = (error)**(-2) #In denominator, each element of y^-2.
            bottom = error**(-2) #In denominator, each element in list y^-2.
            c.append(bottom) #Append these values to the empty list "c" created above.
        else: 
            c.append(error) #Append values of 0 as they are, since any operation on a 0 value won't change its value.
    Sigma = np.sum(c) #Sum the values in list c.
    np.array([Sigma])
    uncertain = np.sqrt(1/Sigma) #Function for uncertainty on weighted mean.
    return uncertain #Return uncertainty on weighted mean.

mean_small = mean(small) #Calculate the mean of the 'Small' data
std_small = std(small) #Calculate the standard deviation of the 'Small' data
mean_large = mean(large) #Calculate the mean of the 'Large' data
std_large = std(large) #Calculate the standard deviation of the 'Large' data
print("The mean of the 'Small' data is", mean_small, "and the standard deviation is", std_small, ".")
print("The mean of the 'Large' data is", mean_large, "and the standard deviation is", std_large, ".")

std_small_squared = std_small**2
std_large_squared = std_large**2
print("The standard deviation of the 'Small' data squared is", std_small_squared, "and the standard deviation of the 'Large' data squared is", std_large_squared, ".")
print("Since the standard deviation squared of each data set is approximately equal to its mean, these results are consistent with what we expect from Poisson statistics.")

#Calculate weighted mean and uncertainty on weighted mean for each data set:
error_small = np.sqrt(small) #The error on each measurement is the square root of the measurement.
error_large = np.sqrt(large) #The error on each measurement is the square root of the measurement.
weighted_mean_small = weightedmean(small, error_small)
weighted_error_small = weightederror(error_small)
weighted_mean_large = weightedmean(large, error_large)
weighted_error_large = weightederror(error_large)
print("The weighted mean for the 'Small' data is", weighted_mean_small, "and the weighted error is", weighted_error_small,".")
print("The weighted mean for the 'Large' data is", weighted_mean_large, "and the weighted error is", weighted_error_large,".")

#Make the x-axis
measurements = np.arange(0,1000,1)

#Scatter plots
plt.scatter(measurements,small)
plt.xlabel("Number of Measurements")
plt.ylabel("Photon Count Rate")
plt.title("'Small' Photon Count Rate Scatterplot")
plt.savefig('small_photon_scatter.pdf')

plt.figure()
plt.scatter(measurements,large)
plt.xlabel("Number of Measurements")
plt.ylabel("Photon Count Rate")
plt.title("'Large' Photon Count Rate Scatterplot")
plt.savefig('large_photo_scatter.pdf')

#Find the minima and maxima of the data sets to find the right bin sizes and ranges
small_min = np.min(small)
small_max = np.max(small)
large_min = np.min(large)
large_max = np.max(large)

print("Small min is", small_min, ", small max is", small_max, ", large min is", large_min, ", large max is", large_max)

#Histograms
bin_count, bin_edges, boxes = plt.hist(small, bins = 9, range=(0,9))
plt.xlabel("Photon Count Rate")
plt.ylabel("Number of Measurements")
plt.title("'Small' Photon Count Rate Histogram")
plt.savefig('small_photon_hist.pdf')

plt.figure()
bin_count, bin_edges, boxes = plt.hist(large, bins = 26, range=(7,33))
plt.xlabel("Photon Count Rate")
plt.ylabel("Number of Measurements")
plt.title("'Large' Photon Count Rate Histogram")
plt.savefig('large_photon_hist.pdf')

import math as m
def Poisson(x): #Poisson function
    y = [] #Create an empty list to append to later
    for i in x: #Iterate through the values in the list that is called by the Poisson function
        P = (((mu**i)/m.factorial(i)))*(np.e**(-mu)) #Poisson formula
        y.append(P) #Append these values to the empty list y
    return y

#Small Poisson data
mu = 3.033
x_small = np.arange(0,9,1)
Poisson_small = Poisson(x_small)
array_small = np.asarray(Poisson_small) #Turn the list into an array
norm_small = array_small*1000 #Normalize the Poisson distribution by multiplying it by the number of measurements

#Large Poisson data
mu = 18.954
x_large = np.arange(7,33,1)
Poisson_large = Poisson(x_large)
array_large = np.asarray(Poisson_large) #Turn the list into an array
norm_large = array_large*1000 #Normalize the Poisson distribution by multiplying it by the number of measurements

#Histograms with Poisson distribution superimposed
#Small data
bin_count, bin_edges, boxes = plt.hist(small, bins = 9, range=(0,9))
plt.plot(x_small,norm_small,'r--')
plt.xlabel("Photon Count Rate")
plt.ylabel("Number of Measurements")
plt.title("'Small' Photon Count Rate with Poisson Distribution Superimposed")
plt.savefig('small_photon_poisson.pdf')

#Large data
plt.figure()
bin_count, bin_edges, boxes = plt.hist(large, bins = 26, range=(7,33))
plt.plot(x_large,norm_large,'r--')
plt.xlabel("Photon Count Rate")
plt.ylabel("Number of Measurements")
plt.title("'Large' Photon Count Rate with Poisson Distribution Superimposed")
plt.savefig('large_photon_poisson.pdf')

#Gaussian function
def Gaussian(x):
    z = []
    for i in x: 
        P = (1/(np.sqrt(2*np.pi)*sigma))*(np.e**((-(i - mu)**2)/(2*(sigma**2))))
        z.append(P)
    return z 

#Small Gaussian data
mu = 3.033
sigma = 1.7638089528126402
x_small = np.arange(0,9,1)
Gaussian_small = Gaussian(x_small)
array_small_Gaussian = np.asarray(Gaussian_small) #Turn the list into an array
norm_small_Gaussian = array_small_Gaussian*1000 #Normalize the Gaussian distribution by multiplying it by the number of measurements

#Large Gaussian data
mu = 18.954
sigma = 4.3851012660043684
x_large = np.arange(7,33,1)
Gaussian_large = Gaussian(x_large)
array_large_Gaussian = np.asarray(Gaussian_large) #Turn the list into an array
norm_large_Gaussian = array_large_Gaussian*1000 #Normalize the Gaussian distribution by multiplying it by the number of measurements

#Histograms with Gaussian distribution superimposed
bin_count, bin_edges, boxes = plt.hist(small, bins = 9, range=(0,9))
plt.plot(x_small,norm_small_Gaussian,'g--')
plt.xlabel("Photon Count Rate")
plt.ylabel("Number of Measurements")
plt.title("'Small' Photon Count Rate with Gaussian Distribution Superimposed")
plt.savefig('small_photon_gaussian.pdf')

plt.figure()
bin_count, bin_edges, boxes = plt.hist(large, bins = 26, range=(7,33))
plt.plot(x_large,norm_large_Gaussian,'g--')
plt.xlabel("Photon Count Rate")
plt.ylabel("Number of Measurements")
plt.title("'Large' Photon Count Rate with Gaussian Distribution Superimposed")
plt.savefig('large_photon_gaussian.pdf')

#Histograms with Gaussian and Poisson distributions superimposed
bin_count, bin_edges, boxes = plt.hist(small, bins = 9, range=(0,9))
plt.plot(x_small,norm_small,'r--',label='Poisson Distribution')
plt.plot(x_small,norm_small_Gaussian,'g--',label='Gaussian Distribution')
plt.xlabel("Photon Count Rate")
plt.ylabel("Number of Measurements")
plt.title("'Small' Photon Count Rate with Poisson and Gaussian Distributions")
plt.legend()
plt.savefig('small_photon_poisson_gaussian.pdf')

plt.figure()
bin_count, bin_edges, boxes = plt.hist(large, bins = 26, range=(7,33))
plt.plot(x_large,norm_large,'r--', label='Poisson Distribution')
plt.plot(x_large,norm_large_Gaussian,'g--', label='Gaussian Distribution')
plt.xlabel("Photon Count Rate")
plt.ylabel("Number of Measurements")
plt.title("'Large' Photon Count Rate with Poisson and Gaussian Distributions")
plt.legend()
plt.savefig('large_photon_poisson_gaussian.pdf')