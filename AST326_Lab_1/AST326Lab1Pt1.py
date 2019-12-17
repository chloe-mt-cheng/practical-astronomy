import numpy as np
import matplotlib.pyplot as plt

distance_measurements = (38.91, 37.14, 38.19, 41.03, 34.86, 37.33, 35.16, 37.96, 36.93, 40.41, 29.50, 37.33, 41.84, 37.53, 34.12, 34.11, 37.94, 34.43, 36.68, 41.31, 39.61, 35.48, 34.98, 39.05, 39.62, 37.96, 39.02, 37.47, 33.76, 36.51) #List of distance measurements data.
measurement_errors = (1.41, 0.36, 0.69, 3.53, 2.64, 0.17, 2.34, 0.46, 0.57, 2.91, 8.00, 0.17, 4.34, 0.03, 3.38, 3.39, 0.44, 3.07, 0.82, 3.81, 2.11, 2.02, 2.52, 1.55, 2.12, 0.46, 1.52, 0.03, 3.74, 0.99) #Make file for measurement errors.  List of measurement errors data.

#Mean function:
def mean(x): #Where x is any list of numbers.  List x is meant to be the list of distance measurements.
	avg = np.sum(x)/len(x) #Mean formula.
	return avg #Return mean.

barD = mean(distance_measurements) #Calculate the mean of the data from the distance_measurements file.

#Standard deviation function:
def std(x): #Where x is any list of numbers.  List x is meant to be the list of distance measurements.
	avg = mean(x) #Calculate the mean of all of the elements in x.
	stdvals = [] #Create an empty list to add to later.
	for measurement in x: #Iterate through the elements in list x.
		measurement = (measurement - avg)**2 #Each element in list x minus mean of all elements in list x, squared.  These will be the new values for the elements in list x.
		stdvals.append(measurement) #Add the new values for each element in list x to the list stdvals.
	stdev = np.sqrt((np.sum(stdvals))/((len(stdvals)) - 1)) #Standard deviation formula.
	return stdev #Return standard deviation.

sigmaD = std(distance_measurements) #Calculate the standard deviation of the data from the distance_measurements file.

print("The distance to the star is ", barD, "+/-", sigmaD, "pc.")

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

wbarD = weightedmean(distance_measurements,measurement_errors) #Calculate the weighted mean of the data from the distance_measurements file and the associated measurement_errors file.

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

wsigmaD = weightederror(measurement_errors) #Calculate the uncertainty on the weighted mean using the data from the measurement_errors file.

print("Using weighted mean statistics, the distance to the star is ", wbarD, "+/-", wsigmaD, "pc.")

#Scatterplot
distance = np.arange(20,50,1) #Create a scale for the y-axis.

plt.scatter(distance_measurements, distance) #Plot a scatterplot of the distance measurements on the x-axis and the distance on the y-axis. 
plt.xlabel("Measurement")
plt.ylabel("Distance (pc)")
plt.title('Scatterplot of the Number of Measurements of Distance to a Star')
plt.savefig('pt1scatter.pdf')

#Histogram
plt.figure()
bin_count, bin_edges, boxes = plt.hist(distance_measurements, bins = 5) #Plot a histogram of the distance measurements.
plt.xlabel("Distance (pc)")
plt.ylabel("Number of Measurements")
plt.title('Histogram of the Number of Measurements of Distance to a Star')
plt.savefig('pt1histogram.pdf')