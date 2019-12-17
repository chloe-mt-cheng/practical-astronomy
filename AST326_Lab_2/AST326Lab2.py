import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#Load data
test_room_temp_data = np.fromfile('roomtempone27_3.dat',dtype='int16')-2.**11 #Initial room temperature sample
room_temp_data = np.fromfile('rtemp_22_3_c.dat',dtype='int16')-2.**11
ice_data = np.fromfile('cwater_0_3_b.dat',dtype='int16')-2.**11
boil_water1_data = np.fromfile('hwater_83_0_f.dat',dtype='int16')-2.**11 #Boiling water at 83 degrees
boil_water2_data = np.fromfile('hwater_61_4_b.dat',dtype='int16')-2.**11 #Boiling water at 61.4 degrees
boil_water3_data = np.fromfile('hotwater_a_49_3.dat',dtype='int16')-2.**11 #Boiling water at 49.3 degrees
dry_ice_data = np.fromfile('dice_d.dat',dtype='int16')-2.**11
ln2_data = np.fromfile('liquidNNf_d.dat',dtype='int16')-2.**11
LTE_data = np.fromfile('LTEsignal_room_temp.dat',dtype='int16')-2.**11
FM_data = np.fromfile('FMradio_room_temp.dat',dtype='int16')-2.**11

#Sections 4.1/4.4
#Calculating statistics for the test room temperature data
test_room_temp_mean = np.mean(test_room_temp_data) #Calculate mean
test_room_temp_median = np.median(test_room_temp_data) #Calculate median
test_room_temp_std = np.std(test_room_temp_data) #Calculate standard deviation
test_room_temp_var = np.var(test_room_temp_data) #Calculate variance
print('The mean of the test room temperature data is', test_room_temp_mean, "the median is", test_room_temp_median, "the variance is", test_room_temp_var, "and the standard deviation is", test_room_temp_std,".")

#Calculating statistics for the room temperature data
room_temp_mean = np.mean(room_temp_data) #Calculate mean
room_temp_median = np.median(room_temp_data) #Calculate median
room_temp_std = np.std(room_temp_data) #Calculate standard deviation
room_temp_var = np.var(room_temp_data) #Calculate variance
print('The mean of the room temperature data is', room_temp_mean, "the median is", room_temp_median, "the variance is", room_temp_var, "and the standard deviation is", room_temp_std,".")

#Calculating statistics for the ice data
ice_mean = np.mean(ice_data) #Calculate mean
ice_median = np.median(ice_data) #Calculate median
ice_std = np.std(ice_data) #Calculate standard deviation
ice_var = np.var(ice_data) #Calculate variance
print('The mean of the ice data is', ice_mean, "the median is", ice_median, "the variance is", ice_var, "and the standard deviation is", ice_std,".")

#Calculating statistics for all of the boiling water data
#Boiling water at 83 degrees
boil1_mean = np.mean(boil_water1_data) #Calculate mean
boil1_median = np.median(boil_water1_data) #Calculate median
boil1_std = np.std(boil_water1_data) #Calculate standard deviation
boil1_var = np.var(boil_water1_data) #Calculate variance 
print('The mean of the boiling water data at 83.0 degrees is', boil1_mean, "the median is", boil1_median, "the variance is", boil1_var, "and the standard deviation is", boil1_std,".")

#Boiling water at 61.4 degrees
boil2_mean = np.mean(boil_water2_data) #Calculate mean
boil2_median = np.median(boil_water2_data) #Calculate median
boil2_std = np.std(boil_water2_data) #Calculate standard deviation
boil2_var = np.var(boil_water2_data) #Calculate variance
print('The mean of the boiling water data at 61.4 degrees is', boil2_mean, "the median is", boil2_median, "the variance is", boil2_var, "and the standard deviation is", boil2_std,".")

#Boiling water at 49.3 degrees
boil3_mean = np.mean(boil_water3_data) #Calculate mean
boil3_median = np.median(boil_water3_data) #Calculate median
boil3_std = np.std(boil_water3_data) #Calculate standard deviation
boil3_var = np.var(boil_water3_data) #Calculate variance
print('The mean of the boiling water data at 49.3 degrees is', boil3_mean, "the median is", boil3_median, "the variance is", boil3_var, "and the standard deviation is", boil3_std,".")

#Calculating statistics for the CO2 data
co2_mean = np.mean(dry_ice_data) #Calculate mean
co2_median = np.median(dry_ice_data) #Calculate median
co2_std = np.std(dry_ice_data) #Calculate standard deviation
co2_var = np.var(dry_ice_data) #Calculate variance
print('The mean of the dry ice data is', co2_mean, "the is", co2_median, "the variance is", co2_var, "and the standard deviation is", co2_std,".")

#Calculating statistics for the LN2 data
ln2_mean = np.mean(ln2_data) #Calculate mean
ln2_median = np.median(ln2_data) #Calculate median
ln2_std = np.std(ln2_data) #Calculate standard deviation
ln2_var = np.var(ln2_data) #Calculate variance
print('The mean of the liquid nitrogen data is', ln2_mean, "the median of the liquid nitrogen data is", ln2_median, "the variance of the liquid nitrogen data is", ln2_var, "and the standard deviation of the liquid nitrogen data is", ln2_std,".")

#Calculating statistics for the LTE data
LTE_mean = np.mean(LTE_data) #Calculate mean
LTE_median = np.median(LTE_data) #Calculate median
LTE_std = np.std(LTE_data) #Calculate standard deviation
LTE_var = np.var(LTE_data) #Calculate variance 
print('The mean of the LTE data is', LTE_mean, "the median of the LTE data is", LTE_median, "the variance of the LTE data is", LTE_var, "and the standard deviation of the LTE data is", LTE_std,".")

#Calculating statistics for the FM data
FM_mean = np.mean(FM_data) #Calculate mean
FM_median = np.median(FM_data) #Calculate median
FM_std = np.std(FM_data) #Calculate standard deviation
FM_var = np.var(FM_data) #Calculate variance
print('The mean of the FM data is', FM_mean, "the median of the FM data is", FM_median, "the variance of the FM data is", FM_var, "and the standard deviation of the FM data is", FM_std,".")

#Select small sections of data for preliminary scatter plots 
test_room_temp_sample = test_room_temp_data[300:1500]
room_temp_sample = room_temp_data[300:1500]
ice_data_sample = ice_data[3000:4500]
boil1_sample = boil_water1_data[2000:3500] #83 degrees
boil2_sample = boil_water2_data[2000:3500] #61.4 degrees
boil3_sample = boil_water3_data[2000:3500] #49.3 degrees
co2_sample = dry_ice_data[3000:4500]
ln2_sample = ln2_data[3000:4500]
LTE_sample = LTE_data[3000:4500]
FM_sample = FM_data[3000:4500]

#Preliminary scatter plots to inspect data
plt.plot(test_room_temp_sample, ls="none", marker='o', markersize=3)
plt.xlabel('Time (200ns samples)')
plt.ylabel('ADC Value (bits)')
plt.title('Test Room Temperature Sample Data')
plt.savefig('test_rt_sample.pdf')
plt.figure()
plt.plot(room_temp_sample, ls="none", marker='o', markersize=3)
plt.xlabel('Time (200ns samples)')
plt.ylabel('ADC Value (bits)')
plt.title('Room Temperature Sample Data')
plt.savefig('rt_sample.pdf')
plt.figure()
plt.plot(ice_data_sample,ls='none', marker='o', markersize=3)
plt.xlabel('Time (200ns samples)')
plt.ylabel('ADC Value (bits)')
plt.title('Ice Sample Data')
plt.savefig('ice_sample.pdf')
plt.figure()
plt.plot(boil1_sample,ls='none', marker='o', markersize=3)
plt.xlabel('Time (200ns samples)')
plt.ylabel('ADC Value (bits)')
plt.title('Boiling Water at 83 degrees Sample Data')
plt.savefig('boil1_sample.pdf')
plt.figure()
plt.plot(boil2_sample,ls='none', marker='o', markersize=3)
plt.xlabel('Time (200ns samples)')
plt.ylabel('ADC Value (bits)')
plt.title('Boiling Water at 61.4 degrees Sample Data')
plt.savefig('boil2_sample.pdf')
plt.figure()
plt.plot(boil3_sample,ls='none', marker='o', markersize=3)
plt.xlabel('Time (200ns samples)')
plt.ylabel('ADC Value (bits)')
plt.title('Boiling Water at 49.3 degrees Sample Data')
plt.savefig('boil3_sample.pdf')
plt.figure()
plt.plot(co2_sample,ls='none', marker='o', markersize=3)
plt.xlabel('Time (200ns samples)')
plt.ylabel('ADC Value (bits)')
plt.title('Dry Ice Sample Data')
plt.savefig('co2_sample.pdf')
plt.figure()
plt.plot(ln2_sample,ls='none', marker='o', markersize=3)
plt.xlabel('Time (200ns samples)')
plt.ylabel('ADC Value (bits)')
plt.title('Liquid Nitrogen Sample Data')
plt.savefig('ln2_sample.pdf')
plt.figure()
plt.plot(LTE_sample,ls='none', marker='o', markersize=3)
plt.xlabel('Time (200ns samples)')
plt.ylabel('ADC Value (bits)')
plt.title('LTE Sample Data')
plt.savefig('LTE_sample.pdf')
plt.figure()
plt.plot(FM_sample,ls='none', marker='o', markersize=3)
plt.xlabel('Time (200ns samples)')
plt.ylabel('ADC Value (bits)')
plt.title('FM Sample Data')
plt.savefig('FM_sample.pdf')

#Histograms of the data to inspect the distributions
#Histogram of test room temperature data
plt.hist(test_room_temp_data, bins=np.arange(-150,150))
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of Test Room Temperature Data')
plt.savefig('test_rt_hist.pdf')
plt.figure()

#Histogram of test room temperature data with logarithmic y-axis
plt.hist(test_room_temp_data, bins=np.arange(-300,300))
plt.semilogy()
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of Test Room Temperature Data with Logarithmic y-axis')
plt.savefig('test_rt_hist_log.pdf')
plt.figure()

#Histogram of room temperature data
plt.hist(room_temp_data, bins=np.arange(-150,150))
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of Room Temperature Data')
plt.savefig('rt_hist.pdf')
plt.figure()

#Histogram of ice data
plt.hist(ice_data, bins=np.arange(-300,300))
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of Ice Temperature Data')
plt.savefig('ice_hist.pdf')
plt.figure()

#Histograms of boiling water data
#83 degrees
plt.hist(boil_water1_data, bins=np.arange(-300, 300))
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of Boiling Water Data at 83 Degrees Celsius')
plt.savefig('boil1_hist.pdf')
plt.figure()

#61.4 degrees
plt.hist(boil_water2_data, bins=np.arange(-300, 300))
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of Boiling Water Data at 61.4 Degrees Celsius')
plt.savefig('boil2_hist.pdf')
plt.figure()

#49.3 degrees
plt.hist(boil_water3_data, bins=np.arange(-300, 300))
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of Boiling Water Data at 49.3 Degrees Celsius')
plt.savefig('boil3_hist.pdf')
plt.figure()

#Histogram of dry ice data
plt.hist(dry_ice_data, bins=np.arange(-300, 300))
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of Dry Ice Data')
plt.savefig('co2_hist.pdf')
plt.figure()

#Histogram of LN2 data
plt.hist(ln2_data, bins=np.arange(-300, 300))
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of Liquid Nitrogen Data')
plt.savefig('ln2_hist.pdf')
plt.figure()

#Histogram of LTE data
plt.hist(LTE_data, bins=np.arange(-200, 200))
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of LTE Data')
plt.savefig('lte_hist.pdf')
plt.figure()

#Histogram of FM data
plt.hist(FM_data, bins=np.arange(-300, 300))
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histogram of FM Data')
plt.savefig('fm_hist.pdf')
plt.figure()

#Plotting all histograms on the same axes 
plt.hist(room_temp_data, bins=np.arange(-300,300), label='Room Temperature')
plt.hist(ice_data, bins=np.arange(-300,300), label='Ice')
plt.hist(boil_water1_data, bins=np.arange(-300, 300), label='83 Degrees')
plt.hist(boil_water2_data, bins=np.arange(-300, 300), label='61.4 Degrees')
plt.hist(boil_water3_data, bins=np.arange(-300, 300), label='49.3 Degrees')
plt.hist(dry_ice_data, bins=np.arange(-300, 300), label='CO2')
plt.hist(ln2_data, bins=np.arange(-300, 300), label='Liquid N2')
plt.xlabel('ADC Value (bits)')
plt.ylabel('Number of Samples')
plt.title('Histograms of All Data')
plt.legend()
plt.savefig('all_hist.pdf')
plt.figure()

#Sections 4.2/4.4
room_temp_data_new = test_room_temp_data - test_room_temp_mean #Subtract the mean from the test room temperature data
room_temp_power = room_temp_data_new**2 #Calculate power using equation 1 (constants can be excluded because the data are not yet in SI units)

#Chi-squared distribution of test room temperature data with 1 degree of freedom
chi1_room_temp = stats.chi2.pdf(np.arange(10, 20000), df=1, loc=0, scale=test_room_temp_var) #Calculate chi^2 with 1dof, mean is 0 because subtracted it from data
plt.hist(room_temp_power, bins=100, range=(0,20000)) #Plot the histogram of the power
plt.plot(chi1_room_temp*1000000*200)#Plot chi^2 times number of data points times range/bins to normalize
plt.loglog() #Plot with logarithmic axes
plt.xlabel('Power (bits^2)')
plt.ylabel('Number of Samples')
plt.title('Test Room Temperature Data vs. chi^2 Distribution with 1 Degree of Freedom')
plt.savefig('chi1.pdf')
plt.figure()

#Chi squared with 2 degrees of freedom
room_temp_power2 = room_temp_power.reshape(2, 500000) #Reshape the data so that the data can be summed over 2 samples
power_df2 = np.sum(room_temp_power2, axis=0) #Calculate the power summed over 2 samples
chi2_room_temp = stats.chi2.pdf(np.arange(0, 30000), df=2, loc=0, scale=test_room_temp_var)  #Calculate chi^2 with 2dof
plt.hist(power_df2, bins=1000, range=(0,30000)) #Plot histogram of power summed over 2 samples
plt.plot(chi2_room_temp*500000*30) #Plot chi^2 distribution with 2 dof
plt.loglog() #Plot with logarithmic axes
plt.xlabel('Power Summed over 2 Samples (bits^2)')
plt.ylabel('Number of Samples')
plt.title('Test Room Temperature Data vs. chi^2 Distribution with 2 Degrees of Freedom')
plt.savefig('chi2.pdf')
plt.figure()

#Chi-squared with 4 degrees of freedom
room_temp_power4 = room_temp_power.reshape(4, 250000) #Reshape data so that it can be summed over 4 samples
power_df4 = np.sum(room_temp_power4, axis=0) #Calculate the power summed over 4 samples
chi4_room_temp = stats.chi2.pdf(np.arange(0, 50000), df=4, loc=0, scale=test_room_temp_var) #Calculate chi^2 with 4dof
plt.hist(power_df4, bins=1000, range=(100,50000)) #Plot the histogram of the power summed over 4 samples
plt.plot(chi4_room_temp*250000*50) #Plot chi^2 distribution with 4dof
plt.loglog() #Plot with logarithmic axes
plt.xlim(100,50000)
plt.xlabel('Power Summed over 4 Samples (bits^2)')
plt.ylabel('Number of Samples')
plt.title('Test Room Temperature Data vs. chi^2 Distribution with 4 Degrees of Freedom')
plt.savefig('chi4.pdf')
plt.figure()

#Chi-squared with 10 degrees of freedom
room_temp_power10 = room_temp_power.reshape(10, 100000) #Reshape data so it can be summed over 10 samples
power_df10 = np.sum(room_temp_power10, axis=0) #Calculate the power summed over 10 samples
chi10_room_temp = stats.chi2.pdf(np.arange(0, 100000), df=10, loc=0, scale=test_room_temp_var) #Calculate chi^2 with 10dof
plt.hist(power_df10, bins=5000, range=(0,100000)) #Plot histogram of power summed over 10 samples
plt.plot(chi10_room_temp*100000*20) #Plot chi^2 distribution with 10dof
#Took out loglog because now closer to being normally distributed
plt.xlabel('Power Summed over 10 Samples (bits^2)')
plt.ylabel('Number of Samples')
plt.title('Test Room Temperature Data vs. chi^2 Distribution with 10 Degrees of Freedom')
plt.savefig('chi10.pdf')
plt.figure()

#Chi-squared with 100 degrees of freedom
room_temp_power100 = room_temp_power.reshape(100, 10000) #Reshape data so it can be summed over 100 samples
power_df100 = np.sum(room_temp_power100, axis=0) #Calculate power summed over 100 samples
chi100_room_temp = stats.chi2.pdf(np.arange(0, 300000), df=100, loc=0, scale=test_room_temp_var) #Calculate chi^2 with 100dof
plt.hist(power_df100, bins=1000, range=(0,300000)) #Plot histogram of power summed over 100 samples
plt.plot(chi100_room_temp*10000*300) #Plot chi^2 distribution with 100dof
plt.xlabel('Power Summed over 100 Samples (bits^2)')
plt.ylabel('Number of Samples')
plt.title('Test Room Temperature Data vs. chi^2 Distribution with 100 Degrees of Freedom')
plt.savefig('chi100.pdf')
plt.figure()

#Plot all of the power histograms and chi^2 distributions on the same axes
#1 degree of freedom
plt.hist(room_temp_power, bins=2500, range=(100,500000), alpha=0.5)
plt.plot(chi1_room_temp*1000000*200, label='N = 1')
#2 degrees of freedom
plt.hist(power_df2, bins=2500, range=(100,500000), alpha=0.5)
plt.plot(chi2_room_temp*500000*200, label='N = 2')
#4 degrees of freedom
plt.hist(power_df4, bins=2500, range=(100,500000), alpha=0.5)
plt.plot(chi4_room_temp*250000*200, label='N = 4')
#10 degrees of freedom
plt.hist(power_df10, bins=2500, range=(100,500000), alpha=0.5)
plt.plot(chi10_room_temp*100000*200, label='N = 10')
#100 degrees of freedom
plt.hist(power_df100, bins=2500, range=(100,500000), alpha=0.5)
plt.plot(chi100_room_temp*10000*200, label='N = 100')
plt.loglog()
plt.legend()
plt.ylim(1,1000000)
plt.xlim(100,500000)
plt.title('AirSpy Data vs. chi^2 Distributions')
plt.ylabel('Number of Samples')
plt.xlabel('Power Summed over N Samples for Test Room Temperature Data (bits^2)')
plt.savefig('allchi.pdf')
plt.figure()

#Averaging over 1000-point chunks to get estimates of temperature
test_room_temp_reshape_1000 = test_room_temp_data.reshape(-1,1000) #Reshape the data into 1000-point chunks
test_room_temp_squared = test_room_temp_reshape_1000**2 #Square all of the data to calculate the energy^2
test_room_temp_avgs = np.mean(test_room_temp_squared,axis=1) #Average over the 1000-point chunks (since averaging over squared data, this is expectation value of energy^2)
test_room_temp_variances = np.var(test_room_temp_reshape_1000, axis=1) #Equivalently, calculate the variance over the 1000-point chunks
test_time_axis = np.arange(0,0.2,0.0002) #Time axis 
plt.plot(test_time_axis,test_room_temp_avgs, ls='', marker='o', markersize=2.5)
#plt.plot(test_time_axis, test_room_temp_variances, ls = '', marker = '+', markersize=2.5) #Same plot
plt.semilogy()
plt.xlabel("Time (s)")
plt.ylabel("System Temperature (bits^2)")
plt.title("System Temperature Estimate vs. Time for Test Room Temperature Data")
plt.savefig('rt_test_temp_estimates.pdf')
plt.figure()

#Calculate mean and standard deviation of expectation value of energy^2
test_rt_mean_temp = np.mean(test_room_temp_avgs) #Calculate mean
test_rt_std_temp = np.std(test_room_temp_avgs) #Calculate standard deviation
print("The mean temperature for the test room temperature data is", test_rt_mean_temp, "+/-", test_rt_std_temp, ".")

#Function for averaging over 1000-point chunks to get estimates of temperature
def temp_avg_1000(x):
    data_reshape_1000 = x.reshape(-1,1000) #Reshape the data into 1000-point chunks
    data_squared = data_reshape_1000**2 #Square the data
    data_avgs = np.mean(data_squared, axis=1) #Average the data to get the expectation value of energy^2
    return data_avgs

#Function for averaging over 1000-point chunks to get variance (equivalent to function above)
def temp_var_1000(x):
    data_reshape_1000 = x.reshape(-1,1000) #Reshape the data into 1000-point chunks
    data_variances = np.var(data_reshape_1000, axis=1) #Calculate the variance over the 1000-point chunks
    return data_variances

#Function for averaging over 100 000-point chunks to get estimates of temperature
def temp_avg_100000(x):
    data_reshape_100000 = x.reshape(-1,100000) #Reshape the data into 100 000-point chunks
    data_squared = data_reshape_100000**2 #Square the data
    data_avgs = np.mean(data_squared, axis=1) #Average the data to get the expectation value of energy^2
    return data_avgs
    
#Temperature timestream estimates for the rest of the data, averaging over 1000 samples
room_temp_avg = temp_avg_1000(room_temp_data) #Calculate temperature estimates for room temperature data
room_temp_mean = np.mean(room_temp_avg) #Calculate mean
room_temp_std = np.std(room_temp_avg) #Calculate standard deviation
print("The mean temperature for the room temperature data is", room_temp_mean, "+/-", room_temp_std, ".")

ice_avg_temp = temp_avg_1000(ice_data) #Calculate temperature estimates for ice data
ice_mean_temp = np.mean(ice_avg_temp) #Calculate mean
ice_std_temp = np.std(ice_avg_temp) #Calculate standard deviation
print("The mean temperature for the ice data is", ice_mean_temp, "+/-", ice_std_temp, ".")

boil1_avg_temp = temp_avg_1000(boil_water1_data) #Calculate temperature estimates for the boiling water data at 83 degrees
boil1_mean_temp = np.mean(boil1_avg_temp) #Calculate mean
boil1_std_temp = np.std(boil1_avg_temp) #Calculate standard deviation
print("The mean temperature for the boiling water data at 93.9 degrees is", boil1_mean_temp, "+/-", boil1_std_temp, ".")


boil2_avg_temp = temp_avg_1000(boil_water2_data) #Calculate temperature estimates for the boiling water at 61.4 degrees
boil2_mean_temp = np.mean(boil2_avg_temp) #Calculate mean
boil2_std_temp = np.std(boil2_avg_temp) #Calculate standard deviation
print("The mean temperature for the boiling water data at 75.6 degrees is", boil2_mean_temp, "+/-", boil2_std_temp, ".")

boil3_avg_temp = temp_avg_1000(boil_water3_data) #Calculate temperature estimates for the boiling water data at 49.3 degrees
boil3_mean_temp = np.mean(boil3_avg_temp) #Calculate mean
boil3_std_temp = np.std(boil3_avg_temp) #Calculate standard deviation
print("The mean temperature for the boiling water data at 74.1 degrees is", boil3_mean_temp, "+/-", boil3_std_temp, ".")

co2_avg_temp = temp_avg_1000(dry_ice_data) #Calculate temperature estimates for the dry ice data
co2_mean_temp = np.mean(co2_avg_temp) #Calculate mean
co2_std_temp = np.std(co2_avg_temp) #Calculate standard deviation
print("The mean temperature for the CO2 data is", co2_mean_temp, "+/-", co2_std_temp, ".")

ln2_avg_temp = temp_avg_1000(ln2_data) #Calculate temperature estimates for the liquid nitrogen data
ln2_mean_temp = np.mean(ln2_avg_temp) #Calculate mean
ln2_std_temp = np.std(ln2_avg_temp) #Calculate standard deviation
print("The mean temperature for the liquid N2 data is", ln2_mean_temp, "+/-", ln2_std_temp, ".")

#Plot all of the temperature estimates on the same axes as a timestream
time_axis = np.arange(0,20,0.0002) #Time is x-axis
plt.plot(time_axis,room_temp_avg, ls='', marker='o', markersize=1, label='Room Temperature')
plt.plot(time_axis,ice_avg_temp, ls='', marker='o', markersize=1, label='Ice Water')
plt.plot(time_axis,boil1_avg_temp, ls='', marker='o', markersize=1, label='93.9 Degrees')
plt.plot(time_axis,boil2_avg_temp, ls='', marker='o', markersize=1, label='75.6 Degrees')
plt.plot(time_axis,boil3_avg_temp, ls='', marker='o', markersize=1, label='74.1 Degrees')
plt.plot(time_axis,co2_avg_temp, ls='', marker='o', markersize=1, label='CO2')
plt.plot(time_axis,ln2_avg_temp, ls='', marker='o', markersize=1, label='Liquid N2')
plt.semilogy()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('System Temperature (bits^2)')
plt.title('System Temperature Estimate vs. Time for AirSpy Data, Averaging over 1000 Points')
plt.savefig('temp_estimates_1000.pdf')
plt.figure()

#Temperature timestream estimates for the rest of the data, averaging over 100 000 samples
room_temp_avg_100000 = temp_avg_100000(room_temp_data) #Calculate temperature estimates for the room temperature data, averaging over 100 000 samples
room_temp_mean = np.mean(room_temp_avg) #Calculate mean
room_temp_std = np.std(room_temp_avg) #Calculate standard deviation 
print("The mean temperature for the room temperature data is", room_temp_mean, "+/-", room_temp_std, ".")

ice_avg_temp_100000 = temp_avg_100000(ice_data) #Calculate temperature estimates for the ice data, averaging over 100 000 samples
ice_mean_temp = np.mean(ice_avg_temp) #Calculate mean
ice_std_temp = np.std(ice_avg_temp) #Calculate standard deviation 
print("The mean temperature for the ice data is", ice_mean_temp, "+/-", ice_std_temp, ".")

boil1_avg_temp_100000 = temp_avg_100000(boil_water1_data) #Calculate temperature estimates for the boiling water data at 83 degrees, averaging over 100 000 samples
boil1_mean_temp = np.mean(boil1_avg_temp) #Calculate mean
boil1_std_temp = np.std(boil1_avg_temp) #Calculate standard deviation 
print("The mean temperature for the boiling water data at 93.9 degrees is", boil1_mean_temp, "+/-", boil1_std_temp, ".")

boil2_avg_temp_100000 = temp_avg_100000(boil_water2_data) #Calculate temperature estimates for the boiling water data at 61.4 degrees, averaging over 100 000 samples
boil2_mean_temp = np.mean(boil2_avg_temp) #Calculate mean
boil2_std_temp = np.std(boil2_avg_temp) #Calculate standard deviation 
print("The mean temperature for the boiling water data at 75.6 degrees is", boil2_mean_temp, "+/-", boil2_std_temp, ".")

boil3_avg_temp_100000 = temp_avg_100000(boil_water3_data) #Calculate temperature estimates for the boiling water data at 49.3 degrees, averaging over 100 000 samples
boil3_mean_temp = np.mean(boil3_avg_temp) #Calculate mean
boil3_std_temp = np.std(boil3_avg_temp) #Calculate standard deviation 
print("The mean temperature for the boiling water data at 74.1 degrees is", boil3_mean_temp, "+/-", boil3_std_temp, ".")

co2_avg_temp_100000 = temp_avg_100000(dry_ice_data) #Calculate temperature estimates for the dry ice data, averaging over 100 000 samples
co2_mean_temp = np.mean(co2_avg_temp) #Calculate mean
co2_std_temp = np.std(co2_avg_temp) #Calculate standard deviation 
print("The mean temperature for the CO2 data is", co2_mean_temp, "+/-", co2_std_temp, ".")

ln2_avg_temp_100000 = temp_avg_100000(ln2_data) #Calculate temperature estimates for the liquid nitrogen data, averaging over 100 000 samples
ln2_mean_temp = np.mean(ln2_avg_temp) #Calculate mean
ln2_std_temp = np.std(ln2_avg_temp) #Calculate standard deviation 
print("The mean temperature for the liquid N2 data is", ln2_mean_temp, "+/-", ln2_std_temp, ".")

#Plot all of the temperature estimates on the same axes as a timestream
time_axis_100000 = np.arange(0,20,0.02)
plt.plot(time_axis_100000,room_temp_avg_100000, ls='', marker='o', markersize=1, label='Room Temperature')
plt.plot(time_axis_100000,ice_avg_temp_100000, ls='', marker='o', markersize=1, label='Ice Water')
plt.plot(time_axis_100000,boil1_avg_temp_100000, ls='', marker='o', markersize=1, label='83 Degrees')
plt.plot(time_axis_100000,boil2_avg_temp_100000, ls='', marker='o', markersize=1, label='61.4 Degrees')
plt.plot(time_axis_100000,boil3_avg_temp_100000, ls='', marker='o', markersize=1, label='49.3 Degrees')
plt.plot(time_axis_100000,co2_avg_temp_100000, ls='', marker='o', markersize=1, label='CO2')
plt.plot(time_axis_100000,ln2_avg_temp_100000, ls='', marker='o', markersize=1, label='Liquid N2')
plt.semilogy()
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('System Temperature (bits^2)')
plt.title('System Temperature Estimate vs. Time for AirSpy Data, Averaging over 100 000 Samples')
plt.savefig('temp_estimates_100000.pdf')
plt.figure()

#Sections 4.3/4.4
#First plot of spectrum for test room temperature data
f_test = np.fft.fft(test_room_temp_data[0:2**19].reshape(-1,1024), axis=1) #Reshape data into n=1024 chunks and run the Fast Fourier Transform over the 1024 chunks
s_test = (f_test.real**2 + f_test.imag**2).sum(axis=0) #Calculate the power using equation 5
freq = 1000 + np.arange(0, 2.5, 2.5/512) #Adding the LO frequency back in to calibrate the frequency axis, using Nyquist's theorem 
plt.plot(freq, s_test[0:512], ls='none', marker='o', markersize=2.5)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Temperature (bits^2)') 
plt.title('Test Room Temperature Spectrum')
plt.savefig('1st_spec.pdf')
plt.figure()

#Spectrum in dB
plt.plot(freq, 10*np.log10(s_test[0:512]), ls='none', marker='o', markersize=2.5)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Temperature (dB arb.)')
plt.title('Test Room Temperature Spectrum in dB')
plt.savefig('test_rt_spec_dB.png')
plt.figure()

#Plot with errorbars
error = np.full((512,), 1/np.sqrt(1000000/2)) #Creating array for the errors, same size as spectrum array, calculated using the radiometer equation
plt.errorbar(freq, 10*np.log10(s_test[0:512]), yerr=error, xerr=None, ls='',marker='o',markersize=3, lw=1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Temperature (dB arb.)')
plt.title('Test Room Temperature Spectrum in dB with Errorbars')
plt.figure()

#Average over 20 000 spectra to get a measurement of the spectrum with 1% uncertainty on the temperature in each spectral bin
test_rt_data_reshape = test_room_temp_data.reshape(20000,-1) #Reshape data into 20 000 chunks of relevant size
f_rt_test = np.fft.fft(test_rt_data_reshape, axis=1) #Fast-Fourier transform and reshaping of data
s_rt_test = (f_rt_test.real**2 + f_rt_test.imag**2) #Calculate power using equation 5
s_rt_test_mean = np.mean(s_rt_test,axis=0) #Averaging over the chunks
freq_test = 1000 + np.arange(0, 2.5, 2.5/25) #Generate frequency axis (x-axis)
error_test_avg = np.full((25,), 1/np.sqrt(25/2)) #Creating array for the errors, the same size as the spectrum array, calculated using the radiometer equation
plt.errorbar(freq_test, 10*np.log10(s_rt_test_mean[:25]), yerr=error_test_avg, xerr=None, ls='none', marker='o', markersize=2.5, lw=1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Temperature (dB arb.)')
plt.title('Test Room Temperature Averaged Spectrum')
plt.savefig('test_avg_spec.pdf')
plt.figure()

#Spectrum function
def specfunc(x):
    fourier = np.fft.fft(x[0:2**19].reshape(-1,1024), axis=1) #Reshape the data into n = 1024 chunks and run the FFT over the chunks
    spec = (fourier.real**2 + fourier.imag**2).sum(axis=0) #Calculate the power using equation 5
    return spec

#Function for averaging over 20 000 spectra (1% fractional error)
def spec_avg(x):
    data_reshape = x.reshape(20000,-1) #Reshape the data into 20 000 chunks of whatever size fits the size of the array
    avg_fft = np.fft.fft(data_reshape,axis=1) #Run the FFT over the chunks
    avg_s = (avg_fft.real**2 + avg_fft.imag**2) #Calculate the power using equation 5
    avg_s_mean = np.mean(avg_s, axis=0) #Average over the power chunks
    return avg_s_mean
    
#LTE and FM spectra
#LTE spectrum
LTE_avg = spec_avg(LTE_data) #Calculate the average spectrum  of the LTE data with 1% fractional error 
error_avg_spec = np.full((2500,), 1/np.sqrt(5000/2)) #Create array for the errors, same size as averaged spectrum array, calculated using the radiometer equation 
freq_axis_new = 1000 + np.arange(0, 2.5, 2.5/2500) #Added LO frequency to calibrate frequency axis, used Nyquist's theorem 
plt.errorbar(freq_axis_new, 10*np.log10(LTE_avg[:2500]), yerr=error_avg_spec, xerr=None, ls='none', marker='o', markersize=2.5, lw=1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Temperature (dB arb.)')
plt.title('LTE Averaged Spectrum')
plt.savefig('LTE_avg_spec.pdf')
plt.figure()

#FM spectrum
FM_avg = spec_avg(FM_data) #Calculate the average spectrum of the FM data  with 1% fractional error 
plt.errorbar(freq_axis_new, 10*np.log10(FM_avg[:2500]), yerr=error_avg_spec, xerr=None, ls='none', marker='o', markersize=2.5, lw=1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Temperature (dB arb.)')
plt.title('FM Averaged Spectrum')
plt.savefig('FM_avg_spec.pdf')
plt.figure()

#All averaged spectra with 1% fractional error, using the spec_avg() function defined above
rt_avg = spec_avg(room_temp_data) #Room temperature data  averaged spectrum
ice_avg = spec_avg(ice_data) #Ice data averaged spectrum
boil1_avg = spec_avg(boil_water1_data) #Boiling water data at 83 degrees averaged spectrum
boil2_avg = spec_avg(boil_water2_data) #Boiling water data at 61.4 degrees averaged spectrum
boil3_avg = spec_avg(boil_water3_data) #Boiling water data at 49.3 degrees averaged spectrum
co2_avg = spec_avg(dry_ice_data) #Dry ice data averaged spectrum
ln2_avg = spec_avg(ln2_data) #Liquid nitrogen data averaged spectrum

#Waterfall plot for test room temperature data
test_rt_waterfall = s_rt_test_mean.reshape(10,5) #Reshape the data so that has allowable dimensions 
plt.imshow(test_rt_waterfall)
plt.colorbar()
plt.title('Waterfall Plot for Test Room Temperature Data')
plt.savefig('test_rt_waterfall.pdf')
plt.figure()

#Waterfall plot for room temperature data
room_temp_waterfall = rt_avg.reshape(100,50) #Reshape the data so that it has allowable dimensions
plt.imshow(room_temp_waterfall)
plt.colorbar()
plt.title('Waterfall Plot for Room Temperature Data')
plt.savefig('rt_waterfall.pdf')
plt.figure()

#Waterfall plot for ice data
ice_waterfall = ice_avg.reshape(100,50) #Reshape the data so that it has allowable dimensions
plt.imshow(ice_waterfall)
plt.colorbar()
plt.title('Waterfall Plot for Ice Data')
plt.savefig('ice_waterfall.pdf')
plt.figure()

#Waterfall plot for boiling water data at 83 degrees 
boil1_waterfall = boil1_avg.reshape(100,50) #Reshape the data so that it has allowable dimensions
plt.imshow(boil1_waterfall)
plt.colorbar()
plt.title('Waterfall Plot for Boiling Water Data at 83 Degrees')
plt.savefig('boil1_waterfall.pdf')
plt.figure()

#Waterfall plot for boiling water data at 61.4 degrees 
boil2_waterfall = boil2_avg.reshape(100,50) #Reshape the data so that it has allowable dimensions
plt.imshow(boil2_waterfall)
plt.colorbar()
plt.title('Waterfall Plot for Boiling Water Data at 61.4 Degrees')
plt.savefig('boil2_waterfall.pdf')
plt.figure()

#Waterfall plot for boiling water data at 49.3 degrees 
boil3_waterfall = boil3_avg.reshape(100,50) #Reshape the data so that it has allowable dimensions
plt.imshow(boil3_waterfall)
plt.colorbar()
plt.title('Waterfall Plot for Boiling Water Data at 49.3 Degrees')
plt.savefig('boil3_waterfall.pdf')
plt.figure()

#Waterfall plot for dry ice data
co2_waterfall = co2_avg.reshape(100,50) #Reshape the data so that it has allowable dimensions
plt.imshow(co2_waterfall)
plt.colorbar()
plt.title('Waterfall Plot for Dry Ice Data')
plt.savefig('co2_waterfall.pdf')
plt.figure()

#Waterfall plot for liquid nitrogen data
ln2_waterfall = ln2_avg.reshape(100,50) #Reshape the data so that it has allowable dimensions
plt.imshow(ln2_waterfall)
plt.colorbar()
plt.title('Waterfall Plot for Liquid N2 Data')
plt.savefig('ln2_waterfall.pdf')
plt.figure()

#Section 4.5
#Load table of load temperature vs. variance data 
t_load, load_err, t_sys, sys_err = np.loadtxt('Tsys_vs_Tload.txt', comments="#", skiprows=1, delimiter=',', unpack = True)

#Plot Tload vs. Tsys with errorbars
plt.errorbar(t_load, t_sys, xerr = load_err, yerr = sys_err, ls='',marker='o',markersize=3, lw=1)
plt.ylabel('System Temperature (bits^2)')
plt.xlabel('Load Temperature (K)')
plt.title('Tsys vs. Tload')
plt.savefig('tsys_vs_tload.pdf')
plt.figure()

#Define linear model function, where y = Tsys, x = Tload, a = p0 = gain, b = p1 = offset
def model_linear(x,a,b): 
    return a*x + b

#Define functions for initial conditions
#Function for gain (p0), based on equation given in lab manual
def p0(x,y,n): #Function takes 3 arguments: load temperature data, system temperature data, and number of data points
    p0 = ((n*np.sum(x*y)) - (np.sum(x)*np.sum(y)))/((n*np.sum(x**2)) - ((np.sum(x))**2))
    return p0

#Function for offset (p1), based on equation given in lab manual 
def p1(x,y,n,p): #Function takes 4 arguments: load temperature data, system temperature data, number of data points, and p0
    p1 = (1/n)*(np.sum(y) - p*np.sum(x))
    return p1

#Calculate gain and offset for initial plot above
p0_1 = p0(t_load, t_sys, 7)
p1_1 = p1(t_load, t_sys, 7, p0_1)

#Plot Tload vs Tsys with the linear fit
plt.errorbar(t_load, t_sys, xerr = load_err, yerr = sys_err, ls='',marker='o',markersize=3, lw=1) #Plot the data with errorbars, as above
plt.plot(t_load, model_linear(t_load,p0_1, p1_1)) #Plot the linear fit
plt.ylabel('System Temperature (bits^2)')
plt.xlabel('Load Temperature (K)')
plt.title('Linear Fit of Tsys vs. Tload')
print('The gain is', p0_1, 'and the offset/receiver temperature is', p1_1)
print('The x-intercept is', (-p1_1)/(p0_1))
plt.savefig('linear_fit.pdf')
plt.figure()

#Section 4.6
#Plot all averaged spectra on the same axes
error_avg_spec = np.full((2500,), 1/np.sqrt(5000/2)) #Create array for errors, same size as averaged spectra arrays, calculated using the radiometer equation 
freq_axis_new = 1000 + np.arange(0, 2.5, 2.5/2500) #Add the LO frequency to calibrate the frequency axis, use Nyquist's theorem 
plt.errorbar(freq_axis_new, 10*np.log10(rt_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Room Temperature')
plt.errorbar(freq_axis_new, 10*np.log10(ice_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Ice')
plt.errorbar(freq_axis_new, 10*np.log10(boil1_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Boiling Water 83C')
plt.errorbar(freq_axis_new, 10*np.log10(boil2_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Boiling Water 61.4C')
plt.errorbar(freq_axis_new, 10*np.log10(boil3_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Boiling Water 49.3C')
plt.errorbar(freq_axis_new, 10*np.log10(co2_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='CO2')
plt.errorbar(freq_axis_new, 10*np.log10(ln2_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Liquid N2')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Temperature (dB arb.)')
plt.title('All Averaged Spectra')
plt.legend()
plt.savefig('all_spec.pdf')
plt.figure()

#Plot all averaged spectra on the same axes with limited y-range in order to see the variation in the spectra more clearly
plt.errorbar(freq_axis_new, 10*np.log10(rt_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Room Temperature')
plt.errorbar(freq_axis_new, 10*np.log10(ice_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Ice')
plt.errorbar(freq_axis_new, 10*np.log10(boil1_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Boiling Water 83C')
plt.errorbar(freq_axis_new, 10*np.log10(boil2_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Boiling Water 61.4C')
plt.errorbar(freq_axis_new, 10*np.log10(boil3_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Boiling Water 49.3C')
plt.errorbar(freq_axis_new, 10*np.log10(co2_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='CO2')
plt.errorbar(freq_axis_new, 10*np.log10(ln2_avg[:2500]), xerr=None, yerr=error_avg_spec, ls='none', marker='o', markersize=1, lw=1, label='Liquid N2')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Temperature (dB arb.)')
plt.title('All Averaged Spectra')
plt.ylim(64,68)
plt.legend()
plt.savefig('all_spec_y.pdf')
plt.figure()

#Create a new array made up of all of the averaged spectra stacked on top of each other in the correct order (i.e. coldest to hottest)
all_variances = np.stack((ln2_avg, co2_avg, ice_avg, rt_avg, boil3_avg, boil2_avg, boil1_avg))

#Calculate all values of p0 and p1 and put them into arrays
p0_array = np.zeros(5000) #Create an array of zeros that is the same size as the averaged spectra arrays, to append to later
p1_array = np.zeros(5000) #Create an array of zeros that is the same size as the averaged spectra arrays, to append to later
for i in range(0, 5000): #Iterate through all of the values in the range of the averaged spectra arrays
    p0_array[i] = p0(t_load, all_variances[:,i], 7) #Calculate p0 for each column of values
    p1_array[i] = p1(t_load, all_variances[:,i], 7, p0_array[i]) #Calculate p1 for each column of values 

#Plot p0 and p1 against frequency
#Plot of gain against frequency
error_p = np.full((2500,), 1/np.sqrt(5000/2))  #Create array for errors, same size as arrays for p0 and p1, calculated using the radiometer equation 
plt.errorbar(freq_axis_new, p0_array[:2500], xerr=None, yerr=error_p, ls='', marker='o',markersize=1, lw=1)
plt.ylabel('Gain (bits^2/K)')
plt.xlabel('Frequency (MHz)')
plt.ylim(-10,5000)
plt.savefig('gain.pdf')
plt.figure()
#Plot of offset against frequency
plt.errorbar(freq_axis_new, p1_array[:2500], xerr=None, yerr=error_p, ls='', marker='o', markersize=1, lw=1)
plt.ylabel('Offset (M bits^2)')
plt.xlabel('Frequency (MHz)')
plt.ylim(0,8000000)
plt.savefig('offset.pdf')
plt.figure()

#Plots of gain and offset on logarithmic axes
#Plot of gain on logarithmic axes
plt.errorbar(freq_axis_new, p0_array[:2500], xerr=None, yerr=error_p, ls='', marker='o',markersize=1, lw=1)
plt.ylabel('Gain (bits^2/K)')
plt.xlabel('Frequency (MHz)')
plt.loglog()
plt.savefig('gain_log.pdf')
plt.figure()
#Plot of offset on logarithmic axes
plt.errorbar(freq_axis_new, p1_array[:2500], xerr=None, yerr=error_p, ls='', marker='o', markersize=1, lw=1)
plt.ylabel('Offset (M bits^2)')
plt.xlabel('Frequency (MHz)')
plt.loglog()
plt.savefig('offset_log.pdf')
plt.figure()

#Plot of receiver temperature 
temperature_final = p1_array/p0_array #Receiver temperature = offset/gain
plt.plot(freq_axis_new, temperature_final[0:2500], ls='', marker='o',markersize=1)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Temperature of the Receiver (K)')
plt.ylim(-1000,12000)
plt.savefig('t_rcvr.pdf')