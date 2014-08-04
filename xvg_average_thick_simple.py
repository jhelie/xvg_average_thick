#generic python modules
import argparse
import operator
from operator import itemgetter
import sys, os, shutil
import os.path

##########################################################################################
# RETRIEVE USER INPUTS
##########################################################################################

#=========================================================================================
# create parser
#=========================================================================================
version_nb = "0.0.1"
parser = argparse.ArgumentParser(prog = 'xvg_average_thick_simple', usage='', add_help = False, formatter_class = argparse.RawDescriptionHelpFormatter, description =\
'''
************************************************
v''' + version_nb + '''
author: Jean Helie (jean.helie@bioch.ox.ac.uk)
git: https://github.com/jhelie/xvg_average_thick
************************************************

[ DESCRIPTION ]
 
This script calculate the average of thickness data contained in several xvg files.

It calculates the avg and std dev for both the avg and std dev given as input (i.e.
the avg and std dev inputs are both just treated as metrics)

NB:
the script may give out a warning 'return np.mean(x,axis)/factor', it's ok. it's just
scipy warning us that there were only nans on a row, the result will be a nan as we
expect (see this thread: https://github.com/scipy/scipy/issues/2898).

[ USAGE ]

Option	      Default  	Description                    
-----------------------------------------------------
-f			: xvg file(s)
-o		thick_avg	: name of outptut file
--membrane		: 'AM_zCter','AM_zNter','SMa','SMz' or 'POPC'
--comments	@,#	: lines starting with these characters will be considered as comment

Other options
-----------------------------------------------------
--version		: show version number and exit
-h, --help		: show this menu and exit
 
''')

#options
parser.add_argument('-f', nargs='+', dest='xvgfilenames', help=argparse.SUPPRESS, required=True)
parser.add_argument('-o', nargs=1, dest='output_file', default=["thick_avg"], help=argparse.SUPPRESS)
parser.add_argument('--membrane', dest='membrane', choices=['AM_zCter','AM_zNter','SMa','SMz','POPC'], default='not specified', help=argparse.SUPPRESS, required=True)
parser.add_argument('--comments', nargs=1, dest='comments', default=['@,#'], help=argparse.SUPPRESS)

#other options
parser.add_argument('--version', action='version', version='%(prog)s v' + version_nb, help=argparse.SUPPRESS)
parser.add_argument('-h','--help', action='help', help=argparse.SUPPRESS)

#=========================================================================================
# store inputs
#=========================================================================================

args = parser.parse_args()
args.output_file = args.output_file[0]
args.comments = args.comments[0].split(',')

#=========================================================================================
# import modules (doing it now otherwise might crash before we can display the help menu!)
#=========================================================================================

#generic science modules
try:
	import numpy as np
except:
	print "Error: you need to install the np module."
	sys.exit(1)
try:
	import scipy
	import scipy.stats
except:
	print "Error: you need to install the scipy module."
	sys.exit(1)

#=======================================================================
# sanity check
#=======================================================================

if len(args.xvgfilenames) == 1:
	print "Error: only 1 data file specified."
	sys.exit(1)
	
for f in args.xvgfilenames:
	if not os.path.isfile(f):
		print "Error: file " + str(f) + " not found."
		sys.exit(1)

##########################################################################################
# FUNCTIONS DEFINITIONS
##########################################################################################

#=========================================================================================
# data loading
#=========================================================================================

def load_xvg():															#DONE
	
	global nb_rows
	global nb_cols
	global weights
	global data_thick_avg
	global data_thick_std
	nb_rows = 0
	nb_cols = 0
	weights = np.ones(len(args.xvgfilenames))
		
	for f_index in range(0,len(args.xvgfilenames)):
		#display progress
		progress = '\r -reading file ' + str(f_index+1) + '/' + str(len(args.xvgfilenames)) + '                      '  
		sys.stdout.flush()
		sys.stdout.write(progress)
		
		#get file content
		filename = args.xvgfilenames[f_index]
		with open(filename) as f:
			lines = f.readlines()
		
		#determine legends and nb of lines to skip
		tmp_nb_rows_to_skip = 0
		for l_index in range(0,len(lines)):
			line = lines[l_index]
			if line[0] in args.comments:
				tmp_nb_rows_to_skip += 1
				if "weight" in line:
					if "-> weight = " in line:
						weights[f_index] = float(line.split("-> weight = ")[1])
						if weights[f_index] < 0:
							print "\nError: the weight in file " + str(filename) + " should be a positive number."
							print " -> " + str(line)
							sys.exit(1)
					else:
						print "\nWarning: keyword 'weight' found in the comments of file " + str(filename) + ", but weight not read in as the format '-> weight = ' wasn't found."
		
		#get data
		tmp_data = np.loadtxt(filename, skiprows = tmp_nb_rows_to_skip)
		
		#check that each file has the same number of data rows
		if f_index == 0:
			nb_rows = np.shape(tmp_data)[0]
			data_thick_avg = np.zeros((nb_rows, len(args.xvgfilenames) + 1))			#distance, thick avg for each file
			data_thick_std = np.zeros((nb_rows, len(args.xvgfilenames)))				#thick std for each file
		else:
			if np.shape(tmp_data)[0] != nb_rows:
				print "Error: file " + str(filename) + " has " + str(np.shape(tmp_data)[0]) + " data rows, whereas file " + str(args.xvgfilenames[0]) + " has " + str(nb_rows) + " data rows."
				sys.exit(1)
		#check that each file has the same number of columns
		if f_index == 0:
			nb_cols = np.shape(tmp_data)[1]
		else:
			if np.shape(tmp_data)[1] != nb_cols:
				print "Error: file " + str(filename) + " has " + str(np.shape(tmp_data)[1]) + " data columns, whereas file " + str(args.xvgfilenames[0]) + " has " + str(nb_cols) + " data columns."
				sys.exit(1)
		#check that each file has the same first column
		if f_index == 0:
			data_thick_avg[:,0] = tmp_data[:,0]
		else:
			if not np.array_equal(tmp_data[:,0],data_thick_avg[:,0]):
				print "\nError: the first column of file " + str(filename) + " is different than that of " + str(args.xvgfilenames[0]) + "."
				sys.exit(1)
		
		#store data
		if args.membrane in ["AM_zCter","AM_zNter","SMa"]:
			data_thick_avg[:, f_index + 1] = tmp_data[:,4]
			data_thick_std[:, f_index] = tmp_data[:,8]
		elif args.membrane == "SMz":
			data_thick_avg[:, f_index + 1] = tmp_data[:,3]
			data_thick_std[:, f_index] = tmp_data[:,6]
		elif args.membrane == "POPC":
			data_thick_avg[:, f_index + 1] = tmp_data[:,2]
			data_thick_std[:, f_index] = tmp_data[:,4]
	return

#=========================================================================================
# core functions
#=========================================================================================

def calculate_avg():													#DONE

	global avg_thick_avg
	global avg_thick_std
	global std_thick_avg
	global std_thick_std
				
	avg_thick_avg = np.zeros((nb_rows, 2))
	avg_thick_std = np.zeros((nb_rows, 1))
	std_thick_avg = np.zeros((nb_rows, 1))
	std_thick_std = np.zeros((nb_rows, 1))

	#distances
	avg_thick_avg[:,0] = data_thick_avg[:,0]

	#remove nan values of the weights for average values
	weights_nan_avg = np.zeros((nb_rows, 1))	
	weights_nan_avg_sq = np.zeros((nb_rows, 1))	
	nb_files_avg = np.ones((nb_rows, 1)) * len(args.xvgfilenames)
	tmp_weights_nan = np.zeros((nb_rows, len(args.xvgfilenames)))
	for r in range(0, nb_rows):
		tmp_weights_nan[r,:] = weights
		for f_index in range(0, len(args.xvgfilenames)):
			if np.isnan(data_thick_avg[r,f_index + 1]):
				tmp_weights_nan[r,f_index] = 0
				nb_files_avg[r,0] -= 1
	weights_nan_avg[:,0] = np.nansum(tmp_weights_nan, axis = 1)
	weights_nan_avg_sq[:,0] = np.nansum(tmp_weights_nan**2, axis = 1)	
	weights_nan_avg[weights_nan_avg == 0] = 1
	
	#remove nan values of the weights for std dev values
	weights_nan_std = np.zeros((nb_rows, 1))	
	weights_nan_std_sq = np.zeros((nb_rows, 1))	
	nb_files_std = np.ones((nb_rows, 1)) * len(args.xvgfilenames)
	tmp_weights_nan = np.zeros((nb_rows, len(args.xvgfilenames)))
	for r in range(0, nb_rows):
		tmp_weights_nan[r,:] = weights
		for f_index in range(0, len(args.xvgfilenames)):
			if np.isnan(data_thick_std[r,f_index]):
				tmp_weights_nan[r,f_index] = 0
				nb_files_std[r,0] -= 1
	weights_nan_std[:,0] = np.nansum(tmp_weights_nan, axis = 1)
	weights_nan_std_sq[:,0] = np.nansum(tmp_weights_nan**2, axis = 1)	
	weights_nan_std[weights_nan_std == 0] = 1

	#calculate weighted average taking into account "nan"
	#----------------------------------------------------
	avg_thick_avg[:,1] =  scipy.stats.nanmean(data_thick_avg[:,1:] * weights * nb_files_avg / weights_nan_avg, axis = 1)
	avg_thick_std[:,0] =  scipy.stats.nanmean(data_thick_std * weights * nb_files_std / weights_nan_std , axis = 1)

	#calculate unbiased weighted std dev taking into account "nan"
	#-------------------------------------------------------------
	tmp_avg = np.zeros((nb_rows, 1))
	tmp_std = np.zeros((nb_rows, 1))
	tmp_avg[:,0] = np.nansum(weights * (data_thick_avg[:,1:] - avg_thick_avg[:,1:2])**2, axis = 1)
	tmp_std[:,0] = np.nansum(weights * (data_thick_std - avg_thick_std)**2, axis = 1)
	tmp_div_avg = np.copy((weights_nan_avg)**2 - weights_nan_avg_sq)
	tmp_div_avg[tmp_div_avg == 0] = 1
	tmp_div_std = np.copy((weights_nan_std)**2 - weights_nan_std_sq)
	tmp_div_std[tmp_div_std == 0] = 1
	std_thick_avg = np.sqrt(weights_nan_avg / tmp_div_avg * tmp_avg)
	std_thick_std = np.sqrt(weights_nan_std / tmp_div_std * tmp_std)
				
	return

#=========================================================================================
# outputs
#=========================================================================================

def write_xvg():														#DONE

	#open files
	filename_xvg = os.getcwd() + '/' + str(args.output_file) + '.xvg'
	output_xvg = open(filename_xvg, 'w')
	
	#general header
	output_xvg.write("# [average xvg - written by xvg_average_thick_simple v" + str(version_nb) + "]\n")
	tmp_files = ""
	for f in args.xvgfilenames:
		tmp_files += "," + str(f)
	output_xvg.write("# - files: " + str(tmp_files[1:]) + "\n")
	if np.sum(weights) > len(args.xvgfilenames):
		output_xvg.write("# -> weight = " + str(np.sum(weights)) + "\n")
	
	#xvg metadata
	output_xvg.write("@ title \"Average xvg\"\n")
	output_xvg.write("@ xaxis label \"distance from cluster z axis (Angstrom)\"\n")
	output_xvg.write("@ yaxis label \"order parameter\"\n")
	output_xvg.write("@ autoscale ONREAD xaxes\n")
	output_xvg.write("@ TYPE XY\n")
	output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
	output_xvg.write("@ legend on\n")
	output_xvg.write("@ legend box on\n")
	output_xvg.write("@ legend loctype view\n")
	output_xvg.write("@ legend 0.98, 0.8\n")
	output_xvg.write("@ legend length 4\n")
	output_xvg.write("@ s0 legend \"thick avg (avg)\"\n")
	output_xvg.write("@ s1 legend \"thick avg (std)\"\n")
	output_xvg.write("@ s2 legend \"thick std (avg)\"\n")
	output_xvg.write("@ s3 legend \"thick std (std)\"\n")

	#data
	for r in range(0, nb_rows):
		results = str(avg_thick_avg[r,0])
		results += "	" + "{:.6e}".format(avg_thick_avg[r,1]) + "	" + "{:.6e}".format(std_thick_avg[r,0]) + "	" + "{:.6e}".format(avg_thick_std[r,0])+ "	" + "{:.6e}".format(std_thick_std[r,0])
		output_xvg.write(results + "\n")		
	output_xvg.close()	
	
	return

##########################################################################################
# MAIN
##########################################################################################

print "\nReading files..."
load_xvg()

print "\n\nWriting average file..."
calculate_avg()
write_xvg()

#=========================================================================================
# exit
#=========================================================================================
print "\nFinished successfully! Check result in file '" + args.output_file + ".xvg'."
print ""
sys.exit(0)
