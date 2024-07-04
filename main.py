import pandas as pd
# import cupy as cp
import numpy as cp
import optuna
import os

from constants import InputFilepath, OutputFilepath, ColumnIndex, PerturbationBounds, SubgroupFilepath
from auxiliary import getFilters, getFilterIndex
#from cost_functions import ComputeKLD, ComputeProximity
from optuna_functions import Optimize


def main():
    
    # Read the datasets
    input_df = pd.read_csv(InputFilepath)
    output_df = pd.read_csv(OutputFilepath)
    subgroups = pd.read_csv(SubgroupFilepath)
    
    # For each subgroup, isolate the data into an array
    for ii in range(10): #range(subgroups.shape[0]):
        subgroup_defn = getFilters(ii, input_df.columns, subgroups)
        input_idx = getFilterIndex(input_df, subgroup_defn)
        output_idx = getFilterIndex(output_df, subgroup_defn)
        
        print(f"\nSubgroup: {subgroup_defn}\n")
        print(" Target \t|\t Before \t|\t After ")
        print("---------------------------------------------------------")
        
        for key, c_index in ColumnIndex.items():
            
            target = subgroups.iloc[ii, c_index]
            input_array = input_df.loc[input_idx, key]
            output_array = output_df.loc[output_idx, key]
            
            # Convert pandas DataFrame to CuPy array for GPU acceleration
            original_input = cp.asarray(input_array, dtype=cp.float32)
            original_output = cp.asarray(output_array, dtype=cp.float32)
            
            # Get bounds of perturbation
            bounds = PerturbationBounds[key]
            
            # Set up optimization study
            perturbed_array, cost = Optimize(original_output, original_input, target, bounds, n_trials = 5)
            
            # Insert perturbed array into output_df
            output_df.loc[output_idx, key] = perturbed_array
            
            print(f"{target}\t|\t{cp.round(cp.sum(output_array),2)}\t|\t{cp.round(cp.sum(perturbed_array),2)}")
            
    output_df.to_csv('Output/PERTURBED_OUTPUT.csv')
    

if __name__ == "__main__":
    main()
