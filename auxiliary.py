import pandas as pd
import sys


def getFilters(index, feature_cols, subgroups_df):
    '''
    FUNCTION to get the subgroup definition from the subgroup dataframe.
    
    index: the index of the subgroup of interest (a row in subgroups_df)
    feature_cols: the feature columns in the input/output data
    subgroups_df: a pandas dataframe with the subgroup definitions and the target sums
    
    value_filters: the definition of the subgroup of interest as a dictionary
    '''
    assert isinstance(subgroups_df, pd.DataFrame), "Require subgroups_df to be a pandas dataframe."
    assert index < subgroups_df.shape[0]
    
    try:
        defn_cols = [col for col in subgroups_df.columns if col in feature_cols]
        value_filters = dict()
        
        for col in defn_cols:
            value_filters[col] = subgroups_df.loc[index, col]
        return value_filters
            
    except Exception as e:
        print(str(e))
    

def getFilterIndex(input_df, value_filters):
    '''
    FUNCTION to get the index associated with a value filter applied to the columns of the input dataframe.
    
    input_df: the input pandas dataframe
    value_filters: a dictionary containing column names as keys and the value to filter by as values
    
    index: a list of rows that satisfy the value_filters criteria
    '''
    assert isinstance(input_df, pd.DataFrame), "Require input_df to be a pandas dataframe."
    assert isinstance(value_filters,
                      dict), "Require value_filters to be a dictionary with column names as keys and the value to filter by as values."
    assert all(col in list(input_df.columns) for col in list(
        value_filters.keys())), f"The columns {list(value_filters.keys())} do not exist in the input dataframe."

    try:
        output_df = input_df.copy()
        for key, value in value_filters.items():
            if isinstance(value, str) and '[' in value and ']' in value:
                items = value.strip('[]').split(',')
                value = [item.strip(" '") for item in items]
            if isinstance(value, list): # if in list ("OR")
                output_df = output_df.loc[output_df[key].isin(value)]
            else:
                output_df = output_df[output_df[key] == value]
        return list(output_df.index)

    except Exception as e:
        print(str(e))
