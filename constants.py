
# Input the filenames of the processed (input) and synthetic (output) data
# NOTE: can be .pkl or .csv
InputFilepath = 'Input/MODEL_INPUT.csv'
OutputFilepath = 'Output/MODEL_OUTPUT.csv'

# Input the filename of the subgroup definitions and sum
SubgroupFilepath = 'Targets/SUBGROUP_TARGETS.csv'

# Input the column index (an integer) corresponding to each spend component of interest in the subgroup file.
ColumnIndex = {
    '$_ACCOMMODATION_TOTAL': 5,
    '$_F&B_TOTAL': 6,
    '$_TRANSPORT_TOTAL': 7,
    '$_S&E_TOTAL': 9,
    '$_SHOPPING_TOTAL': 8
}

# Set the bounds of perturbation for each component of interest.
PerturbationBounds = {
    '$_ACCOMMODATION_TOTAL':
    {
        'min_value': 0,
        'min_mult': 0.5,
        'max_mult': 4
    },
    '$_F&B_TOTAL':
    {
        'min_value': 0,
        'min_mult': 0.5,
        'max_mult': 4
    },
    '$_TRANSPORT_TOTAL':
    {
        'min_value': 0,
        'min_mult': 0.5,
        'max_mult': 4
    },
    '$_S&E_TOTAL':
    {
        'min_value': 0,
        'min_mult': 0.5,
        'max_mult': 4
    },
    '$_SHOPPING_TOTAL':
    {
        'min_value': 0,
        'min_mult': 0.5,
        'max_mult': 4
    }
}
