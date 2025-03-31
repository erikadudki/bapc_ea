from scripts_bapc_ea.helper_functions import *

# Call this function (rountine_get_person_id_to_datapoint_oi) to
# find the person_ID, which corresponds to a given
# datapoint of interest.
# This gives a sub-dataframe, with all datapoints that are
# equal to the given datapoint_of_interest.
# The person_ID is given in the column "Index" of df_y_sub.

# json_dir = 'base_model/read_simulated_data.json'

datapoint_of_interest = [3, 4]
json_dir = "base_model/read_Foundercheck_data.json"
json_dec_tree = "base_model/read_dec_tree_parameters.json"
df_y_sub = routine_get_person_id_to_datapoint_oi(json_dir,
                                                 json_dec_tree,
                                                 datapoint_of_interest)