from scripts_bapc_ea.error_amplification_hdtree import *

# Read Simulation Data.
# json_dir = 'base_model/read_simulated_data.json'
# agg_data, data_specs = read_simulated_data(json_dir)

# Read Foundercheck Data.
json_data_dir = "base_model/Algorithms_Dez2023/read_Foundercheck_data.json"
# Read decision_tree_parameters
json_dec_tree_params = "base_model/Algorithms_Dez2023/read_dec_tree_parameters.json"

# In order to find the person_id_to_check:
# run the script: get_person_id.py.
person_id_to_check = 244

run_error_amplification_routine(json_data_dir,
                                json_dec_tree_params,
                                person_id_to_check)