from handle_data import clean
from handle_data import refine
from handle_data import merge
from handle_data import preparation
from handle_data import feat_extract

def main():
    """
    Main function - More abstract and now only calls packages underneath to handle data. In this file we have access to everything implemented for handling data.
    This file contains the flow for: cleaning, refining, merging, preparing (shifting time steps ahead for prediction), and feature extraction methods.

    Hardcoded are the parameters necessary and can be modified depending on the folder structure.
    """
    p_start_year='1990'
    p_end_year='2017'
    county='113'
    state='48'
    site=False # either False or a vector with the number of the site, like: ['0069']
    predict_var = '44201_0069' # name of column that we want to predict
    timesteps = '3' # number of steps ahead to look


    clean_output_path = '8hmax-clean-' # variable to define the path of the clean output process

    refine_input_path = clean_output_path # var to define input of the process that re-indexes the data.
    refine_output_path = '8hmax-refine-'

    merge_input_path = refine_output_path # var with path for process that imputes data
    merge_output_path = 'datasets/multi_sites_8hmax-merged_'

    prepare_input_path = merge_output_path # var for process that shifts data in time
    prepare_output_path = 'datasets/multi_sites_8hmax-prepared_' # final path before any extraction feature method

    algs_to_use = [3] # where 0: decision_tree, 3: pca. Inside a dict declared inside feat_extract.
    extracted_input_path = prepare_output_path + predict_var + '-' + timesteps + '_'
    extracted_output_path = 'datasets/multi_sites_8hmax-extracted_' + predict_var + '-' + timesteps + '_' # final path with the extracted features

    # clean.clean_data_8h(p_start_year, p_end_year, county, clean_output_path, site, state)

    refine.refine_data(p_start_year, p_end_year, county, refine_input_path, refine_output_path, state)
    #
    # merge.merge_data(p_start_year, p_end_year, county, merge_input_path, merge_output_path, state)
    #
    # preparation.prepare(p_start_year, p_end_year, county, prepare_input_path, prepare_output_path, predict_var, timesteps)
    #
    # feat_extract.extract_features(p_start_year, p_end_year, algs_to_use, county, extracted_input_path, extracted_output_path, predict_var, timesteps)


if __name__ == "__main__":
    main()

