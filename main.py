from handle_data import clean
from handle_data import refine
from handle_data import merge
from handle_data import preparation
from handle_data import feat_extract

def main():
    """
    Main function - More abstract and now only calls packages underneath to handle data. In this file we have access to everything implemented.

    Hardcoded are the parameters necessary and can be modified depending on the folder structure.
    """
    p_start_year='2000'
    p_end_year='2016'
    county='113'
    state='48'
    site='0069'
    predict_var = '44201_0069'
    timesteps = '3'


    clean_output_path = '8hmax-clean-'

    refine_input_path = '8hmax-clean-'
    refine_output_path = '8hmax-refine-'

    merge_input_path = '8hmax-refine-'
    merge_output_path = 'datasets/8hmax-merged_'

    prepare_input_path = 'datasets/8hmax-merged_'
    prepare_output_path = 'datasets/8hmax-prepared_'

    algs_to_use = [3]
    extracted_input_path = 'datasets/8hmax-prepared_' + predict_var + '-' + timesteps + '_'
    extracted_output_path = 'datasets/8hmax-extracted_' + predict_var + '-' + timesteps + '_'


    # clean.clean_data_8h(p_start_year, p_end_year, county, clean_output_path, state, site)
    #
    # refine.refine_data(p_start_year, p_end_year, county, refine_input_path, refine_output_path, state, site)
    #
    # merge.merge_data(p_start_year, p_end_year, county, merge_input_path, merge_output_path, state, site)
    #
    # preparation.prepare(p_start_year, p_end_year, county, prepare_input_path, prepare_output_path, predict_var, timesteps, state, site)
    #
    feat_extract.extract_features(p_start_year, p_end_year, algs_to_use, county, extracted_input_path, extracted_output_path, predict_var, timesteps, state, site)



if __name__ == "__main__":
    main()

