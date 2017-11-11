from handle_data import clean
from handle_data import refine
from handle_data import merge

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
    clean_output_path = '8haverage-clean-'

    refine_input_path = '8haverage-clean-'
    refine_output_path = '8haverage-refine-'

    merge_input_path = '8haverage-refine-'
    merge_output_path = 'datasets/scaller-8haverage-merged_'


    clean.clean_data_8h(p_start_year, p_end_year, county, state, site, clean_output_path)

    refine.refine_data(p_start_year, p_end_year, county, refine_input_path, refine_output_path, state, site)

    merge.merge_data(p_start_year, p_end_year, county, merge_input_path, merge_output_path, state, site)


if __name__ == "__main__":
    main()

