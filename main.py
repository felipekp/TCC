from handle_data import clean
from handle_data import refine
from handle_data import merge

def main():
    # TODO: all of these calls need to have verifications at the beggining.. verify if everything is valid.. and make sure the useful parameters are being given here
    
    # clean.clean_data_8h(p_start_year='2000', p_end_year='2016', county='113', state='48')

    # refine.refine_data(p_start_year='2000', p_end_year='2016', county='113', prefix='8haverage-clean-', state='48', site='0069')

    merge.merge_data(p_start_year='2000', p_end_year='2016', county='113', prefix='8haverage-refine-', state='48', site='0069')


if __name__ == "__main__":
    main()

