from handle_data import clean as clean


def main():
    clean.clean_data_8h(p_start_year='2000', p_end_year='2016', county='113', state='48')


if __name__ == "__main__":
    main()

