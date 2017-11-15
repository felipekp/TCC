import os
import time 
import logging

def setup_logger(logger_name, log_file, level=logging.INFO):
    """
        Function to create a logger
        :param logger_name: unique name to identify the logger
        :param log_file: relative name with path of file of log
        :param level: level of logging to be used
    """
    l = logging.getLogger(logger_name)
    l.propagate = False
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(funcName)20s() %(levelname)-8s %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)


def write_new_csv(df, prefix, filename, county, state, start_year, end_year):
    """
        Saves the dataframe inside a new file in a new path (a folder with a given prefix)
        :param df: dataframe with the modified data
        :param prefix: prefix of the path
        :param filename: filename from file being read (file name will stay the same)
        :param county: county number
        :param state: state number
        :param start_year: starting year
        :param end_year: end year
    """
    newpath = state + '/' + county + '/' + prefix + start_year + '-'+ end_year
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    df.to_csv(os.path.join(newpath, filename))


# --- measuring time                                     
def timeit(method):
    """
        Decorator that measures time of functions
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed
