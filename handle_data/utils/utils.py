import os
import time 
import logging

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    l.propagate = False
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(funcName)20s() %(levelname)-8s %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    # streamHandler = logging.StreamHandler(stream=None)
    # streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    # l.addHandler(streamHandler)


def write_new_csv(df, prefix, filename, county, state, start_year, end_year):
    """
        Saves the dataframe inside a new file in a new path (a folder with 'clean-' as prefix)
        :param df: dataframe with the modified data
        :param filename: filename from file being read (file name will stay the same)
        :param county: county number
    """
    # logging.info('Saving file into new folder')
    newpath = state + '/' + county + '/' + prefix + start_year + '-'+ end_year
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    df.to_csv(os.path.join(newpath, filename))

# --- measuring time                                     
def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed
