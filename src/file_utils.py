import os
import logging

def safe_open_dir(dirpath):
    if not os.path.isdir(dirpath):
        logging.info("Directory %s does not exist, creating it" % dirpath)
        os.mkdir(dirpath)
    return dirpath