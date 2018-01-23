import os
import logging

def safe_open_dir(dirpath):
    if os.path.isdir(dirpath):
        return
    logging.info("Directory %s does not exist, creating it" % dirpath)
    os.mkdir(dirpath)