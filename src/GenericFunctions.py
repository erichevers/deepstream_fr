#
# eev: these functions can be used by every process
# version: 1
#

# imports
from datetime import datetime      # to use date and time for timestamp
import logging
from logging.handlers import RotatingFileHandler


# function to initialize logfile
def init_log(
    il_logfile_path,
    il_process,
    il_loglevel,
    il_size,
    il_backups
):

    numeric_level = getattr(logging, il_loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % il_loglevel)
    # create logger
    logger = logging.getLogger(il_process)
    logger.setLevel(logging.DEBUG)  # this defines the maximum loglevel for each of the loggers (screen or file)

    # create console handler and set level to info (overrule default, unless it is debug) for messages on screen
    screen_handler = logging.StreamHandler()
    if il_loglevel.upper() == 'DEBUG':
        screen_handler.setLevel(logging.DEBUG)
        # print(f'-- Screen handler set to: Debug level')
    else:
        screen_handler.setLevel(logging.INFO)
        # print(f'-- Screen handler set to: Info level')

    # create formatter
    formatter = logging.Formatter(fmt='%(asctime)s# %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
    # add formatter to screen handler
    screen_handler.setFormatter(formatter)
    # add screenhandler to logger
    logger.addHandler(screen_handler)

    # Now do the same for the rotating logfile
    file_handler = RotatingFileHandler(il_logfile_path, maxBytes=il_size, backupCount=il_backups)
    file_handler.setLevel(numeric_level)
    formatter = logging.Formatter(fmt='%(asctime)s# %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # test messages
    # logger.debug('-- Logging inialized: showing Debug Level')
    # logger.info('-- Logging initialized: showing Info Level')
    # logger.warning('-- Logging inialized: showing Warning Level')
    # logger.error('-- Logging inialized: showing Error Level')
    # logger.critical('-- Logging inialized: showing Critical Level')
    return(logger)
# end function init log
