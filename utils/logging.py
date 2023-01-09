import logging


def setup_logger(save_log_path):

    logger = logging.getLogger('main_logger')
    # create console handler and set level to info
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # create file handler and set level to info
    file_handler = logging.FileHandler(filename=save_log_path)
    file_handler.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatters to our handlers
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # add Handlers to our logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
