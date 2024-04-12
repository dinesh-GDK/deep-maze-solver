import logging


def get_logger():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(log_file_path)
        ],
    )

    logger = logging.getLogger()
    return logger
