import logging
from datetime import datetime

class Logger:
    def __init__(self, log_file : str = 'log'):
        self.logger = logging.getLogger(__name__)

        # Set the logging level
        level = logging.DEBUG
        self.logger.setLevel(level)

        # Create a file handler which logs even debug messages
        filename = f'logs/{log_file}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
        fh = logging.FileHandler(filename)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(fh)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

# Example usage
if __name__ == "__main__":
    # Set up the logger with a specified log file name
    logger = Logger('app')

    # Test logging
    logger.info('This is an info message')
    logger.debug('This is a debug message')
    logger.error('This is an error message')
