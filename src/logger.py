import logging

class LoggerConsole:
   def configure_logger(logLevel=logging.DEBUG):
        logger = logging.getLogger(__name__)
        logger.setLevel(logLevel)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s :'
            ' %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
        logger.propagate = False
        
        return logger

logger = LoggerConsole.configure_logger()