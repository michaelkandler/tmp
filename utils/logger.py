import logging
import datetime
import os.path


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Ensure only one instance of the class is created.

        Returns
        -------
        object
            The class instance.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class CustomLogger(metaclass=Singleton):
    """
    Singleton logger class with customizable logging capabilities.

    This class provides a logging mechanism with options for both file and console output.
    It allows setting different log levels and formatting.

    Attributes
    ----------
    file_name : str
        The name of the log file.
    log_path : str
        The path to the log directory.
    logger : logging.Logger
        The logger instance.
    file_handler : logging.FileHandler
        The file handler for log file output.
    console_handler : logging.StreamHandler
        The console handler for log console output.
    formatter_file : logging.Formatter
        The formatter for log file messages.

    Methods
    -------
    debug(message)
        Log a message with DEBUG level.
    info(message)
        Log a message with INFO level.
    warning(message)
        Log a message with WARNING level.
    error(message)
        Log a message with ERROR level.
    critical(message)
        Log a message with CRITICAL level.
    exception(message)
        Log a message with ERROR level and exception info.
    change_file_handler()
        Change the file handler to a new log file.
    """

    def __init__(self):
        self.file_name = f"{self._get_current_time()}.log"

        self.log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.file_handler = logging.FileHandler(os.path.join(self.log_path, self.file_name))

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)

        self.formatter_file = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        self.console_handler.setFormatter(CustomFormatter())
        self.file_handler.setFormatter(self.formatter_file)

        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)

    def debug(self, message: str) -> str:
        """
        Log a message with DEBUG level.

        Parameters
        ----------
        message : str
            The message to be logged.
        """
        message = self.remove_line_breaks(message)
        self.logger.debug(message)

    def info(self, message: str) -> str:
        """
        Log a message with INFO level.

        Parameters
        ----------
        message : str
            The message to be logged.
        """
        message = self.remove_line_breaks(message)
        self.logger.info(message)

    def warning(self, message: str) -> str:
        """
        Log a message with WARNING level.

        Parameters
        ----------
        message : str
            The message to be logged.
        """
        message = self.remove_line_breaks(message)
        self.logger.warning(message)

    def error(self, message: str) -> str:
        """
        Log a message with ERROR level.

        Parameters
        ----------
        message : str
            The message to be logged.
        """
        message = self.remove_line_breaks(message)
        self.logger.error(message)

    def critical(self, message: str) -> str:
        """
        Log a message with CRITICAL level.

        Parameters
        ----------
        message : str
            The message to be logged.
        """
        message = self.remove_line_breaks(message)
        self.logger.critical(message)

    def exception(self, message: str) -> str:
        """
        Log a message with ERROR level and exception info.

        Parameters
        ----------
        message : str
            The message to be logged.
        """
        message = self.remove_line_breaks(message)
        self.logger.exception(message)

    def change_file_handler(self):
        """
        Change the file handler to a new log file.

        Used to create new name and logger files
        """
        # get new file name
        self.file_name = f"{self._get_current_time()}.log"

        # remove old logger
        self.logger.handlers[-1].close()
        self.logger.removeHandler(self.logger.handlers[-1])

        # create and configure new filehandler
        self.file_handler = logging.FileHandler(os.path.join(self.log_path, self.file_name))

        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(self.formatter_file)

        # add new handler to logger
        self.logger.addHandler(self.file_handler)

    @staticmethod
    def _get_current_time():
        """
        Get the current time in a formatted string.

        Returns
        -------
        str
            The formatted current time string.
        """
        return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    @staticmethod
    def remove_line_breaks(input_string: str) -> str:
        """
        Remove all line breaks from a given string.

        Parameters
        ----------
        input_string : str
            The input string containing line breaks.

        Returns
        -------
        str
            The input string with all line breaks removed.
        """
        return input_string.replace('\n', ' ').replace('\r', ' ')


class CustomFormatter(logging.Formatter):
    """
    Custom formatter for log messages.

    This class defines a custom log message formatter with colored output
    based on the log level.

    Attributes
    ----------
    grey : str
        ANSI escape sequence for grey text color.
    blue : str
        ANSI escape sequence for blue text color.
    yellow : str
        ANSI escape sequence for yellow text color.
    red : str
        ANSI escape sequence for red text color.
    bold_red : str
        ANSI escape sequence for bold red text color.
    reset : str
        ANSI escape sequence to reset text color.
    fmt_console : str
        The format string for console output.

    Methods
    -------
    format(record)
        Format the log record message based on the log level.

    """
    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    fmt_console = '%(message)s'
    FORMATS_console = {
        logging.DEBUG: grey + fmt_console + reset,
        logging.INFO: blue + fmt_console + reset,
        logging.WARNING: yellow + fmt_console + reset,
        logging.ERROR: red + fmt_console + reset,
        logging.CRITICAL: bold_red + fmt_console + reset
    }

    def format(self, record: str) -> str:
        """
        Format the log record message based on the log level.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be formatted.

        Returns
        -------
        str
            The formatted log message.
        """
        log_fmt = self.FORMATS_console.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
