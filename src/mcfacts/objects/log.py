""" This module defines a protocol for logging functions.
"""
######## Imports ########

#from typing import Protocol
# Vera: I don't think we really want to use a Protocol
# Python is not a strongly typed language, and our interface with mcfacts
#  does not enforce typing hints.
# This would make sense in rust, as a trait. It doesn't make sense in Python.
# Not the way we are using it.
# It makes hard type checking through isinstance more painful.

#### Standard Library Imports ####
from contextlib import redirect_stdout, redirect_std
from abc import abstractmethod, ABC

######## Abstract Base Class ########
class LogFunction(ABC):
    """An abstract base class logger"""
    @abstractmethod
    def __call__(self, msg: str, *args, **kwargs) -> None:
        """Log a message"""
        return

######## Class instances ########
#### Print Log Function ####
class PrintLogFunction(LogFunction):
    """A Log function which calls print

    Note
    ----
    Without a defined __init__ method, PrintLogFunction inherits
        object.__init__, which no-ops and accepts no arguments.
    """
    def __call__(self, *args, **kwargs):
        print(*args, **kwargs)

#### Print Log Function ####
class ContextLogFunction(LogFunction):
    def __init__(self, filename, catch_stderr=False, safety=False):
        self._filename = filename
        # Check that the file doesn't already exist at ini
        if safety:
            with open(self.filename, 'w-') as File:
                pass
        self._catch_stderr = catch_stderr

    @property
    def filename(self):
        return self._filename
        
    @property
    def cache_stderr(self):
        return self._catch_stderr
        
    # Call function
    def __call__(self, *args, **kwargs):
        with open(self.filename, 'a') as File:
            if self.cache_stderr:
                with redirect_stdout(File) and redirect_stderr(File):
                    print(*args, **kwargs)
            else:
                with redirect_stdout(File):
                    print(*args, **kwargs)
