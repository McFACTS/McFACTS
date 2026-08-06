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
from contextlib import redirect_stdout, redirect_stderr
from abc import abstractmethod, ABC

######## Abstract Base Class ########
class LogFunction(ABC):
    """An abstract base class logger"""
    @abstractmethod
    def __call__(self, msg: str, *args, **kwargs) -> None:
        """Log a message"""
        return
    @abstractmethod
    def new(self, *args, **kwargs):
        return
    @abstractmethod
    def spawn(self, *args, **kwargs):
        return

######## Class instances ########
#### Print Log Function ####
class PrintLogFunction(LogFunction):
    """A Log function which calls print"""
    def __init__(self, preamble=""):
        self._preamble = preamble

    @property
    def preamble(self):
        return self._preamble

    def __call__(self, msg, *args, **kwargs):
        print(f"{self.preamble}{msg}", *args, **kwargs)

    def new(self, preamble=""):
        return self.__class__(preamble=preamble)
    def spawn(self, preamble=""):
        return self.__class__(preamble=self.preamble + preamble)

#### Print Log Function ####
class ContextLogFunction(LogFunction):
    def __init__(
            self, 
            filename, 
            preamble="",
            catch_stderr=False, 
            safety=False,
        ):
        self._filename = filename
        # Check that the file doesn't already exist at ini
        if safety:
            with open(self.filename, 'w-') as File:
                pass
        self._preamble = preamble
        self._catch_stderr = catch_stderr

    @property
    def filename(self):
        return self._filename
        
    @property
    def preamble(self):
        return self._preamble

    @property
    def cache_stderr(self):
        return self._catch_stderr
        
    # Call function
    def __call__(self, msg, *args, **kwargs):
        with open(self.filename, 'a') as File:
            if self.cache_stderr:
                with redirect_stdout(File) and redirect_stderr(File):
                    print(f"{self.preamble}{msg}", *args, **kwargs)
            else:
                with redirect_stdout(File):
                    print(f"{self.preamble}{msg}", *args, **kwargs)

    def new(self, preamble=""):
        return self.__class__(
            self.filename,
            preamble=preamble,
            catch_stderr=self.catch_stderr,
            safety=False, # By necessity
        )
    def spawn(self, preamble=""):
        return self.new(self.preamble + preamble)
