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
import os
from abc import abstractmethod, ABC
from contextlib import redirect_stdout, redirect_stderr

######## Abstract Base Class ########
class LogFunction(ABC):
    """An abstract base class logger

    A LogFunction must have the following methods defined:
        __call__ : a call function which performs the opperation of logging
        new_line: a function which adds a blank line to a logfile (or
            does what a given LogFunction should do when a new_line request
            is made)
        new: Return a new instance of the same kind of LogFunction with
            a fresh preamble
        spawn: Return a new instance of the same kind of LogFunction which
            concatenates the preamble given with the existing preamble
    """
    @abstractmethod
    def __call__(self, msg: str, *args, **kwargs) -> None:
        """Log a message"""
        raise NotImplementedError
    @abstractmethod
    def new_line(self):
        """Adds a blank line to a logfile 
        (or does what a given LogFunction should do when a new_line request
        is made)
        """
        raise NotImplementedError
    @abstractmethod
    def new(self, *args, **kwargs):
        """
        Return a new instance of the same kind of LogFunction with
            a fresh preamble
        """
        raise NotImplementedError
    @abstractmethod
    def spawn(self, *args, **kwargs):
        """
        spawn: Return a new instance of the same kind of LogFunction which
            concatenates the preamble given with the existing preamble
        """
        raise NotImplementedError

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
        """Print some text, and accept print arguments"""
        print(f"{self.preamble}{msg}", *args, **kwargs)

    def new_line(self):
        """Print a newline character"""
        print(os.linesep, end="")

    def new(self, preamble=""):
        """Return a new PrintLogFunction with a fresh preamble"""
        return self.__class__(preamble=preamble)
    def spawn(self, preamble=""):
        """Return a new PrintLogFunction with an extended preamble"""
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
    def catch_stderr(self):
        return self._catch_stderr

    # Secret print function
    def _print(self, msg, *args, **kwargs):
        """Redirect standard out and pass arguments ahead to print"""
        with open(self.filename, 'a') as File:
            if self.catch_stderr:
                with redirect_stdout(File) and redirect_stderr(File):
                    print(msg, *args, **kwargs)
            else:
                with redirect_stdout(File):
                    print(msg, *args, **kwargs)
        
    # Call function
    def __call__(self, msg, *args, **kwargs):
        """Output some text, and accept print arguments"""
        self._print(f"{self.preamble}{msg}", *args, **kwargs)

    def new_line(self):
        """Output a newline character"""
        self._print(os.linesep, end="")

    def new(self, preamble=""):
        """Return a new ContextLogFunction with a fresh preamble
            pointing at the same output file
        """
        return self.__class__(
            self.filename,
            preamble=preamble,
            catch_stderr=self.catch_stderr,
            safety=False, # By necessity
        )
    def spawn(self, preamble=""):
        """Return a new ContextLogFunction with an extended preamble
            pointing at the same output file
        """
        return self.new(self.preamble + preamble)
