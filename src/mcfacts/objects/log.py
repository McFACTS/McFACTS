"""
This module defines a protocol for logging functions.

The `LogFunction` protocol can be used to type-check callable objects or functions
that are intended to log messages, optionally appending a new line.
"""

from typing import Protocol


class LogFunction(Protocol):
    """
    A protocol that represents a logging function.

    Any callable that matches this protocol should accept a message as a string
    and an optional `new_line` flag (defaulting to True), and return None.
    """

    def __call__(self, msg: str, new_line: bool = True, /) -> None:
        """
        Log a message, optionally appending a new line.

        Args:
            msg (str): The message to be logged.
            new_line (bool, optional): Whether to append a newline after the message.
                Defaults to True.
        """
        ...
