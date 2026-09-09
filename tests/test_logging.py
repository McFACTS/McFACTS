#!/usr/bin/env python3
"""Test the log function
"""
######## Imports ########
import tempfile
import os
from contextlib import redirect_stdout, redirect_stderr
#### Standard library ####
#### Local ####
from mcfacts import fiducial_plots, simulation
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.log import LogFunction
from mcfacts.objects.log import PrintLogFunction, ContextLogFunction

######## Tests ########

def test_print_log_function():
    # Define some data
    prefix_1 = "(Alpha) :: "
    prefix_2 = "(Bravo) :: "
    msg1 = "AAAA"
    msg2 = "BBBB"
    msg3 = "CCCC"
    msg4 = "DDDD"
    # Instantiate print_log_function
    print_log = PrintLogFunction(prefix=prefix_1)
    # Open a temporary directory
    with tempfile.TemporaryDirectory() as wkdir:
        # logfile 
        logfile = os.path.join(wkdir, "print.log")
        # Redirect standard out to a file
        with open(logfile, 'w') as F:
          with redirect_stdout(F):
            # Print something
            print_log(msg1)
            # Skip a line
            print_log.new_line()
            # Generate a new one with no prefix
            print_new = print_log.new()
            # Print something
            print_new(msg2)
            # Generate a new one with a different prefix
            print_bravo = print_log.new(prefix=prefix_2)
            # Print something
            print_bravo(msg3)
            # Spawn from print_log
            print_concat_prefix = print_log.spawn(prefix=prefix_2)
            # Print something
            print_concat_prefix(msg4)

        # Read the log and check the lines
        with open(logfile, "r") as F:
            for i, line in enumerate(F):
                # Remove exactly one newline character from the end of the string
                cleaned = line.removesuffix(os.linesep)
                # Check lines 
                # (I know there's probably a better way, but I don't care).
                if i == 0:
                    assert cleaned == prefix_1 + msg1
                elif i == 1:
                    assert cleaned == ""
                elif i == 2:
                    assert cleaned == msg2
                elif i == 3:
                    assert cleaned == prefix_2 + msg3
                elif i == 4:
                    assert cleaned == prefix_1 + prefix_2 + msg4
        # Pass!

def test_context_log_function():
    # Define some data
    prefix_1 = "(Alpha) :: "
    prefix_2 = "(Bravo) :: "
    msg1 = "AAAA"
    msg2 = "BBBB"
    msg3 = "CCCC"
    msg4 = "DDDD"
    # Open a temporary directory
    with tempfile.TemporaryDirectory() as wkdir:
        # logfile 
        logfile = os.path.join(wkdir, "context.log")
        # Instantiate context_log_function
        context_log = ContextLogFunction(
            logfile,
            prefix=prefix_1,
        )
        # Print something
        context_log(msg1)
        # Skip a line
        context_log.new_line()
        # Generate a new one with no prefix
        context_new = context_log.new()
        # Print something
        context_new(msg2)
        # Generate a new one with a different prefix
        context_bravo = context_log.new(prefix=prefix_2)
        # Print something
        context_bravo(msg3)
        # Spawn from context_log
        context_concat_prefix = context_log.spawn(prefix=prefix_2)
        # Print something
        context_concat_prefix(msg4)

        # Read the log and check the lines
        with open(logfile, "r") as F:
            for i, line in enumerate(F):
                # Remove exactly one newline character from the end of the string
                cleaned = line.removesuffix(os.linesep)
                # Check lines 
                # (I know there's probably a better way, but I don't care).
                if i == 0:
                    assert cleaned == prefix_1 + msg1
                elif i == 1:
                    assert cleaned == ""
                elif i == 2:
                    assert cleaned == msg2
                elif i == 3:
                    assert cleaned == prefix_2 + msg3
                elif i == 4:
                    assert cleaned == prefix_1 + prefix_2 + msg4

######## Main ########
def main():
    test_print_log_function()
    test_context_log_function()
    pass

######## Execution ########
if __name__ == "__main__":
    main()
