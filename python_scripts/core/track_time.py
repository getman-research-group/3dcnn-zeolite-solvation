# -*- coding: utf-8 -*-
"""
track_time.py
The purpose of this script is to keep track of time

Functions:
    convert_sec_to_hms: converts seconds to hours, minutes, and seconds
    print_total_time: prints total time a process took, given the start time
"""
import time

### Function To Keep Track Of Time
def convert_sec_to_hms( seconds ):
    '''
    This function simply takes the total seconds and converts it to hours, minutes, and seconds
    INPUTS:
        seconds: Total seconds
    OUTPUTS:
        h: hours
        m: minutes
        s: seconds
    '''
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

### Function To Print Time Given Initial Time
def print_total_time(start_time, string = 'Total time: '):
    '''
    The purpose of this function is to print the total time given a start time
    INPUTS:
        start_time: time from time module, e.g.
            start_time = time.time()
        string: [str] string in front of the total time
    OUTPUTS:
        prints total time
        total_time: total time that was taken in seconds
    '''
    ## Finding Total Time
    total_time = time.time() - start_time
    ## Converting To Hours, Minutes, And Seconds
    h, m, s = convert_sec_to_hms(total_time)
    ## Printing
    print("%s %d hrs, %d mins, %d sec"%(string, h, m, s) )
    return total_time