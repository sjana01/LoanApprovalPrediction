# writing a custom exception
import sys
from src.logger import logging

def error_message_detail(error, error_deltail:sys):
    _,_,exc_tb = error_deltail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = "Error occured in line number [{0}] of file [{1}]. Error message: [{2}]".format(line_number,file_name,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_deltail=error_detail)

    def __str__(self):
        return self.error_message
    

