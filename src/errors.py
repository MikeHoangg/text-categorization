"""
Module for custom errors
"""


class BrokenConfigException(Exception):
    def __init__(self, ex: str):
        super().__init__(f"Config file error. {ex}")


class MissingArgException(BrokenConfigException):
    def __init__(self, pipeline: str, arg: str):
        super().__init__(f"Missing {arg} argument for {pipeline} pipeline.")
