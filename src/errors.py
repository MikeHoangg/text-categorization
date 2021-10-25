class BrokenConfigException(Exception):
    def __init__(self, ex: str):
        super().__init__(f"Config file error. {ex}")
