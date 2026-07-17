"""read-only Cached property decorator"""
######## Imports ########
#### Standard Library ####
from functools import cached_property as cached_property

######## cached_property ########
class read_only_cached_property(cached_property):
    """Custom readonly cached property"""
    def __init__(self, func):
        super().__init__(func)
        self.is_set = True

    def __set__(self, instance, value):
        raise AttributeError("Property is readonly")

