"""
This module contains some helper function, which are **not** physics related.

For utility functions related to physics and math, see ``qftlib.utils``.
"""


from __future__ import annotations


def _alias_attribute(field_name: str) -> property:
    """
    This function takes the attribute name of field to make a alias and return
    a property that work to get and set.
    """
    field = property(lambda self: getattr(self, field_name))
    field = field.setter(lambda self, value: setattr(self, field_name, value))
    return field
