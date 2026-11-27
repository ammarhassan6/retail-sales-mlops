"""Feast ENTITIES = the "keys" we look features up by.

Our features belong to a store-item pair, so we have two entities.
A lookup looks like: {"store": 1, "item": 17}
"""

from feast import Entity, ValueType

store = Entity(
    name="store",
    join_keys=["store"],
    value_type=ValueType.INT64,
    description="Store id (1-10 in the Kaggle dataset)",
)

item = Entity(
    name="item",
    join_keys=["item"],
    value_type=ValueType.INT64,
    description="Item / product id (1-50 in the Kaggle dataset)",
)
