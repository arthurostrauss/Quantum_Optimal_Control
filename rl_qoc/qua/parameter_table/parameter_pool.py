from __future__ import annotations

import itertools
from typing import Any, Dict, Optional, TYPE_CHECKING
import sys

if TYPE_CHECKING:
    from .parameter_table import ParameterTable
    from .parameter import Parameter


class ParameterPool:
    """
    A class to manage unique IDs for parameters.
    """

    _counter = itertools.count(1)
    _parameters_dict: Dict[int, ParameterTable | Parameter] = {}
    _patched = False
    _configured = False

    @classmethod
    def get_id(cls, obj: Any = None) -> int:
        """
        Get the next unique ID.

        Returns:
            int: The next unique ID.
            obj: The object associated with the ID.
        """
        next_id = next(cls._counter)
        if obj is not None:
            names = [param.name for param in cls._parameters_dict.values()]
            for param in cls._parameters_dict.values():
                if hasattr(param, "parameters"):
                    names.extend([p.name for p in param.parameters])
            if obj.name in names:
                raise ValueError(f"Parameter with name {obj.name} already exists.")
            if hasattr(obj, "parameters"):
                for obj_param in obj.parameters:
                    if obj_param.name in names:
                        raise ValueError(
                            f"Parameter with name {obj_param.name} already exists."
                        )
            cls._parameters_dict[next_id] = obj
        return next_id

    @classmethod
    def get_obj(cls, id: int) -> ParameterTable | Parameter:
        """
        Get the object associated with the given ID.

        Args:
            id (int): The ID of the object.

        Returns:
            obj: The object associated with the ID.
        """
        return cls._parameters_dict.get(id)

    @classmethod
    def reset(cls):
        """
        Reset the counter and the dictionary.
        """
        cls._counter = itertools.count(1)
        cls._parameters_dict = {}

    @classmethod
    def get_all_ids(cls):
        """
        Get all the IDs.

        Returns:
            List[int]: A list of all the IDs.
        """
        return list(cls._parameters_dict.keys())

    @classmethod
    def get_all_objs(cls) -> Any:
        """
        Get all the objects.

        Returns:
            Dict[int, Any]: A dictionary containing the IDs and the associated objects.
        """
        return list(cls._parameters_dict.values())

    @classmethod
    def get_all(cls) -> Dict[int, ParameterTable | Parameter]:
        """
        Get all the IDs and the associated objects.

        Returns:
            Dict[int, Any]: A dictionary containing the IDs and the associated objects.
        """
        return cls._parameters_dict

    def __getitem__(self, id: int) -> Any:
        """
        Get the object associated with the given ID.

        Args:
            id (int): The ID of the object.

        Returns:
            obj: The object associated with the ID.
        """
        return self.get_obj(id)

    def __setitem__(self, id: int, obj: Any):
        """
        Set the object associated with the given ID.

        Args:
            id (int): The ID of the object.
            obj: The object to be associated with the ID.
        """
        if id in self._parameters_dict:
            raise ValueError(f"Parameter with ID {id} already exists.")
        self._parameters_dict[id] = obj

    def __delitem__(self, id: int):
        """
        Delete the object associated with the given ID.

        Args:
            id (int): The ID of the object.
        """
        if id in self._parameters_dict:
            del self._parameters_dict[id]
        else:
            raise KeyError(f"Parameter with ID {id} does not exist.")

    def __contains__(self, id: int) -> bool:
        """
        Check if the object associated with the given ID exists.

        Args:
            id (int): The ID of the object.

        Returns:
            bool: True if the object exists, False otherwise.
        """
        return id in self._parameters_dict

    def __len__(self) -> int:
        """
        Get the number of objects in the pool.

        Returns:
            int: The number of objects in the pool.
        """
        return len(self._parameters_dict)

    def __iter__(self):
        """
        Iterate over the objects in the pool.

        Returns:
            Iterator: An iterator over the objects in the pool.
        """
        return iter(self._parameters_dict.values())

    def __str__(self) -> str:
        """
        Get a string representation of the pool.

        Returns:
            str: A string representation of the pool.
        """
        return str(self._parameters_dict)

    def __repr__(self) -> str:
        """
        Get a string representation of the pool.

        Returns:
            str: A string representation of the pool.
        """
        return f"ParameterPool({self._parameters_dict})"

    @classmethod
    def patch_opnic_wrapper(
        cls, path_to_opnic_dev: Optional[str] = "/home/dpoulos/opnic-dev"
    ):
        """
        Patch the OPNIC wrapper.

        Args:
            path_to_opnic_dev (Optional[str]): The path to the OPNIC development directory
        """
        from .opnic_utils import patch_opnic_wrapper

        param_tables = list(cls.get_all_objs())
        patch_opnic_wrapper(param_tables, path_to_opnic_dev)
        cls._patched = True

    @classmethod
    def configure_stream(
        cls,
        path_to_opnic_wrapper: Optional[
            str
        ] = "/home/dpoulos/aps_demo/python-wrapper/wrapper/build/python",
    ):
        # from opnic_python.opnic_wrapper import configure_stream
        # from opnic_python.opnic_wrapper import Direction_INCOMING, Direction_OUTGOING
        if "opnic_wrapper" not in sys.modules:
            sys.path.append(path_to_opnic_wrapper)
        from opnic_wrapper import (
            Direction_INCOMING,
            Direction_OUTGOING,
            configure_stream,
        )

        for obj in cls.get_all_objs():
            direction = (
                Direction_INCOMING
                if obj.direction == "INCOMING"
                else Direction_OUTGOING
            )
            configure_stream(obj.stream_id, direction)
        cls._configured = True

    @classmethod
    def initialize_streams(
        cls, path_to_opnic_dev: Optional[str] = "/home/dpoulos/opnic-dev"
    ):
        """
        Initialize the OPNIC and the necessary streams for the current stage of the ParameterPool.
        Args:
            path_to_opnic_dev:

        Returns:

        """
        cls.patch_opnic_wrapper(path_to_opnic_dev)
        cls.configure_stream()

    @classmethod
    @property
    def patched(cls) -> bool:
        return cls._patched

    @classmethod
    @property
    def configured(cls) -> bool:
        return cls._configured

    @classmethod
    def close_streams(cls):
        if cls._configured and cls._patched:
            if "opnic_wrapper" not in sys.modules:
                sys.path.append(
                    "/home/dpoulos/aps_demo/python-wrapper/wrapper/build/python"
                )
            from opnic_wrapper import close_stream

            for obj in cls.get_all_objs():
                close_stream(obj.stream_id)
            cls._configured = False
            cls._patched = False
