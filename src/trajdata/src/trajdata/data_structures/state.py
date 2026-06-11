# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""
Defines subclasses of np.array and torch.Tensor which give
property access to different state elements and allow for easy conversion
between types to help make code that works with state elements more readable
and more robust to future changes in state format (e.g. adding additional dimensions)

Currently, these subclasses are designed to be lightweight and ephemeral:
any np/torch operation on a State subclass will drop the format metadata.
TODO: we could make this more robust by making exceptions for operations which
preserve the semantic meanings of the elements.
TODO: implement setters for all properties
"""
from abc import abstractclassmethod
from collections import defaultdict
from typing import Callable, ClassVar, Dict, List, Set, Type, TypeVar

import numpy as np
import torch
from torch import Tensor

STATE_ELEMS_REQUIREMENTS = {
    "x": None,  # x position in world frame (m)
    "y": None,  # y position in world frame (m)
    "z": None,  # z position in world frame (m)
    "xd": ("x_component", "v_lon", "v_lat", "c", "s"),  # x vel in world frame (m/s)
    "yd": ("y_component", "v_lon", "v_lat", "c", "s"),  # y vel in world frame (m/s)
    "zd": None,  # z velocity in world frame (m/s)
    "xdd": None,  # x acceleration in world frame (m/s^2)
    "ydd": None,  # y acceleration in world frame (m/s^2)
    "zdd": None,  # z acceleration in world frame (m/s^2)
    "h": ("arctan", "s", "c"),  # heading (rad)
    "dh": None,  # heading rate (rad)
    "c": ("cos", "h"),  # cos(h)
    "s": ("sin", "h"),  # sin(h)
    "v_lon": ("lon_component", "xd", "yd", "c", "s"),  # longitudinal velocity
    "v_lat": ("lat_component", "xd", "yd", "c", "s"),  # latitudinal velocity
}

# How many levels deep we'll try to check if requirements for certain attributes
# themselves need to be computed and are not directly available
MAX_RECURSION_LEVELS = 2


Array = TypeVar("Array", np.ndarray, torch.Tensor)


def lon_component(x, y, c, s):
    """
    Returns magnitude of x,y that is parallel
    to unit vector c,s
    """
    return x * c + y * s


def lat_component(x, y, c, s):
    """
    Returns magnitude of x,y that is orthogonal to
    unit vector c,s (i.e., parallel to -s,c)
    """
    return -x * s + y * c


def x_component(long, lat, c, s):
    """
    Returns x component given long and lat components
    and cos and sin of heading
    """
    return long * c - lat * s


def y_component(long, lat, c, s):
    """
    Returns y component given long and lat components
    and cos and sin of heading
    """
    return long * s + lat * c


def _check_format_length_compatible_with_array_dim(array, format):
    if array.shape[-1] != len(format.split(",")):
        raise ValueError(
            f"Array shape {array.shape} incompatible with format {format}: "
            f"Array has {array.shape[-1]} entries but format has length "
            f"{len(format.split(','))}."
        )


class State:
    """
    Base class implementing property access to state elements
    Needs to be subclassed for concrete underlying datatypes, e.g.
    torch.Tensor vs np.ndarray, to equip self object with
    indexing support
    """

    _format: str = ""

    # set upon subclass init
    state_dim: int = 0

    # needs to be defined in subclass
    _FUNCS: ClassVar[Dict[str, Callable]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # use subclass _format string to initialize class specific _format_dict
        cls._format_dict: Dict[str, int] = {}
        for j, attr in enumerate(cls._format.split(",")):
            cls._format_dict[attr] = j

        # intialize properties
        cls.position = cls._init_property("x,y")
        cls.position3d = cls._init_property("x,y,z")
        cls.velocity = cls._init_property("xd,yd")
        cls.acceleration = cls._init_property("xdd,ydd")
        cls.heading = cls._init_property("h")
        cls.heading_vector = cls._init_property("c,s")

        # initialize state_dim
        cls.state_dim = len(cls._format_dict)

    @abstractclassmethod
    def from_array(cls, array: Array, format: str) -> "State":
        """
        Returns State instance given Array with correct format.

        Args:
            array (Array): Array
            format (str): format string
        """
        raise NotImplementedError

    @abstractclassmethod
    def _combine(cls, arrays: List[Array]):
        """
        Concatenates arrays along last dimension, and returns result
        according to new format string

        Args:
            arrays (List[Array]): _description_
            format (str): _description_
        """
        raise NotImplementedError

    def as_format(self, new_format: str, create_type=True):
        """
        Returns a new StateTensor with the specified format,
        constructed using data in the current format
        """
        requested_attrs = new_format.split(",")
        components = []  # contains either indicies into self, or attrs
        index_list = None
        for j, attr in enumerate(requested_attrs):
            if attr in self._format_dict:
                if index_list is None:
                    # start a new block of indices
                    index_list = []
                    components.append(index_list)
                index_list.append(self._format_dict[attr])
            else:
                if index_list is not None:
                    # if we had been pulling indices, stop
                    index_list = None
                components.append(attr)
        # assemble
        arrays = []
        for component in components:
            if isinstance(component, list):
                arrays.append(self[..., component])
            elif isinstance(component, str):
                arrays.append(self._compute_attr(component)[..., None])
            else:
                raise ValueError

        result = self._combine(arrays)
        if create_type:
            return self.from_array(result, new_format)
        else:
            return result

    def has_attr(self, attr: str) -> bool:
        try:
            # Check whether we can compute attr without raising ValueError.
            getattr(self, attr)
            return True
        except ValueError:
            return False

    def has_attr(self, attr: str) -> bool:
        try:
            # Check whether we can compute attr without raising ValueError.
            getattr(self, attr)
            return True
        except ValueError:
            return False

    def _compute_attr(self, attr: str, _depth: int = MAX_RECURSION_LEVELS):
        """
        Tries to compute attr that isn't directly part of the tensor
        given the information available.

        if a requirement for the attr isn't directly part of the tensor
        either, then we recurse to compute that attribute.
        _depth controls the depth of recursion

        If impossible raises ValueError
        """
        if _depth == 0:
            raise RecursionError
        try:
            formula = STATE_ELEMS_REQUIREMENTS[attr]
            if formula is None:
                raise KeyError(f"No formula for {attr}")
            func_name, *requirements = formula
            func = self._FUNCS[func_name]
            args = [self.get_attr(req, _depth=_depth - 1) for req in requirements]
        except KeyError as ke:
            raise ValueError(
                f"{attr} cannot be computed from available data at the current timestep."
            )
        except RecursionError as re:
            raise ValueError(
                f"{attr} cannot be computed: Recursion depth exceeded when trying to computerequirements"
            )
        return func(*args)

    def get_attr(self, attr: str, _depth: int = MAX_RECURSION_LEVELS):
        """
        Returns slice of tensor corresponding to attr

        """
        if attr in self._format_dict:
            return self[..., self._format_dict[attr]]
        else:
            return self._compute_attr(attr, _depth=_depth)

    def set_attr(self, attr: str, val: Tensor):
        if attr in self._format_dict:
            self[..., self._format_dict[attr]] = val
        else:
            raise ValueError(f"{attr} not part of State")

    @classmethod
    def _init_property(cls, format: str) -> property:
        split_format = format.split(",")
        try:
            index_list = tuple(cls._format_dict[attr] for attr in split_format)

            def getter(self: State) -> Array:
                return self[..., index_list]

            def setter(self: State, val: Array) -> None:
                self[..., index_list] = val

        except KeyError:
            # getter is nontrivial, let as_format handle the logic
            def getter(self: State) -> Array:
                return self.as_format(format, create_type=False)

            # can't set this property since not all elements are part of format
            setter = None

        return property(
            getter,
            setter,
            doc=f"""
            Returns:
                Array: shape [..., {len(split_format)}] corresponding to {split_format}.
            """,
        )


class StateArray(State, np.ndarray):
    _FUNCS = {
        "cos": np.cos,
        "sin": np.sin,
        "arctan": np.arctan2,
        "lon_component": lon_component,
        "lat_component": lat_component,
        "x_component": x_component,
        "y_component": y_component,
    }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({super().__str__()})"

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        args = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, type(self)):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.get("out", None)
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, type(self)):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs["out"] = tuple(out_args)
        else:
            outputs = (None,) * function.nout

        # call original function
        results = super().__array_ufunc__(function, method, *args, **kwargs)

        return results

    def __getitem__(self, key) -> np.ndarray:
        """
        StateArray[key] always returns an np.ndarray, as we can't
        be sure that key isn't indexing into the state elements
        without adding logic in python which adds significant overhead
        to the base numpy implementation which is in C.

        In cases where we just want to index batch dimensions, use
        StateArray.at(key). We add logic for slice indexing
        """
        return_type = np.ndarray
        if isinstance(key, (int, slice)) and self.ndim > 1:
            return_type = type(self)
        return super().__getitem__(key).view(return_type)

    def at(self, key) -> "StateArray":
        """
        Equivalent to self[key], but assumes (without checking!)
        that key selects only batch dimensions, so return type
        is the same as type(self)
        """
        return super().__getitem__(key)

    def as_ndarray(self) -> np.ndarray:
        """Convenience function to convert to default ndarray type
        Applying np operations to StateArrays can silently convert them
        to basic np.ndarrays, so making this conversion explicit
        can improve code readability.

        Returns:
            np.ndarray: pointing to same data as self
        """
        return self.view(np.ndarray)

    @classmethod
    def from_array(cls, array: Array, format: str):
        _check_format_length_compatible_with_array_dim(array, format)
        return array.view(NP_STATE_TYPES[format])

    @classmethod
    def _combine(cls, arrays: List[Array]):
        """
        Concatenates arrays along last dimension, and returns result
        according to new format string
        """
        return np.concatenate(arrays, axis=-1)


class StateTensor(State, Tensor):
    """
    Convenience class which offers property access to state elements
    Standardizes order of state dimensions
    """

    _FUNCS = {
        "cos": torch.cos,
        "sin": torch.sin,
        "arctan": torch.atan2,
        "lon_component": lon_component,
        "lat_component": lat_component,
        "x_component": x_component,
        "y_component": y_component,
    }

    CAPTURED_FUNCS: Set[Callable] = {
        Tensor.cpu,
        Tensor.cuda,
        Tensor.add,
        Tensor.add_,
        Tensor.__deepcopy__,
    }

    @classmethod
    def new_empty(cls, *args, **kwargs):
        return torch.empty(*args, **kwargs).as_subclass(cls)

    def clone(self, *args, **kwargs):
        return super().clone(*args, **kwargs).as_subclass(type(self))

    def to(self, *args, **kwargs):
        new_obj = self.__class__()
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = tempTensor.requires_grad
        return new_obj

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # overriding this to ensure operations yield base Tensors
        if kwargs is None:
            kwargs = {}

        new_class = Tensor
        result = super().__torch_function__(func, types, args, kwargs)

        if func in StateTensor.CAPTURED_FUNCS:
            new_class = cls

        if func == Tensor.__getitem__:
            self = args[0]
            indices = args[1]
            if isinstance(indices, int):
                if self.ndim > 1:
                    new_class = cls
            elif isinstance(indices, slice):
                if self.ndim > 1:
                    new_class = cls
                elif indices == slice(None):
                    new_class = cls
            elif isinstance(indices, tuple):
                if len(indices) < self.ndim and Ellipsis not in indices:
                    new_class = cls
                elif len(indices) == self.ndim and indices[-1] == slice(None):
                    new_class = cls

        if func in (torch.reshape, Tensor.reshape):
            original_state_size = args[0].shape[-1]
            if len(args) > 1:
                # shape passed in as positional argument
                # It can be either a separate tuple or a series of ints
                shape = args[1] if isinstance(args[1], tuple) else args[1:]
            else:
                shape = kwargs.get("shape")
            if shape[-1] == original_state_size:
                new_class = cls

        if isinstance(result, Tensor) and new_class != cls:
            result = result.as_subclass(new_class)

        if func == Tensor.numpy:
            result: np.ndarray
            result = result.view(NP_STATE_TYPES[cls._format])

        return result

    def as_tensor(self) -> Tensor:
        """Convenience function to convert to default tensor type
        Applying torch operations to StateTensors can silently convert them
        to basic torch.Tensors, so making this conversion explicit
        can improve code readability.

        Returns:
            Tensor: pointing to same data as self
        """
        return self.as_subclass(Tensor)

    @classmethod
    def from_numpy(cls, state: StateArray, **kwargs):
        return torch.from_numpy(state, **kwargs).as_subclass(
            TORCH_STATE_TYPES[state._format]
        )

    @classmethod
    def from_array(cls, array: Array, format: str):
        _check_format_length_compatible_with_array_dim(array, format)
        return array.as_subclass(TORCH_STATE_TYPES[format])

    @classmethod
    def _combine(cls, arrays: List[Array]):
        """
        Concatenates arrays along last dimension, and returns result
        according to new format string
        """
        return torch.cat(arrays, dim=-1)


def createStateType(format: str, base: Type[State]) -> Type[State]:
    name = base.__name__ + "".join(map(str.capitalize, format.split(",")))
    cls = type(
        name,
        (base,),
        {
            "_format": format,
        },
    )
    # This is needed so that these dynamically created classes are understood
    # by pickle, which is used in multiprocessing, e.g. in a dataset
    globals()[name] = cls
    return cls


class StateTypeFactory(defaultdict):
    def __init__(self, base: Type[State]):
        self.base_type = base

    def __missing__(self, format: str) -> Type[State]:
        self[format] = createStateType(format, self.base_type)
        return self[format]


# DEFINE STATE TYPES
TORCH_STATE_TYPES: Dict[str, Type[StateTensor]] = StateTypeFactory(StateTensor)
NP_STATE_TYPES: Dict[str, Type[StateArray]] = StateTypeFactory(StateArray)
