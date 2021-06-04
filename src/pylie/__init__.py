from .solve import solve, _MANIFOLDS, _METHODS


def manifolds():
    for key, manifold_class in _MANIFOLDS.items():
        output_string = f'"{key}"'
        manifold_instance = manifold_class()
        if hasattr(manifold_instance, "description"):
            output_string += f":\t{manifold_instance.description}"
        print(output_string)


def methods():
    # Get a temp manifold to instantiate the methods
    temp_manifold = _MANIFOLDS["hmnsphere"]
    temp_manifold.exp = None
    temp_manifold.dexpinv = None
    temp_manifold.action = None
    for key, method in _METHODS.items():
        method_instance = method(temp_manifold)
        output_string = f'"{key}":\t'
        output_string += (
            f"{method_instance.s} stage method of order {method_instance.order}"
        )
        print(output_string)


__all__ = ["solve", "manifolds", "methods"]
