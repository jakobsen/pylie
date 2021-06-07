from .solve import solve, _MANIFOLDS, _METHODS


def manifolds():
    for key, manifold_class in _MANIFOLDS.items():
        output_string = f'"{key}"'
        output_string += f":\t{manifold_class.__doc__}"
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
        if hasattr(method_instance, "s") and hasattr(method_instance, "order"):
            output_string += (
                f"{method_instance.s} stage method of order {method_instance.order}"
            )
        elif hasattr(method_instance, "__doc__"):
            output_string += method_instance.description
        print(output_string)


__all__ = ["solve", "manifolds", "methods"]
