try:
    from .adetailer import ADetailerCompositeInvocation, ADetailerInvocation, ADetailerMaskInvocation
except ModuleNotFoundError as error:
    if error.name != "invokeai":
        raise
    ADetailerInvocation = None  # type: ignore[assignment]
    ADetailerMaskInvocation = None  # type: ignore[assignment]
    ADetailerCompositeInvocation = None  # type: ignore[assignment]

__all__ = ["ADetailerInvocation", "ADetailerMaskInvocation", "ADetailerCompositeInvocation"]
