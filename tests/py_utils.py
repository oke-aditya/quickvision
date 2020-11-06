
__all__ = ["is_iterable", "cycle_over"]


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def cycle_over(objs):
    for idx, obj in enumerate(objs):
        yield obj, objs[:idx] + objs[idx + 1:]
