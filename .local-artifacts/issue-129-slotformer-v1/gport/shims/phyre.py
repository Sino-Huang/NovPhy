"""b127 shim: the PHYRE package (py3.6 wheels only) is imported unconditionally by SlotFormer's dataset
package; no PHYRE code path runs in the CLEVRER evaluation or the G-port. Any attribute access raises."""


def __getattr__(name):
    raise ImportError(f"phyre is not installed in b127-slotformer (phyre.{name} requested)")
