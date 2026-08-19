class CUDAGraphWrapper:
    def __init__(self, runnable):
        self.runnable = runnable

    def __getattr__(self, key):
        return getattr(self.runnable, key)


class Root:
    def __init__(self):
        self.x = 1


wrapped = CUDAGraphWrapper(Root())
print("Has runnable:", getattr(wrapped, "runnable", None) is not None)
print("Has unwrap:", getattr(wrapped, "unwrap", None) is not None)
