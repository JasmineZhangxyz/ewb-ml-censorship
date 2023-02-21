import collections

class Strings:
    def __init__(self, strings, name=None):
        self.name = name
        self.list = list(strings)
        self.set = set(self.list)
        self.counter = collections.Counter(self.list)
    def __len__(self):
        return len(self.list)
