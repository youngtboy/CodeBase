
class Compose(object):

    def __init__(self, transforms):
        self.transforms = []
        for transform in transforms:
            if callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable')

    def __call__(self, data):

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data