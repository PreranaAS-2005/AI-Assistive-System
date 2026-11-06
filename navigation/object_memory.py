import time


class ObjectMemory:

    def __init__(self):
        self.memory = {}

    def update(
        self,
        label,
        direction,
        distance,
        confidence,
        bbox
    ):
        """
        Store the latest detection of an object.
        """

        self.memory[label] = {
            "label": label,
            "direction": direction,
            "distance": distance,
            "confidence": confidence,
            "bbox": bbox,
            "timestamp": time.time()
        }

    def get_last_seen(self, label):
        """
        Return the latest memory record for the object.
        Returns None if the object was never detected.
        """

        return self.memory.get(label)

    def has_object(self, label):
        """
        Check whether an object exists in memory.
        """

        return label in self.memory

    def clear(self):
        """
        Clear all stored object memories.
        """

        self.memory.clear()