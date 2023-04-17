class Monitor:
    """
    Monitor base class
    """

    def save_results(self, name, writer, step, data_dir):
        raise NotImplementedError("Subclass of Monitor needs to implement this")
