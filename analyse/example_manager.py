class ExampleManager:
    def __init__(self):
        # Dictionary of examples
        self.examples = {
            "1": {"name": "Example 1", "image": "assets/images/example1.jpg", "value": {"x": 275, "y": 174}},
            "2": {"name": "Example 2", "image": "assets/images/example2.jpg", "value": {"x": 431, "y": 289}},
        }

    def get_example_names(self):
        """
        Returns a list of name values of examples
        """
        names = [item["name"] for item in self.examples.values()]
        return names

    def get_example_by_name(self, name):
        """
        Find the instance of example by name property
        """
        for item in self.examples.values():
            if item["name"] == name:
                return item
        return None


example_manager = ExampleManager()
