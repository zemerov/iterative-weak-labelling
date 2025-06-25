from jinja2 import Template


class CriteriaGenerator:
    def __init__(self, generate_criteria_file: str, deduplicate_criteria_file: str):
        with open(generate_criteria_file, "r") as file:
            self.generate_criteria_template = Template("\n".join(file.readlines()))


        with open(deduplicate_criteria_file, "r") as file:
            self.deduplicate_criteria_template = Template("\n".join(file.readlines()))
        
    def get_new_criteria(self):
        pass

    def deduplicate_new_criteria(self):
        pass