import json

class settings():
    def write(settings):
        with open("settings.json", "w") as f:
          json.dump(settings, f, indent=4)  

    def read():
        with open("settings.json", "r") as f:
            return json.load(f)