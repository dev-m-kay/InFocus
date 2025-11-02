import json

class settings():
    def write(settings):
        with open("settings.ini", "w") as f:
          json.dump(settings, f, indent=4)  

    def read():
        with open("settings.ini", "r") as f:
            return json.load(f)