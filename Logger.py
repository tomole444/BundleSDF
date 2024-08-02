import sys
import os

class Logger():
    def __init__(self, filename, loglevel = sys.stdout):
        self.terminal = loglevel
        if(os.path.exists(filename)):
            os.remove(filename)
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.terminal.close()
        self.log.close()

if __name__ == "__main__":
    sys.stdout = Logger("/home/thws_robotik/Downloads/console.log")

    print("test")