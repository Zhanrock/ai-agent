import subprocess
import sys

# Install pysqlite3-binary
subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
