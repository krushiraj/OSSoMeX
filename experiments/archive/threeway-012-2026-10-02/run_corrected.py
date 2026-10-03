import sys
from pathlib import Path
import run
run.OUT=Path(__file__).resolve().parent/'corrected'
run.run(sys.argv[1])
