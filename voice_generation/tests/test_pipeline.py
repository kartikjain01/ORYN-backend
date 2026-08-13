import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from text_intelligence.pipeline import TextIntelligencePipeline

pipeline = TextIntelligencePipeline()

text = "I didn’t think I would succeed, but I did!"

result = pipeline.run(text)

print("\n===== FINAL PIPELINE OUTPUT =====\n")
print(json.dumps(result, indent=2))
