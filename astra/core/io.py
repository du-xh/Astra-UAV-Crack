# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import json
from pathlib import Path
DEPLOY_PARAMETERS = 15310826
TRAIN_PARAMETERS = 20080234

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
