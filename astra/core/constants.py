# Final Astra dependency. Third-party attribution: THIRD_PARTY_NOTICES.md.
import json
from pathlib import Path
TRAIN_PARAMETERS = 20047466
DEPLOY_PARAMETERS = 15310826

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)
