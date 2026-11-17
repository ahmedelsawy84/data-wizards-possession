# BAR-ATH possession detection - Team Data Wizards (Barcelona More Than A Hack 2027)

Rule-based, fully deterministic, no learning and no hand labels. Tested with Python 3.12.3 and numpy 2.5.4 (pinned).

## Reproduce
```bash
pip install -r requirements.txt
# put BAR-ATH_tactical_atd.txt, _atd.xml and _smart_tagging.xml in data/tactical/ (NOT committed; see .gitignore)
python possession.py            # writes "Data Wizards.csv" and runs a format check
python evaluation/validate_submission.py "Data Wizards.csv"   # organisers' validator
```
Optional: `--data-dir <folder>` and `--out <file>`. Expected md5 of the output: `ef7adb9242f653cc6c67b0f1140a0e1c`.
