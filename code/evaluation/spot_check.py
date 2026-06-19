import csv

rows = list(csv.DictReader(open('dataset/output.csv', encoding='utf-8')))
print('Total rows:', len(rows))
print('valid_image values:', set(r['valid_image'] for r in rows))

cases = [
    ('case_018', 'keyboard', 'water_damage'),
    ('case_020', 'trackpad', 'crack'),
    ('case_026', 'body',     'crack'),
    ('case_046', 'side_mirror', 'broken_part'),
    ('case_008', 'hood',     'dent'),
]
for case, exp_part, exp_issue in cases:
    r = next((x for x in rows if case in x['image_paths']), None)
    if r:
        part_ok  = 'OK' if r['object_part'] == exp_part  else 'WRONG'
        issue_ok = 'OK' if r['issue_type']  == exp_issue else 'WRONG'
        print(f"{case}: part={r['object_part']}({part_ok}) issue={r['issue_type']}({issue_ok}) valid={r['valid_image']}")
