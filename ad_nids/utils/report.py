
from pathlib import Path

import ad_nids

templates_path = Path(ad_nids.__path__[0])/'templates'
with open(templates_path/'base.html', 'r') as f:
    BASE = f.read()


def merge_reports(reports, sort=True):

    report_merged = ''

    if sort:
        reports = sorted(reports, key=lambda x: x[0])

    for name, report in reports:
        report_merged += f'<h1> {name} </h1></br>'
        report_merged += report
        report_merged += '</br></br>'

    report_merged = BASE.replace('{{STUFF}}', report_merged)

    return report_merged
